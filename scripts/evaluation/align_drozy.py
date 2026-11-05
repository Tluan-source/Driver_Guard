"""Export masked DROZY alignment maps after verifying the full-decode intake evidence."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import mne
import numpy as np

from driverguard.evaluation.drozy import map_frames, map_grid
from driverguard.execution import file_sha256


def align(audit_path, settings_path, out):
    audit = json.loads(audit_path.read_text(encoding='utf-8'))
    config = json.loads(settings_path.read_text(encoding='utf-8'))
    if config['sync_threshold_volts'] != 0 or not config['require_unique_edge_per_accepted_frame']:
        raise ValueError('Unsupported sync detection policy')
    out.mkdir(parents=True, exist_ok=True)
    release = Path(audit['release_root']).resolve()
    if file_sha256(release/'README.md') != audit['readme_sha256']:
        raise ValueError('Release README changed')
    rows = []
    for session in audit['sessions']:
        for evidence in session['files'].values():
            path = Path(evidence['path']).resolve()
            if not path.is_relative_to(release) or file_sha256(path) != evidence['sha256']:
                raise ValueError('Audited source changed or escaped release root')
        sid = session['session_id']
        if 'video' not in session or session['video']['nonincreasing_pts']:
            raise ValueError('Require successfully decoded video evidence')
        raw = mne.io.read_raw_edf(session['files']['psg']['path'], preload=False, verbose='ERROR')
        sfreq = float(raw.info['sfreq'])
        times = np.loadtxt(session['files']['timestamps']['path'])[:, -1]/1000
        if len(times) != session['video']['decoded_frames']:
            raise ValueError('Video/timestamp count mismatch')
        samples, frame_ok, errors = map_frames(raw.get_data(picks=['Cam-Sync'])[0], sfreq, times,
                                               max_error_s=config['max_frame_to_sync_error_s'])
        mapped, grid_ok, reasons = map_grid(np.loadtxt(session['files']['interp_indices']['path']),
                                            times, samples, frame_ok, sfreq,
                                            max_span_s=config['max_interpolation_span_s'],
                                            max_grid_error_s=config['max_grid_origin_candidate_error_s'])
        qualified = (float(frame_ok.mean()) >= config['minimum_session_frame_coverage']
                     and float(grid_ok.mean()) >= config['minimum_session_grid_coverage'])
        target = out/f'{sid}.npz'
        # One entry per decoded video frame and per official interpolation row.
        # Excluded rows remain -1/NaN, never stale or silently interpolated EEG.
        np.savez_compressed(target, camera_experiment_s=times, psg_sample_zero_based=samples,
                            frame_valid=frame_ok, sync_residual_s=errors,
                            interpolation_psg_s=mapped, interpolation_valid=grid_ok)
        row = {'session_id': sid, 'subject': session['subject'], 'decoded_frames': len(times),
               'accepted_frames': int(frame_ok.sum()), 'frame_coverage': float(frame_ok.mean()),
               'accepted_grid_rows': int(grid_ok.sum()), 'grid_coverage': float(grid_ok.mean()),
               'accepted_max_sync_error_s': float(abs(errors[frame_ok]).max()) if frame_ok.any() else None,
               'session_qualified_for_alignment_analysis': qualified, 'rejections': reasons,
               'map': str(target.resolve()), 'map_sha256': file_sha256(target),
               'source_hashes': {k: v['sha256'] for k, v in session['files'].items()}}
        rows.append(row)
        print(json.dumps({'session': sid, 'frame_coverage': row['frame_coverage'],
                          'grid_coverage': row['grid_coverage'], 'qualified': qualified}), flush=True)
    report = {'status': 'technical_alignment_completed_with_masks_and_exclusions',
              'settings_sha256': file_sha256(settings_path), 'intake_audit_sha256': file_sha256(audit_path),
              'sessions': rows, 'sessions_audited': len(rows),
              'qualified_sessions': [r['session_id'] for r in rows if r['session_qualified_for_alignment_analysis']],
              'excluded_sessions': [r['session_id'] for r in rows if not r['session_qualified_for_alignment_analysis']],
              'physiological_alignment_certified': False, 'ready_for_eeg_supervision': False,
              'scope': 'fixed timestamp-to-edge tolerance and unique matching; rejected frames stay masked',
              'grid_origin_resolved': False,
              'limitations': ['Accepted zero crossings are algorithmic correspondences, not expert-certified triggers',
                              'Grid row origin is unresolved; both candidate origins are bounded by the configured tolerance',
                              'Do not use maps for physiological onset/teacher training without a reviewed labeling protocol']}
    (out/'summary.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit', type=Path, default=Path('outputs/week6/drozy_alignment_audit.json'))
    parser.add_argument('--settings', type=Path, default=Path('configs/training/drozy_alignment_v1.json'))
    parser.add_argument('--out', type=Path, default=Path('outputs/week6/drozy_maps'))
    args = parser.parse_args()
    report = align(args.audit, args.settings, args.out)
    print(json.dumps({k: v for k, v in report.items() if k != 'sessions'}, indent=2))


if __name__ == '__main__':
    main()
