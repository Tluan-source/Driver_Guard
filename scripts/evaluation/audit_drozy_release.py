"""Audit the official 2018 DROZY EDF/TXT release without inventing EEG labels.

Semantics come from the downloaded release's README: interpIndices references
one-based encoded-video frame indices, with zero as an unused second index.
Video PTS is playback time; timestamps/elapsedtime is the PVT experiment clock.
These clocks must never be substituted for one another.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import av
import mne
import numpy as np

from driverguard.evaluation.alignment import fit_clock
from driverguard.execution import file_sha256


def interpolation_check(table, timestamps, n_frames):
    values, times = np.asarray(table, float), np.asarray(timestamps, float)
    if values.shape != (18000, 3) or times.shape != (n_frames,):
        raise ValueError('Require 18000 interpolation rows and one timestamp per decoded frame')
    if not np.isfinite(values).all() or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError('Nonfinite data or nonincreasing timestamps')
    a, b, weight = values.T
    if not (np.equal(a, np.floor(a)).all() and np.equal(b, np.floor(b)).all()):
        raise ValueError('Frame indices must be integers')
    if np.any((a < 1) | (a > n_frames)) or np.any((weight < 0) | (weight > 1)):
        raise ValueError('Frame index or interpolation weight out of bounds')
    unused = b == 0
    if np.any(unused & (weight != 1)) or np.any(~unused & ((b < 1) | (b > n_frames))):
        raise ValueError('Zero second index requires unit first weight; active indices must be in bounds')
    if np.any(~unused & (b < a)) or np.any(np.diff(a) < 0):
        raise ValueError('Interpolation frame indices must preserve time order')
    left, right = a.astype(int)-1, np.maximum(b.astype(int)-1, 0)
    reconstructed = weight*times[left] + (1-weight)*times[right]
    # Compare both possible grid origins explicitly; do not choose an origin
    # based on which one gives the more favorable residual.
    residuals = {str(start): float(np.max(np.abs(reconstructed-(np.arange(18000)+start)/30)))
                 for start in (0, 1)}
    return {'rows': len(values), 'index_base': 1, 'unused_second_index': 0,
            'interpolated_rows': int((~unused).sum()), 'reconstructed_time_nondecreasing':
            bool(np.all(np.diff(reconstructed) >= -1e-6)),
            'grid_origin_candidates_max_error_s': residuals,
            'both_grid_origins_within_100ms': all(r <= .1 for r in residuals.values()),
            'scope': 'timestamp consistency; PVT grid origin still requires protocol confirmation'}


def sync_check(signal, sfreq, timestamps):
    signal, times = np.asarray(signal, float), np.asarray(timestamps, float)
    if signal.ndim != 1 or not len(signal) or not np.isfinite(signal).all() or sfreq <= 0:
        raise ValueError('Invalid Cam-Sync waveform or sampling rate')
    # Physical zero-voltage crossings, both rising and falling; exact analog
    # sample differences are not frame triggers.
    edges = (np.flatnonzero(np.diff((signal > 0).astype(np.int8)))+1)/sfreq
    if not len(edges):
        return {'status': 'no_zero_crossings', 'ordinal_clock_verified': False, 'edges': 0}
    k = np.searchsorted(edges, times)
    a, b = np.clip(k-1, 0, len(edges)-1), np.clip(k, 0, len(edges)-1)
    nearest = np.where(abs(edges[a]-times) < abs(edges[b]-times), a, b)
    residual = abs(edges[nearest]-times)
    clock = None
    if len(edges) == len(times):
        clock = fit_clock(times, edges, video_duration=max(float(times[-1]), len(signal)/sfreq),
                          psg_duration=len(signal)/sfreq, tolerance_s=.02)
    verified = bool(clock and clock['within_tolerance'])
    return {'status': 'ordinal_clock_consistent' if verified else 'requires_review',
            'method': 'zero-voltage crossings, both polarities; nearest-edge errors are diagnostic only',
            'edges': len(edges), 'camera_frames': len(times),
            'duplicate_nearest_matches': len(nearest)-len(np.unique(nearest)),
            'nearest_error_s_p50': float(np.median(residual)),
            'nearest_error_s_p95': float(np.percentile(residual, 95)),
            'nearest_error_s_max': float(residual.max()),
            'ordinal_clock': clock, 'ordinal_clock_verified': verified}


def audit_release(root, out):
    out.mkdir(parents=True, exist_ok=True)
    paths = sorted((root/'videos_i8').glob('*.mp4'), key=lambda p: tuple(map(int, p.stem.split('-'))))
    if not paths:
        raise FileNotFoundError('No official DROZY videos found')
    kss = np.loadtxt(root/'KSS.txt')
    sessions = []
    for video in paths:
        session = video.stem
        files = {'video': video, 'psg': root/'psg'/f'{session}.edf',
                 'timestamps': root/'timestamps'/f'{session}.txt',
                 'interp_indices': root/'interpIndices'/f'{session}.txt'}
        row = {'session_id': session, 'subject': session.split('-')[0],
               'files': {k: {'path': str(p.resolve()), 'sha256': file_sha256(p)} for k, p in files.items()}}
        try:
            count, first, last, previous, bad_pts = 0, None, None, None, 0
            with av.open(str(video)) as container:
                stream = container.streams.video[0]
                header_frames, rate = stream.frames, str(stream.average_rate)
                for frame in container.decode(stream):
                    if frame.pts is None:
                        raise ValueError('Decoded video frame has no PTS')
                    t = float(frame.pts*frame.time_base)
                    bad_pts += int(previous is not None and t <= previous)
                    first = t if first is None else first
                    previous = last = t
                    count += 1
            timestamps = np.loadtxt(files['timestamps'])
            if timestamps.shape != (count, 8):
                raise ValueError('Decoded frame and timestamp count disagree')
            elapsed = timestamps[:, -1]/1000
            raw = mne.io.read_raw_edf(files['psg'], preload=False, verbose='ERROR')
            sfreq = float(raw.info['sfreq'])
            row['video'] = {'decoded_frames': count, 'header_frames': header_frames,
                            'encoded_fps': rate, 'pts_first_s': first, 'pts_last_s': last,
                            'nonincreasing_pts': bad_pts}
            row['experiment_clock'] = {'first_s': float(elapsed[0]), 'last_s': float(elapsed[-1]),
                                       'median_frame_interval_s': float(np.median(np.diff(elapsed)))}
            row['psg'] = {'sfreq': sfreq, 'samples': int(raw.n_times), 'duration_s': float(raw.n_times/sfreq),
                          'channels': raw.ch_names}
            row['interpolation'] = interpolation_check(np.loadtxt(files['interp_indices']), elapsed, count)
            row['cam_sync'] = sync_check(raw.get_data(picks=['Cam-Sync'])[0], sfreq, elapsed)
            subject, trial = map(int, session.split('-'))
            row['kss'] = int(kss[subject-1, trial-1])
            row['structural_checks_passed'] = (not bad_pts and sfreq == 512
                and row['interpolation']['reconstructed_time_nondecreasing']
                and row['interpolation']['both_grid_origins_within_100ms'])
            row['strict_clock_candidate'] = row['structural_checks_passed'] and row['cam_sync']['ordinal_clock_verified']
        except (ValueError, OSError, KeyError, IndexError) as exc:
            row.update(structural_checks_passed=False, strict_clock_candidate=False,
                       error=f'{type(exc).__name__}: {exc}')
        sessions.append(row)
        (out/'sessions.json').write_text(json.dumps(sessions, indent=2)+'\n', encoding='utf-8')
        print(json.dumps({'session': session, 'structural': row['structural_checks_passed'],
                          'clock_candidate': row['strict_clock_candidate'], 'error': row.get('error')}), flush=True)
    report = {'status': 'audited_with_alignment_review_pending', 'release_root': str(root.resolve()),
              'readme_sha256': file_sha256(root/'README.md'), 'sessions_verified': len(sessions),
              'subjects': len({r['subject'] for r in sessions}),
              'structural_sessions_passed': sum(r['structural_checks_passed'] for r in sessions),
              'strict_clock_candidates': [r['session_id'] for r in sessions if r['strict_clock_candidate']],
              'sessions_requiring_clock_review': [r['session_id'] for r in sessions if not r['strict_clock_candidate']],
              'alignment_verified': False, 'ready_for_eeg_supervision': False,
              'scope': 'All videos fully decoded; EDF/TXT checks are not physiological onset labels',
              'limitations': ['Cam-Sync zero-crossing detection needs waveform review for unmatched pulses',
                              'interpIndices grid origin (0 or 1 frame) needs protocol confirmation',
                              'Encoded PTS is compressed playback time, not PVT experiment time',
                              'KSS is session-level self-report; no frame-level fatigue onset certified'],
              'sessions': sessions}
    (out.parent/'drozy_alignment_audit.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('data/raw/drozy/DROZY'))
    parser.add_argument('--out', type=Path, default=Path('outputs/week6/drozy_alignment'))
    args = parser.parse_args()
    report = audit_release(args.root, args.out)
    print(json.dumps({k: v for k, v in report.items() if k != 'sessions'}, indent=2))


if __name__ == '__main__':
    main()
