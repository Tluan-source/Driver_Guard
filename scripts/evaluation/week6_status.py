"""Summarize week-6 evidence without promoting missing data to completed work."""
from __future__ import annotations

import json
from pathlib import Path

from driverguard.execution import file_sha256
from scripts.evaluation.audit_drozy_alignment import audit
from scripts.evaluation.audit_week6_cohort import audit as audit_cohort
from scripts.evaluation.prepare_week6 import review_progress


def read(path):
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def main():
    root = Path('outputs/week6')
    config_path = Path('configs/training/week6_ablation_v1.json')
    settings = read(config_path)
    plan = read(root / 'ablation/plan.json')
    smoke = read(Path('outputs/smoke/week6_v1/smoke_summary.json')) or []
    if plan and plan['settings_sha256'] != file_sha256(config_path):
        raise ValueError('Prepared ablation plan no longer matches locked settings')
    smoke_verified = len(smoke) == len(settings['variants'])
    for result in smoke:
        path = Path('outputs/smoke/week6_v1') / f"smoke_{result['variant']}" / 'inference/model.pt'
        smoke_verified &= result['status'] == 'completed' and file_sha256(path) == result['checkpoint_sha256']
    review = review_progress(root / 'a2_review')
    ai_review = read(root / 'a2_review/ai_review/summary.json')
    if ai_review:
        if ai_review.get('review_kind') != 'AI_provisional' or ai_review.get('ready_for_training_labels'):
            raise ValueError('AI observations must remain explicitly provisional')
        package = read(root / 'a2_review/private_manifest.json')
        clips = {c['review_id']: c for c in package['clips']}
        for evidence in ai_review['evidence']:
            rid = evidence['review_id']
            if (file_sha256(root / 'a2_review' / clips[rid]['video']) != evidence['video_sha256']
                    or file_sha256(root / 'a2_review/ai_evidence' / f'{rid}.jpg') != evidence['contact_sheet_sha256']):
                raise ValueError('AI review video or visual evidence changed')
    drozy = read(root / 'drozy_alignment_audit.json')
    if drozy and drozy.get('release_root'):
        release_root = Path(drozy['release_root'])
        if file_sha256(release_root / 'README.md') != drozy['readme_sha256']:
            raise ValueError('DROZY release no longer matches the audited README')
        for session in drozy['sessions']:
            for evidence in session['files'].values():
                source = Path(evidence['path']).resolve()
                if not source.is_relative_to(release_root.resolve()) or file_sha256(source) != evidence['sha256']:
                    raise ValueError('DROZY audited source changed or escaped its release root')
    else:
        drozy = audit(Path('data/raw/drozy'))
    maps = read(root / 'drozy_maps/summary.json')
    if maps:
        if (maps['intake_audit_sha256'] != file_sha256(root / 'drozy_alignment_audit.json')
                or maps['settings_sha256'] != file_sha256(Path('configs/training/drozy_alignment_v1.json'))):
            raise ValueError('DROZY mapping no longer matches intake evidence or settings')
        for session in maps['sessions']:
            path = Path(session['map']).resolve()
            if not path.is_relative_to((root / 'drozy_maps').resolve()) or file_sha256(path) != session['map_sha256']:
                raise ValueError('DROZY map provenance changed')
    cohort = read(Path('configs/training/week6_cohort_protocol_v1.json'))
    cohort_evidence = audit_cohort(root / 'cohort', Path('configs/training/week6_cohort_protocol_v1.json'),
                                   Path('data/raw/week6_cohort'))
    preparation = read(root / 'preparation.json')
    report = {
        'source': 'https://docs.google.com/spreadsheets/d/1xTmUC8TSeNlwzOtknd8n4HYEc-2OISQRW6JcvNxYzI8/edit?gid=911502963#gid=911502963',
        'source_rows': [34, 35, 36, 37, 38],
        'work': [
            {'sheet_row': 34, 'item': 'Error analysis and locked hypotheses', 'status': 'completed',
             'feature_statistics_rows': preparation['feature_diagnostics']['feature_statistics_rows'],
             'locked_hypotheses': len(settings['hypotheses']), 'settings_sha256': file_sha256(config_path)},
            {'sheet_row': 35, 'item': 'A2 review and adjudication',
             'status': ('ai_review_complete_human_confirmation_pending' if ai_review
                        and not review['ready_for_training_labels'] else review['status']),
             'evidence': review,
             'ai_review': {k: v for k, v in ai_review.items() if k != 'evidence'} if ai_review else None},
            {'sheet_row': 36, 'item': 'DROZY camera/PSG alignment',
             'status': maps['status'] if maps else drozy['status'],
             'tool_ready': True, 'alignment_verified': drozy.get('alignment_verified', False),
             'sessions_audited': drozy.get('sessions_verified', 0),
             'structural_sessions_passed': drozy.get('structural_sessions_passed', 0),
             'strict_clock_candidates': drozy.get('strict_clock_candidates', []),
             'qualified_sessions_for_masked_analysis': len(maps['qualified_sessions']) if maps else 0,
             'excluded_sessions': maps['excluded_sessions'] if maps else [],
             'ready_for_eeg_supervision': False},
            {'sheet_row': 37, 'item': 'Prepare ablation experiments',
             'status': 'prepared_smoke_verified' if smoke_verified and plan else 'incomplete',
             'new_gpu_jobs_prepared': len(plan['jobs']) if plan else 0, 'cpu_smoke_variants_verified': len(smoke),
             'full_ablation_cv_executed': False, 'smoke_is_research_quality_evidence': False},
            {'sheet_row': 38, 'item': 'Device, labeled day/night sessions and independent cohort',
             'status': 'device_verified_protocol_prepared_collection_pending', 'device': cohort['device'],
             'cohort_collected': cohort_evidence['cohort_collected'],
             'collection_audit': cohort_evidence, 'day_night_labels_verified': False,
             'false_alert_rate_acceptance_cap_locked': False}],
        'week6_fully_complete': False, 'independent_acceptance': False, 'original_test_opened': False,
        'remaining_inputs': ([ 'raw UTA recordings for A2 crops'] if review['verified_videos'] < 40 else [])
                            + ([] if review['ready_for_training_labels'] else ['independent human reviews and adjudication'])
                            + ([] if maps else ['DROZY alignment maps'])
                            + ['new independent labeled day/night sessions'],
        'timestamp_fix': {'observed_duplicate_ms': 29547, 'consecutive_duplicate_frames': 1,
                          'cause': 'integer millisecond camera timestamps collided, resetting runtime history',
                          'fix': 'strictly increasing capture timestamps; real long gaps preserved'},
        'post_fix_webcam': read(root / 'webcam_after_fix/summary.json')}
    (root / 'status.json').write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
