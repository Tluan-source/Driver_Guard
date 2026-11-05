"""Freeze consented independent participants before media intake; audit collection metadata.

No video or label payload is opened, including locked test recordings. Consent
and non-UTA membership are operator attestations, not independently proven here.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import re
from pathlib import Path

from driverguard.execution import file_sha256

FIELDS = ['participant_id', 'consent_id', 'not_uta_confirmed']


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def freeze(roster_path, protocol_path, destination):
    if destination.exists():
        raise FileExistsError('Participant split is immutable; use a new version for changes')
    participants = read_csv(roster_path)
    protocol = json.loads(protocol_path.read_text(encoding='utf-8'))
    design = protocol['design']
    if len(participants) != design['planning_target_participants']:
        raise ValueError('Recruit the declared number of participants before freezing the split')
    ids = [r['participant_id'].strip() for r in participants]
    if len(set(ids)) != len(ids) or any(not re.fullmatch(r'P[0-9]{3}', p) for p in ids):
        raise ValueError('Use unique pseudonymous IDs P001..P999, never names or UTA identifiers')
    if any(not r['consent_id'].strip() or r['not_uta_confirmed'].strip().lower() != 'true' for r in participants):
        raise ValueError('Every participant needs recorded consent and an explicit non-UTA attestation')
    if design['planning_development_participants'] + design['planning_locked_test_participants'] != len(ids):
        raise ValueError('Protocol split counts do not add up')
    shuffled = sorted(ids)
    random.Random(design['split_seed']).shuffle(shuffled)
    n = design['planning_development_participants']
    result = {'version': protocol['version'], 'roster_sha256': file_sha256(roster_path),
              'protocol_sha256': file_sha256(protocol_path), 'split_unit': 'participant',
              'development': sorted(shuffled[:n]), 'locked_test': sorted(shuffled[n:]),
              'independence': 'operator attested; participant identities not independently verified',
              'media_or_labels_opened': False}
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    return result


def audit(directory, protocol_path, media_root):
    directory.mkdir(parents=True, exist_ok=True)
    roster = directory/'participants.csv'
    if not roster.exists():
        roster.write_text(','.join(FIELDS)+'\n', encoding='utf-8')
    protocol = json.loads(protocol_path.read_text(encoding='utf-8'))
    design = protocol['design']
    split_path = directory/'participant_split.json'
    base = {'participants_enrolled': len(read_csv(roster)), 'split_frozen': split_path.is_file(),
            'sessions_metadata_valid': 0, 'cohort_collected': False, 'human_labels_verified': False,
            'independent_acceptance': False, 'media_or_labels_opened': False,
            'minimum_continuous_session_s': design['minimum_continuous_session_seconds'],
            'conditions': design['conditions'], 'issues': []}
    if not split_path.is_file():
        return {**base, 'status': 'recruitment_and_split_freeze_pending'}
    split = json.loads(split_path.read_text(encoding='utf-8'))
    if split['roster_sha256'] != file_sha256(roster) or split['protocol_sha256'] != file_sha256(protocol_path):
        raise ValueError('Roster or protocol changed after participant split freeze')
    if set(split['development']) & set(split['locked_test']):
        raise ValueError('Participant leakage')
    people = {p: kind for kind in ('development', 'locked_test') for p in split[kind]}
    consent = {r['participant_id']: r['consent_id'] for r in read_csv(roster)}
    rows = read_csv(directory/'sessions.csv')
    seen, observed = set(), set()
    for row in rows:
        sid, person, condition = row['session_id'], row['participant_id'], row['condition']
        if sid in seen or person not in people or condition not in design['conditions']:
            raise ValueError('Duplicate session or unregistered participant/condition')
        seen.add(sid)
        expected_split = {'development': 'train', 'locked_test': 'test'}[people[person]]
        if row['split'] != expected_split or row['consent_id'] != consent[person]:
            raise ValueError('Session split or consent does not match frozen enrollment')
        if row['uta_overlap_checked'].strip().lower() != 'true':
            raise ValueError('Missing participant independence attestation')
        video = Path(row['video_path']).resolve()
        labels = Path(row['label_file']).resolve()
        # stat/presence only; model developers must not read held-out payloads.
        if not video.is_relative_to(media_root.resolve()) or not labels.is_relative_to(media_root.resolve()):
            raise ValueError('Cohort paths must remain inside the controlled media root')
        if not video.is_file() or not labels.is_file() or row['review_status'] != 'adjudicated':
            base['issues'].append({'session_id': sid, 'issue': 'video or adjudicated labels missing'})
            continue
        if not row['timestamps_source'].strip() or not row['camera_device'].strip():
            base['issues'].append({'session_id': sid, 'issue': 'device or timing provenance missing'})
            continue
        observed.add((person, condition))
        base['sessions_metadata_valid'] += 1
    expected = {(person, condition) for person in people for condition in design['conditions']}
    base['missing_participant_conditions'] = [list(p) for p in sorted(expected-observed)]
    base['cohort_collected'] = expected <= observed
    base['status'] = 'metadata_complete_independent_audit_pending' if base['cohort_collected'] else 'collection_pending'
    base['scope'] = 'metadata/attestations only; duration, content and human labels require an independent auditor'
    return base


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, default=Path('outputs/week6/cohort'))
    parser.add_argument('--protocol', type=Path, default=Path('configs/training/week6_cohort_protocol_v1.json'))
    parser.add_argument('--media-root', type=Path, default=Path('data/raw/week6_cohort'))
    parser.add_argument('--freeze', action='store_true')
    args = parser.parse_args()
    if args.freeze:
        freeze(args.directory/'participants.csv', args.protocol, args.directory/'participant_split.json')
    report = audit(args.directory, args.protocol, args.media_root)
    (args.directory/'audit.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
