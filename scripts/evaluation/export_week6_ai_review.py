"""Export explicitly provisional AI observations; never modify human forms or training labels."""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from driverguard.execution import file_sha256
from scripts.evaluation.prepare_week6 import FIELDS
from scripts.evaluation.review_week6 import validate_rows


def export(observations_path, package, out):
    observations = json.loads(observations_path.read_text(encoding='utf-8'))
    clips = {c['review_id']: c for c in json.loads((package/'private_manifest.json').read_text(encoding='utf-8'))['clips']}
    if len(observations) != 40 or {r['id'] for r in observations} != set(clips):
        raise ValueError('Require an explicit visual observation for each of the forty clips')
    rows, evidence = [], []
    for item in observations:
        rid = item['id']
        video, sheet = package/clips[rid]['video'], package/'ai_evidence'/f'{rid}.jpg'
        if file_sha256(video) != clips[rid]['video_sha256'] or not sheet.is_file():
            raise ValueError('AI observation does not have matching crop/evidence provenance')
        base = {key: item[key] for key in FIELDS if key in item}
        base.update(review_id=rid, reviewer='AI_assistant_v1', review_status='reviewed')
        # Explicit changes in visibility/interpretation seen in this visual pass.
        intervals = [(0, 10, {})]
        if rid == 'R003':
            intervals = [(0, 4.8, {'state': 'alert', 'eye_closure': 'blink'}), (4.8, 6, {}),
                         (6, 10, {'state': 'alert', 'eye_closure': 'none'})]
        elif rid == 'R037':
            intervals = [(0, 1.7, {'visibility': 'usable'}), (1.7, 10, {})]
        elif rid == 'R038':
            intervals = [(0, 5, {'state': 'alert', 'visibility': 'usable', 'yawn': 'absent'}),
                         (5, 6, {}), (6, 10, {'state': 'alert', 'visibility': 'usable', 'yawn': 'absent'})]
        for start, stop, overrides in intervals:
            rows.append({**base, 'start_s': start, 'end_s': stop, **overrides})
        evidence.append({'review_id': rid, 'video_sha256': file_sha256(video),
                         'contact_sheet_sha256': file_sha256(sheet),
                         'review_method': 'visual inspection of all 100 decoded frames, arranged chronologically at 10 Hz'})
    complete, _ = validate_rows(rows, clips)
    duration = Counter()
    for row in rows:
        duration[row['state']] += float(row['end_s'])-float(row['start_s'])
    counts = Counter(item['state'] for item in observations)
    report = {'version': 'week6_ai_visual_review_20261105_v1', 'review_kind': 'AI_provisional',
              'clips_reviewed': complete, 'intervals': len(rows), 'clip_level_states': dict(counts),
              'interval_duration_s': dict(duration), 'independent_human_reviews': 0,
              'adjudicated': False, 'ready_for_training_labels': False,
              'observations_sha256': file_sha256(observations_path), 'evidence': evidence,
              'limitations': ['Contact sheets of decoded frames; not a continuous-playback human review',
                              'One AI review; does not count as two independent people or adjudication',
                              'Assistant had implementation context; not certified as fully blinded',
                              'Face landmarks were used only to position viewing crops, not to create labels',
                              'Only 10-second clips were visually reviewed; longer temporal context is still needed for uncertain cases',
                              'No physiological onset inferred and no original training label replaced']}
    out.mkdir(parents=True, exist_ok=True)
    target = out/'summary.json'
    if target.exists() and json.loads(target.read_text(encoding='utf-8')) != report:
        raise FileExistsError('AI review provenance changed; export a new version directory')
    with (out/'ai_review.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    target.write_text(json.dumps(report, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package', type=Path, default=Path('outputs/week6/a2_review'))
    parser.add_argument('--observations', type=Path, default=Path('outputs/week6/a2_review/ai_observations.json'))
    parser.add_argument('--out', type=Path, default=Path('outputs/week6/a2_review/ai_review'))
    args = parser.parse_args()
    report = export(args.observations, args.package, args.out)
    print(json.dumps({k: v for k, v in report.items() if k != 'evidence'}, indent=2))


if __name__ == '__main__':
    main()
