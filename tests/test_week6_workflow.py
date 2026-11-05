import csv
import json

import numpy as np
import pytest
from fastapi.testclient import TestClient

from driverguard.evaluation.drozy import map_frames, map_grid
from scripts.evaluation.audit_week6_cohort import audit as audit_cohort
from scripts.evaluation.audit_week6_cohort import freeze as freeze_cohort
from scripts.evaluation.prepare_week6 import create_review_queue
from scripts.evaluation.review_week6 import ReviewStore, create_app, validate_rows


def test_frame_mapping_drops_missing_pulse_without_shifting_future_frames():
    signal = np.ones(1000)
    signal[100:300] = -1
    signal[400:500] = -1
    samples, valid, _ = map_frames(signal, 1000, [.1, .2, .3, .4, .5])
    assert samples.tolist() == [100, -1, 300, 400, 500]
    assert valid.tolist() == [True, False, True, True, True]


def test_duplicate_candidates_reject_both_instead_of_sharing_one_pulse():
    signal = np.ones(1000)
    signal[100:200] = -1
    samples, valid, _ = map_frames(signal, 1000, [.099, .101, .2])
    assert samples.tolist() == [-1, -1, 200]
    assert valid.tolist() == [False, False, True]


def test_flat_sync_returns_only_explicit_missing_samples():
    samples, valid, errors = map_frames(np.ones(1000), 1000, [.1, .2, .3])
    assert (samples == -1).all() and not valid.any() and np.isnan(errors).all()


def test_grid_mapping_propagates_missing_frames_and_rejects_long_gaps():
    times = (np.arange(18000)+1)/30
    samples = np.rint(times*512).astype(int)
    valid = np.ones(len(times), dtype=bool)
    valid[100] = False
    table = np.column_stack([np.arange(1, 18001), np.zeros(18000), np.ones(18000)])
    table[100] = [100, 102, .5]
    table[194:207, 0] = 195
    table[200] = [195, 207, .5]
    mapped, mask, reasons = map_grid(table, times, samples, valid, 512)
    assert mask[100]  # The bad frame is not used by the bracketing pair.
    table[101] = [101, 103, .5]
    mapped, mask, reasons = map_grid(table, times, samples, valid, 512)
    assert not mask[101] and np.isnan(mapped[101])
    assert not mask[200] and np.isnan(mapped[200])
    assert reasons['large_gap_rows'] == 1


def _review_package(tmp_path):
    import pandas as pd
    clips, predictions = [], []
    for i in range(40):
        clips.append({'clip_id': f'C{i}', 'subject': str(100+i), 'split': 'train',
                      'n_frames': 6000, 'part_index': None, 'source_sha256': 'a'*64})
        predictions.append({'clip_id': f'C{i}', 'label': i % 2, 'prediction': 1-i % 2,
                            'accepted': True, 'end_ms': 200000})
    create_review_queue({'clips': clips}, pd.DataFrame(predictions), tmp_path)
    return tmp_path


def _human_row(store, reviewer='Person A'):
    row = store.role_rows('reviewer_a')[0]
    row.update(state='alert', visibility='usable', review_status='reviewed', reviewer=reviewer,
               notes='Fixture evidence, not a real annotation')
    return row


def test_human_save_versions_old_form_and_keeps_other_role_blind(tmp_path):
    package = _review_package(tmp_path)
    store = ReviewStore(package)
    original = (package/'reviewer_a.csv').read_bytes()
    other = (package/'reviewer_b.csv').read_bytes()
    row = _human_row(store)
    store.save('reviewer_a', row['review_id'], [row])
    backups = list((package/'history/reviewer_a').glob('*.csv'))
    assert len(backups) == 1 and backups[0].read_bytes() == original
    assert (package/'reviewer_b.csv').read_bytes() == other
    assert validate_rows(store.role_rows('reviewer_a'), store.clips)[0] == 1
    assert (package/'review_audit.jsonl').is_file()


def test_invalid_review_or_same_reviewer_cannot_modify_forms(tmp_path):
    store = ReviewStore(_review_package(tmp_path))
    row = _human_row(store)
    bad = dict(row, start_s=1)
    before = (tmp_path/'reviewer_a.csv').read_bytes()
    with pytest.raises(ValueError, match='Intervals'):
        store.save('reviewer_a', row['review_id'], [bad])
    assert (tmp_path/'reviewer_a.csv').read_bytes() == before
    store.save('reviewer_a', row['review_id'], [row])
    with pytest.raises(ValueError, match='different people'):
        store.save('reviewer_b', row['review_id'], [dict(row, reviewer='person a')])
    with pytest.raises(ValueError, match='Both independent'):
        store.save('adjudication', row['review_id'], [row])


def test_review_server_auth_and_private_manifest_is_never_served(tmp_path):
    package = _review_package(tmp_path)
    tokens = {'reviewer_a': 'a', 'reviewer_b': 'b', 'adjudication': 'c'}
    app = create_app(package, tokens)
    with TestClient(app) as client:
        assert client.get('/api/reviews?token=wrong').status_code == 403
        assert client.get('/private_manifest.json').status_code == 404
        data = client.get('/api/reviews?token=a').json()
        assert len(data['ids']) == 40 and 'source_label' not in json.dumps(data)
        row = data['rows'][0]
        assert client.post('/api/reviews/'+row['review_id']+'?token=a',
                           json={'intervals': [row]}, headers={'Origin': 'https://example.com'}).status_code == 403
        assert client.post('/api/reviews/'+row['review_id']+'?token=a', json=[]).status_code == 422
        response = client.get('/export?token=a')
        assert response.status_code == 200
        assert len(list(csv.DictReader(response.text.splitlines()))) == 40


def _cohort(tmp_path):
    directory = tmp_path/'cohort'
    directory.mkdir()
    protocol = tmp_path/'protocol.json'
    protocol.write_text(json.dumps({'version': 'test', 'design': {
        'planning_target_participants': 2, 'planning_development_participants': 1,
        'planning_locked_test_participants': 1, 'split_seed': 42,
        'conditions': ['day_RGB', 'night_RGB'], 'minimum_continuous_session_seconds': 600}}))
    (directory/'sessions.csv').write_text('participant_id,session_id,split,condition,video_path,consent_id,'
                                        'uta_overlap_checked,review_status,label_file,timestamps_source,camera_device\n')
    return directory, protocol


def test_missing_cohort_remains_pending_without_inventing_participants(tmp_path):
    directory, protocol = _cohort(tmp_path)
    result = audit_cohort(directory, protocol, tmp_path/'raw')
    assert result['participants_enrolled'] == 0
    assert not result['cohort_collected'] and not result['human_labels_verified']
    with pytest.raises(ValueError, match='Recruit'):
        freeze_cohort(directory/'participants.csv', protocol, directory/'participant_split.json')


def test_cohort_freezes_people_before_collection_and_rejects_roster_changes(tmp_path):
    directory, protocol = _cohort(tmp_path)
    roster = directory/'participants.csv'
    roster.write_text('participant_id,consent_id,not_uta_confirmed\nP001,C1,true\nP002,C2,true\n')
    split = freeze_cohort(roster, protocol, directory/'participant_split.json')
    assert not set(split['development']) & set(split['locked_test'])
    assert not split['media_or_labels_opened']
    assert audit_cohort(directory, protocol, tmp_path/'raw')['status'] == 'collection_pending'
    roster.write_text(roster.read_text().replace('C1', 'NEW_CONSENT'))
    with pytest.raises(ValueError, match='changed'):
        audit_cohort(directory, protocol, tmp_path/'raw')


def test_cohort_without_consent_or_with_uta_id_cannot_be_frozen(tmp_path):
    directory, protocol = _cohort(tmp_path)
    roster = directory/'participants.csv'
    roster.write_text('participant_id,consent_id,not_uta_confirmed\nP001,,true\nP002,C2,true\n')
    with pytest.raises(ValueError, match='consent'):
        freeze_cohort(roster, protocol, directory/'participant_split.json')
    roster.write_text('participant_id,consent_id,not_uta_confirmed\nUTA_01,C1,true\nP002,C2,true\n')
    with pytest.raises(ValueError, match='pseudonymous'):
        freeze_cohort(roster, protocol, directory/'participant_split.json')
