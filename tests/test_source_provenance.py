import json
import subprocess

from driverguard.execution import source_provenance


def test_restored_build_info_cannot_override_current_checkout_commit(tmp_path):
    subprocess.run(['git', 'init', str(tmp_path)], check=True, capture_output=True)
    (tmp_path/'example.py').write_text('print("example")\n')
    subprocess.run(['git', '-C', str(tmp_path), 'add', 'example.py'], check=True, capture_output=True)
    subprocess.run(['git', '-C', str(tmp_path), '-c', 'user.name=Test', '-c', 'user.email=test@example.invalid',
                    'commit', '-m', 'fixture'], check=True, capture_output=True)
    expected = subprocess.check_output(['git', '-C', str(tmp_path), 'rev-parse', 'HEAD'], text=True).strip()
    (tmp_path/'build_info.json').write_text(json.dumps({'git_sha': 'stale-imported-commit'}))
    result = source_provenance(tmp_path)
    assert result['git_sha'] == expected and result['provenance_source'] == 'git'


def test_exported_source_under_another_checkout_keeps_its_own_provenance(tmp_path):
    subprocess.run(['git', 'init', str(tmp_path)], check=True, capture_output=True)
    exported = tmp_path/'export'
    exported.mkdir()
    info = {'git_sha': 'exported-source-commit', 'provenance_source': 'export'}
    (exported/'build_info.json').write_text(json.dumps(info))
    assert source_provenance(exported) == info
