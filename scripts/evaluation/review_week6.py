"""Loopback-only blinded A2 review UI; save human annotations with versioned backups."""
from __future__ import annotations

import argparse
import csv
import io
import json
import secrets
import threading
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

from driverguard.execution import file_sha256
from scripts.evaluation.prepare_week6 import FIELDS, OPTIONS

ROLES = ('reviewer_a', 'reviewer_b', 'adjudication')


def validate_rows(rows, ids):
    if {r.get('review_id') for r in rows} != set(ids):
        raise ValueError('Forms must cover exactly the selected review IDs')
    identities, complete = set(), 0
    for rid in ids:
        parts = sorted((r for r in rows if r['review_id'] == rid), key=lambda r: float(r['start_s']))
        end, done = 0., True
        for row in parts:
            start, stop = float(row['start_s']), float(row['end_s'])
            if not (abs(start-end) < 1e-6 and start < stop <= 10):
                raise ValueError('Intervals must cover [0,10) without gaps or overlap')
            if any(row.get(key) not in options for key, options in OPTIONS.items()):
                raise ValueError('Unknown annotation category')
            if row['visibility'] != 'usable' and row['state'] != 'uncertain':
                raise ValueError('Unusable visibility requires uncertain state')
            reviewed = row['review_status'] == 'reviewed'
            if reviewed:
                if not row.get('reviewer', '').strip() or not row.get('notes', '').strip():
                    raise ValueError('Reviewed intervals require reviewer identity and observed evidence')
                identities.add(row['reviewer'].strip().casefold())
            done &= reviewed
            end = stop
        if abs(end-10) > 1e-6:
            raise ValueError('Intervals must cover all ten seconds')
        complete += done
    return complete, identities


def read_rows(path):
    with path.open(encoding='utf-8', newline='') as stream:
        return list(csv.DictReader(stream))


class ReviewStore:
    def __init__(self, package):
        self.package = package.resolve()
        data = json.loads((package/'private_manifest.json').read_text(encoding='utf-8'))
        self.clips = {c['review_id']: c for c in data['clips']}
        self.lock = threading.Lock()

    def role_rows(self, role):
        if role not in ROLES:
            raise ValueError('Unknown role')
        return read_rows(self.package/f'{role}.csv')

    def save(self, role, rid, intervals):
        with self.lock:
            if rid not in self.clips or role not in ROLES or not intervals or len(intervals) > 100:
                raise ValueError('Invalid review role, clip or interval count')
            if role == 'adjudication' and any(
                    validate_rows(self.role_rows(r), self.clips)[0] != len(self.clips) for r in ROLES[:2]):
                raise ValueError('Both independent reviews must be complete before adjudication')
            existing = self.role_rows(role)
            supplied = [{key: str(row.get(key, '')) for key in FIELDS} for row in intervals]
            if any(r['review_id'] != rid or any(len(v) > 5000 for v in r.values()) for r in supplied):
                raise ValueError('Review identifier mismatch or field too long')
            candidate = [r for r in existing if r['review_id'] != rid] + supplied
            _, identities = validate_rows(candidate, self.clips)
            if role in ROLES[:2]:
                other = ROLES[1] if role == ROLES[0] else ROLES[0]
                if identities & validate_rows(self.role_rows(other), self.clips)[1]:
                    raise ValueError('Reviewers A and B must be different people')
            destination = self.package/f'{role}.csv'
            history = self.package/'history'/role
            history.mkdir(parents=True, exist_ok=True)
            digest = file_sha256(destination)
            backup = history/f'{digest}.csv'
            if not backup.exists():
                backup.write_bytes(destination.read_bytes())
            temporary = destination.with_suffix('.csv.tmp')
            with temporary.open('w', encoding='utf-8', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=FIELDS)
                writer.writeheader()
                writer.writerows(sorted(candidate, key=lambda r: (r['review_id'], float(r['start_s']))))
            temporary.replace(destination)
            event = {'time': datetime.now(timezone.utc).isoformat(), 'role': role, 'review_id': rid,
                     'previous_sha256': digest, 'current_sha256': file_sha256(destination)}
            with (self.package/'review_audit.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(event)+'\n')


def create_app(package, tokens=None):
    store = ReviewStore(package)
    tokens = tokens or {r: secrets.token_urlsafe(32) for r in ROLES}
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)

    def role(token):
        for name, value in tokens.items():
            if secrets.compare_digest(value, token):
                return name
        raise HTTPException(403, 'Invalid reviewer link')

    @app.middleware('http')
    async def local_only(request: Request, call_next):
        # Reject external host/origin even if a proxy forwards to loopback.
        host = request.headers.get('host', '').split(':')[0]
        origin = request.headers.get('origin')
        if host not in ('127.0.0.1', 'localhost', 'testserver') or (origin and origin != str(request.base_url).rstrip('/')):
            return JSONResponse({'detail': 'Local review only'}, status_code=403)
        response = await call_next(request)
        response.headers['Cache-Control'] = 'no-store'
        response.headers['Referrer-Policy'] = 'no-referrer'
        return response

    @app.get('/', response_class=HTMLResponse)
    def index():
        return (Path(__file__).parent/'review_week6.html').read_text(encoding='utf-8')

    @app.get('/api/reviews')
    def reviews(token: str):
        name = role(token)
        rows = store.role_rows(name)
        complete, _ = validate_rows(rows, store.clips)
        result = {'role': name, 'ids': list(store.clips), 'rows': rows, 'complete': complete,
                  'options': {k: sorted(v) for k, v in OPTIONS.items()}}
        if name == 'adjudication':
            references = {r: store.role_rows(r) for r in ROLES[:2]}
            both_complete = all(validate_rows(v, store.clips)[0] == len(store.clips) for v in references.values())
            result['reference_reviews'] = references if both_complete else None
        return result

    @app.post('/api/reviews/{rid}')
    async def save(rid: str, request: Request, token: str):
        name = role(token)
        if int(request.headers.get('content-length', 0)) > 500000:
            raise HTTPException(413, 'Form too large')
        try:
            data = await request.json()
            if not isinstance(data, dict) or not isinstance(data.get('intervals'), list):
                raise TypeError('Require an intervals array')
            store.save(name, rid, data['intervals'])
        except (ValueError, KeyError, TypeError) as exc:
            raise HTTPException(422, str(exc)) from exc
        return {'saved': True}

    @app.get('/video/{rid}')
    def video(rid: str, token: str):
        role(token)
        if rid not in store.clips:
            raise HTTPException(404)
        clip = store.clips[rid]
        path = (store.package/clip['video']).resolve()
        if (not path.is_relative_to(store.package/'videos') or not path.is_file()
                or file_sha256(path) != clip.get('video_sha256')):
            raise HTTPException(409, 'Video provenance mismatch')
        return FileResponse(path, media_type='video/mp4')

    @app.get('/export')
    def export(token: str):
        name = role(token)
        output = io.StringIO(newline='')
        writer = csv.DictWriter(output, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(store.role_rows(name))
        from fastapi.responses import Response
        return Response(output.getvalue(), media_type='text/csv',
                        headers={'Content-Disposition': f'attachment; filename="{name}.csv"'})

    app.state.review_tokens = tokens
    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package', type=Path, default=Path('outputs/week6/a2_review'))
    parser.add_argument('--port', type=int, default=8011)
    args = parser.parse_args()
    app = create_app(args.package)
    links = {r: f'http://127.0.0.1:{args.port}/#{token}' for r, token in app.state.review_tokens.items()}
    (args.package/'local_review_links.json').write_text(json.dumps(links, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(links, indent=2), flush=True)
    import uvicorn
    uvicorn.run(app, host='127.0.0.1', port=args.port, access_log=False)


if __name__ == '__main__':
    main()
