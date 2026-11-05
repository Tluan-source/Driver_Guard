"""Cut blinded 10 Hz A2 review videos from verified raw development recordings."""
from __future__ import annotations

import argparse
import json
from fractions import Fraction
from pathlib import Path

import av
import numpy as np

from driverguard.execution import file_sha256
from scripts.data.prepare_uta_full import VIDEO_SUFFIXES, parse_video_path


def display_oriented(image, rotation):
    if not np.isfinite(rotation) or abs(rotation/90-round(rotation/90)) > 1e-6:
        raise ValueError('Review export requires a right-angle display rotation')
    return np.ascontiguousarray(np.rot90(image, k=round(rotation/90)))


def cut_interval(source, destination, start_s, end_s):
    if start_s < 0 or end_s <= start_s or destination.exists():
        raise ValueError("Require a valid interval and a new output video")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix('.part.mp4')
    if temporary.exists():
        raise FileExistsError("Partial review video exists; inspect it before retrying")
    count, max_error, display_rotation = 0, 0., None
    try:
        with av.open(str(source)) as video, av.open(str(temporary), 'w', format='mp4',
                                                   options={'movflags': '+faststart'}) as output:
            stream = video.streams.video[0]
            origin = float(stream.start_time or 0) * float(stream.time_base)
            video.seek(int((start_s + origin) / float(stream.time_base)), stream=stream, backward=True)
            encoder = None
            next_time = start_s
            for frame in video.decode(stream):
                if frame.pts is None:
                    raise ValueError("Raw review source requires presentation timestamps")
                timestamp = float(frame.pts * frame.time_base) - origin
                if timestamp < next_time - 1e-6:
                    continue
                if timestamp >= end_s - 1e-6:
                    break
                error = timestamp-next_time
                if error > .15:
                    raise ValueError("Source gap is too large for the 10 Hz review timing contract")
                max_error = max(max_error, error)
                rotation = float(getattr(frame, 'rotation', 0))
                if display_rotation is not None and display_rotation != rotation:
                    raise ValueError('Display rotation changes within the review interval')
                display_rotation = rotation
                image = display_oriented(frame.to_ndarray(format='bgr24'), rotation)
                if encoder is None:
                    encoder = output.add_stream('libx264', rate=10)
                    encoder.width, encoder.height = image.shape[1], image.shape[0]
                    encoder.pix_fmt = 'yuv420p'
                    encoder.options = {'preset': 'veryfast', 'crf': '18'}
                encoded_frame = av.VideoFrame.from_ndarray(image, format='bgr24')
                encoded_frame.pts, encoded_frame.time_base = count, Fraction(1, 10)
                for packet in encoder.encode(encoded_frame):
                    output.mux(packet)
                count += 1
                next_time = start_s + count/10
            if encoder is not None:
                for packet in encoder.encode():
                    output.mux(packet)
        if count != round((end_s-start_s)*10):
            raise ValueError("Source interval does not contain the requested complete review duration")
        temporary.replace(destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return {"frames": count, "fps": 10, "codec": "h264", "max_sample_timing_error_s": max_error,
            "source_start_s": start_s, "source_end_s": end_s,
            "source_display_rotation_degrees": display_rotation,
            "video_sha256": file_sha256(destination)}


def discover_sources(raw_root, package):
    wanted = {c['clip_id']: c for c in package['clips']}
    records = {}
    for source in raw_root.rglob('*'):
        if not source.is_file() or source.suffix.lower() not in VIDEO_SUFFIXES:
            continue
        try:
            row = parse_video_path(source.relative_to(raw_root).as_posix())
        except ValueError:
            continue
        if row['clip_id'] not in wanted:
            continue  # Never read bytes of original test or unrelated recordings.
        if row['clip_id'] in records:
            raise ValueError('Duplicate raw recording for a requested review clip')
        records[row['clip_id']] = {'clip_id': row['clip_id'], 'path': str(source.resolve()),
                                   'split': wanted[row['clip_id']]['original_split']}
    if set(records) != set(wanted):
        raise FileNotFoundError(f"Missing raw review recordings: {sorted(set(wanted)-set(records))}")
    return records


def materialize(raw_manifest, package_dir, raw_root=None):
    path = package_dir / 'private_manifest.json'
    package = json.loads(path.read_text(encoding='utf-8'))
    if raw_manifest is not None:
        raw = json.loads(raw_manifest.read_text(encoding='utf-8'))
        records = {r['clip_id']: r for r in raw['clips']}
    else:
        records = discover_sources(raw_root, package)
    for clip in package['clips']:
        record = records[clip['clip_id']]
        if record['split'] not in ('train', 'validation'):
            raise ValueError('Original test video is forbidden')
        source = Path(record['path'])
        base = raw_manifest.parent if raw_manifest is not None else raw_root
        source = source if source.is_absolute() else base / source
        if file_sha256(source) != clip['source_sha256']:
            raise ValueError('Raw recording differs from the source used for the original NPZ')
        destination = (package_dir / clip['video']).resolve()
        if not destination.is_relative_to((package_dir / 'videos').resolve()):
            raise ValueError('Review destination escapes package')
        if destination.exists():
            if not clip.get('video_sha256') or file_sha256(destination) != clip['video_sha256']:
                raise ValueError('Existing review video has no matching provenance')
        else:
            metadata = cut_interval(source, destination, clip['start_s'], clip['end_s'])
            clip.update(video_sha256=metadata['video_sha256'], extraction=metadata)
        clip['source_path'] = str(source.resolve())
        package['materialized_videos'] = sum(bool(c.get('video_sha256')) for c in package['clips'])
        package['status'] = 'videos_prepared_human_review_pending'
        path.write_text(json.dumps(package, indent=2) + '\n', encoding='utf-8')
    return package['materialized_videos']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-manifest', type=Path)
    parser.add_argument('--raw-root', type=Path, default=Path('data/raw/uta_rldd_full'))
    parser.add_argument('--package', type=Path, default=Path('outputs/week6/a2_review'))
    args = parser.parse_args()
    print(json.dumps({'materialized_videos': materialize(args.raw_manifest, args.package, args.raw_root),
                      'human_review_complete': False}))


if __name__ == '__main__':
    main()
