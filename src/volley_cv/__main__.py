"""CLI (SPEC-002 RF-7): python -m volley_cv run --clip A2  |  --video ... --start 9:05 --end 9:25."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from volley_cv.config import PROJECT_ROOT, ConfigError, load_clips, load_settings, parse_timestamp


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="volley_cv")
    sub = ap.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run", help="procesa un clip del catálogo o un tramo de video")
    src = run.add_mutually_exclusive_group(required=True)
    src.add_argument("--clip", help="id de configs/eval/clips.yaml (p. ej. A2)")
    src.add_argument("--video", help="ruta relativa al directorio de datos")
    run.add_argument("--video-config", help="id de configs/videos/ (obligatorio con --video)")
    run.add_argument("--start", default="0:00")
    run.add_argument("--end")
    run.add_argument("--out", help="directorio de salida (default: <datos>/outputs/<clip>)")
    run.add_argument("--no-video", action="store_true", help="no escribir debug.mp4")
    run.add_argument("--weights", default="yolov8m.pt", help="pesos YOLO relativos al directorio de datos")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        return _run(args)
    except ConfigError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2


def _run(args: argparse.Namespace) -> int:
    from volley_cv.adapters import ByteTrackTracker, TorchvisionEmbedder, YoloPersonDetector
    from volley_cv.court import CourtMask
    from volley_cv.pipeline import Pipeline
    from volley_cv.team import TeamClassifier
    from volley_cv.video.source import probe, read_frames
    from volley_cv.video_config import load_video_config

    data = load_settings().data_dir
    if args.clip:
        catalog = load_clips(PROJECT_ROOT / "configs" / "eval" / "clips.yaml")
        clip = next((c for c in catalog.clips if c.id == args.clip), None)
        if clip is None:
            raise ConfigError(f"clip desconocido: {args.clip}")
        path, video_id = catalog.video_path(clip, data), clip.video
        start, end, name = clip.start_s, clip.end_s, clip.id
    else:
        if not args.video_config or not args.end:
            raise ConfigError("--video requiere --video-config y --end")
        path, video_id = data / args.video, args.video_config
        start, end = parse_timestamp(args.start), parse_timestamp(args.end)
        name = f"{Path(args.video).stem}_{int(start)}_{int(end)}"
    vcfg = load_video_config(video_id)
    info = probe(path)
    out_dir = Path(args.out) if args.out else data / "outputs" / name
    pipe = Pipeline(
        detector=YoloPersonDetector(data / args.weights),
        tracker=ByteTrackTracker(frame_rate=round(info.fps)),
        embedder=TorchvisionEmbedder(),
        court=CourtMask(vcfg.court, vcfg.exclude_regions),
        teams=TeamClassifier(vcfg.teams),
    )
    summary = pipe.run(read_frames(path, start, end), out_dir, fps=info.fps, write_video=not args.no_video)
    print(
        json.dumps(
            {
                "frames": summary.frames,
                "fps": round(summary.fps, 2),
                "cuts": summary.cuts,
                "jsonl": str(summary.jsonl),
                "video": str(summary.video) if summary.video else None,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
