#!/usr/bin/env python3
"""Extract timestamped evidence frames and review templates from a replay video."""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from clock_ocr import (
    ROI_DEFAULT,
    ClockOCRError,
    enrich_manifest_with_clock_ocr,
    parse_playback_speed,
    parse_roi,
)


LEDGER_COLUMNS = [
    "video_time",
    "game_time",
    "phase",
    "event_type",
    "actors",
    "observed",
    "inferred",
    "consequence",
    "confidence",
    "follow_up",
]


def fail(message: str) -> "NoReturn":
    raise SystemExit(f"error: {message}")


def parse_timecode(value: str) -> float:
    parts = value.strip().split(":")
    if not 1 <= len(parts) <= 3:
        fail(f"invalid timecode {value!r}; use SS, MM:SS, or HH:MM:SS")
    try:
        numbers = [float(part) for part in parts]
    except ValueError:
        fail(f"invalid numeric timecode {value!r}")
    if any(number < 0 for number in numbers):
        fail(f"timecode cannot be negative: {value!r}")
    if len(numbers) == 3:
        hours, minutes, seconds = numbers
        if minutes >= 60 or seconds >= 60:
            fail(f"minutes and seconds must be below 60: {value!r}")
        return hours * 3600 + minutes * 60 + seconds
    if len(numbers) == 2:
        minutes, seconds = numbers
        if seconds >= 60:
            fail(f"seconds must be below 60: {value!r}")
        return minutes * 60 + seconds
    return numbers[0]


def format_timecode(seconds: float) -> str:
    milliseconds = int(round(seconds * 1000))
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    whole_seconds, millis = divmod(remainder, 1000)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{whole_seconds:02d}.{millis:03d}"
    return f"{minutes:02d}:{whole_seconds:02d}.{millis:03d}"


def parse_focus(value: str) -> tuple[float, float]:
    if "-" not in value:
        fail(f"invalid focus window {value!r}; use START-END, e.g. 08:10-08:35")
    start_text, end_text = value.split("-", 1)
    start = parse_timecode(start_text)
    end = parse_timecode(end_text)
    if end <= start:
        fail(f"focus window end must be after start: {value!r}")
    return start, end


def require_binary(name: str) -> str:
    path = shutil.which(name)
    if not path:
        fail(f"{name} is required but was not found on PATH")
    return path


def run(command: list[str]) -> None:
    completed = subprocess.run(command, text=True, capture_output=True)
    if completed.returncode != 0:
        details = completed.stderr.strip() or completed.stdout.strip()
        fail(f"command failed ({command[0]}): {details}")


def probe_video(ffprobe: str, video: Path) -> dict[str, Any]:
    command = [
        ffprobe,
        "-v",
        "error",
        "-show_entries",
        "format=duration,format_name:stream=index,codec_type,codec_name,width,height,r_frame_rate",
        "-of",
        "json",
        str(video),
    ]
    completed = subprocess.run(command, text=True, capture_output=True)
    if completed.returncode != 0:
        fail(f"ffprobe could not read {video}: {completed.stderr.strip()}")
    data = json.loads(completed.stdout)
    duration_text = data.get("format", {}).get("duration")
    if duration_text is None:
        fail("video duration is unavailable")
    video_stream = next(
        (stream for stream in data.get("streams", []) if stream.get("codec_type") == "video"),
        None,
    )
    if not video_stream:
        fail("input has no video stream")
    frame_rate = video_stream.get("r_frame_rate", "0/1")
    try:
        numerator, denominator = frame_rate.split("/", 1)
        fps = float(numerator) / float(denominator)
    except (ValueError, ZeroDivisionError):
        fps = None
    return {
        "duration_seconds": float(duration_text),
        "duration_timecode": format_timecode(float(duration_text)),
        "format": data.get("format", {}).get("format_name"),
        "codec": video_stream.get("codec_name"),
        "width": video_stream.get("width"),
        "height": video_stream.get("height"),
        "fps": fps,
    }


def ensure_clean_target(path: Path, force: bool) -> None:
    if path.exists() and not path.is_dir():
        fail(f"output path is not a directory: {path}")
    if path.exists() and any(path.iterdir()) and not force:
        fail(f"output directory is not empty: {path}; choose another directory or pass --force")
    path.mkdir(parents=True, exist_ok=True)


def extract_frames(
    ffmpeg: str,
    video: Path,
    target: Path,
    step: float,
    force: bool,
    start: float | None = None,
    end: float | None = None,
) -> list[Path]:
    target.mkdir(parents=True, exist_ok=True)
    if force:
        for pattern in ("frame_*.jpg", "contact_*.jpg"):
            for generated_file in target.glob(pattern):
                generated_file.unlink()
    overwrite = "-y" if force else "-n"
    command = [ffmpeg, "-hide_banner", "-loglevel", "error", overwrite]
    if start is not None:
        command.extend(["-ss", f"{start:.3f}"])
    command.extend(["-i", str(video)])
    if start is not None and end is not None:
        command.extend(["-t", f"{end - start:.3f}"])
    command.extend(
        [
            "-vf",
            f"fps=1/{step:.6f}:round=down",
            "-q:v",
            "3",
            str(target / "frame_%05d.jpg"),
        ]
    )
    run(command)
    return sorted(target.glob("frame_*.jpg"))


def create_contact_sheets(
    ffmpeg: str,
    frame_dir: Path,
    frame_count: int,
    columns: int,
    rows: int,
    force: bool,
) -> list[Path]:
    if frame_count == 0:
        return []
    overwrite = "-y" if force else "-n"
    sheet_count = math.ceil(frame_count / (columns * rows))
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        overwrite,
        "-framerate",
        "1",
        "-start_number",
        "1",
        "-i",
        str(frame_dir / "frame_%05d.jpg"),
        "-vf",
        f"scale=480:-2,tile={columns}x{rows}:padding=6:margin=6:color=white",
        "-frames:v",
        str(sheet_count),
        "-fps_mode",
        "vfr",
        "-q:v",
        "3",
        str(frame_dir / "contact_%03d.jpg"),
    ]
    run(command)
    return sorted(frame_dir.glob("contact_*.jpg"))


def make_frame_records(
    files: list[Path],
    output: Path,
    kind: str,
    step: float,
    start: float = 0.0,
) -> list[dict[str, Any]]:
    records = []
    for index, path in enumerate(files):
        timestamp = start + index * step
        records.append(
            {
                "file": path.relative_to(output).as_posix(),
                "kind": kind,
                "video_time_seconds": round(timestamp, 3),
                "video_timecode": format_timecode(timestamp),
            }
        )
    return records


def make_contact_records(
    sheets: list[Path],
    frames: list[dict[str, Any]],
    output: Path,
    page_size: int,
) -> list[dict[str, Any]]:
    records = []
    for index, sheet in enumerate(sheets):
        members = frames[index * page_size : (index + 1) * page_size]
        records.append(
            {
                "file": sheet.relative_to(output).as_posix(),
                "members": [member["file"] for member in members],
                "video_time_range": [
                    members[0]["video_timecode"],
                    members[-1]["video_timecode"],
                ]
                if members
                else None,
            }
        )
    return records


def write_ledger(path: Path) -> None:
    if path.exists():
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(LEDGER_COLUMNS)


def write_index(path: Path, manifest: dict[str, Any]) -> None:
    groups: dict[str, list[dict[str, Any]]] = {}
    for frame in manifest["frames"]:
        groups.setdefault(frame["kind"], []).append(frame)
    sections = []
    contact_cards = []
    for sheet in manifest["contact_sheets"]:
        file_name = html.escape(sheet["file"])
        time_range = sheet.get("video_time_range") or ["未知", "未知"]
        label = html.escape(f"{time_range[0]} – {time_range[1]}")
        contact_cards.append(
            f'<figure><a href="{file_name}"><img loading="lazy" src="{file_name}" '
            f'alt="{label}"></a><figcaption>{label}</figcaption></figure>'
        )
    if contact_cards:
        sections.append(f"<h2>联系表</h2><div class=grid>{''.join(contact_cards)}</div>")
    for group_name, frames in groups.items():
        cards = []
        for frame in frames:
            file_name = html.escape(frame["file"])
            video_timecode = frame["video_timecode"]
            game_timecode = frame.get("game_timecode")
            timecode = html.escape(
                f"视频 {video_timecode} · 游戏 {game_timecode}"
                if game_timecode
                else f"视频 {video_timecode}"
            )
            cards.append(
                f'<figure><a href="{file_name}"><img loading="lazy" src="{file_name}" '
                f'alt="{timecode}"></a><figcaption>{timecode}</figcaption></figure>'
            )
        sections.append(f"<h2>{html.escape(group_name)}</h2><div class=grid>{''.join(cards)}</div>")
    document = f"""<!doctype html>
<html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>王者荣耀录像证据索引</title>
<style>
body{{font:15px system-ui;margin:24px;background:#111;color:#eee}}h1,h2{{margin-top:28px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(260px,1fr));gap:12px}}
figure{{margin:0;background:#222;padding:8px;border-radius:8px}}img{{display:block;width:100%;height:auto}}
figcaption{{margin-top:6px;color:#bbb;font-variant-numeric:tabular-nums}}
</style><body><h1>王者荣耀录像证据索引</h1>
<p>标签优先显示 OCR 局内时间；未识别时仅显示视频时间。以 clock-map.tsv 的状态和置信度为准。</p>{''.join(sections)}</body></html>"""
    path.write_text(document, encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract overview/focus frames, contact sheets, and an event ledger from a replay video."
    )
    parser.add_argument("video", type=Path, help="input replay video")
    parser.add_argument("--output", type=Path, required=True, help="evidence output directory")
    parser.add_argument(
        "--playback-speed",
        help="declared replay speed, such as 1, 2x, 4x, or variable; required with --clock-ocr",
    )
    parser.add_argument(
        "--interval",
        type=float,
        help="overview sampling interval in seconds (default: adaptive, at most --max-overview-frames)",
    )
    parser.add_argument(
        "--max-overview-frames",
        type=int,
        default=180,
        help="target maximum number of overview frames when --interval is omitted (default: 180)",
    )
    parser.add_argument(
        "--focus",
        action="append",
        default=[],
        metavar="START-END",
        help="dense video-time window, repeatable; example: 08:10-08:35",
    )
    parser.add_argument(
        "--focus-step",
        type=float,
        default=1.0,
        help="seconds between frames inside focus windows (default: 1.0)",
    )
    parser.add_argument("--columns", type=int, default=4, help="contact-sheet columns (default: 4)")
    parser.add_argument("--rows", type=int, default=4, help="contact-sheet rows (default: 4)")
    parser.add_argument("--no-contact-sheets", action="store_true", help="skip contact-sheet generation")
    parser.add_argument(
        "--clock-ocr",
        action="store_true",
        help="recognize the in-game HUD clock and generate clock-map.tsv",
    )
    parser.add_argument(
        "--clock-roi",
        default=",".join(str(value) for value in ROI_DEFAULT),
        metavar="X,Y,W,H",
        help="normalized top-left HUD clock region (default: top-center 0.25,0,0.5,0.18)",
    )
    parser.add_argument(
        "--clock-ocr-backend",
        choices=("auto", "vision", "tesseract"),
        default="auto",
        help="OCR engine (default: macOS Vision, otherwise Tesseract)",
    )
    parser.add_argument(
        "--clock-min-confidence",
        type=float,
        default=0.45,
        help="minimum raw OCR confidence for a time anchor (default: 0.45)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="overwrite generated files, but preserve an existing event-ledger.tsv",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    video = args.video.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if not video.is_file():
        fail(f"input video does not exist: {video}")
    if args.interval is not None and args.interval <= 0:
        fail("--interval must be greater than zero")
    if args.focus_step <= 0:
        fail("--focus-step must be greater than zero")
    if args.max_overview_frames <= 0:
        fail("--max-overview-frames must be greater than zero")
    if args.columns <= 0 or args.rows <= 0:
        fail("--columns and --rows must be greater than zero")
    if not 0 <= args.clock_min_confidence <= 1:
        fail("--clock-min-confidence must be between zero and one")
    if args.clock_ocr and not args.playback_speed:
        fail("--clock-ocr requires --playback-speed (for example 1, 2x, or variable)")
    try:
        playback_speed_label, playback_speed = parse_playback_speed(
            args.playback_speed or "unknown"
        )
        clock_roi = parse_roi(args.clock_roi)
    except ClockOCRError as exc:
        fail(str(exc))

    ffmpeg = require_binary("ffmpeg")
    ffprobe = require_binary("ffprobe")
    ensure_clean_target(output, args.force)
    metadata = probe_video(ffprobe, video)
    duration = metadata["duration_seconds"]
    raw_interval = args.interval or max(2.0, duration / args.max_overview_frames)
    interval = math.ceil(raw_interval * 2) / 2
    focus_windows = []
    for focus_text in args.focus:
        start, end = parse_focus(focus_text)
        if start >= duration:
            fail(f"focus window starts after the video ends: {focus_text!r}")
        focus_windows.append((focus_text, start, min(end, duration)))

    overview_dir = output / "overview"
    overview_files = extract_frames(ffmpeg, video, overview_dir, interval, args.force)
    frames = make_frame_records(overview_files, output, "overview", interval)
    contacts: list[dict[str, Any]] = []
    page_size = args.columns * args.rows
    if not args.no_contact_sheets:
        sheets = create_contact_sheets(
            ffmpeg, overview_dir, len(overview_files), args.columns, args.rows, args.force
        )
        contacts.extend(make_contact_records(sheets, frames, output, page_size))

    focus_settings = []
    for focus_index, (focus_text, start, end) in enumerate(focus_windows, start=1):
        focus_dir = output / f"focus-{focus_index:02d}-{int(start):06d}-{int(end):06d}"
        focus_files = extract_frames(
            ffmpeg, video, focus_dir, args.focus_step, args.force, start=start, end=end
        )
        focus_frames = make_frame_records(
            focus_files, output, f"focus-{focus_index:02d}", args.focus_step, start
        )
        frames.extend(focus_frames)
        if not args.no_contact_sheets:
            sheets = create_contact_sheets(
                ffmpeg, focus_dir, len(focus_files), args.columns, args.rows, args.force
            )
            contacts.extend(make_contact_records(sheets, focus_frames, output, page_size))
        focus_settings.append(
            {
                "requested": focus_text,
                "start_seconds": start,
                "end_seconds": end,
                "step_seconds": args.focus_step,
            }
        )

    manifest = {
        "schema_version": 2,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": {"path": str(video), **metadata},
        "sampling": {
            "declared_playback_speed": playback_speed_label,
            "overview_interval_seconds": interval,
            "focus_windows": focus_settings,
            "contact_sheet_grid": None if args.no_contact_sheets else [args.columns, args.rows],
        },
        "frames": frames,
        "contact_sheets": contacts,
        "notes": [
            "All generated timestamps are source-video times, not in-game HUD times.",
            "Use contact sheets for navigation only; inspect dense frames or video for mechanics.",
            "An existing event-ledger.tsv is never overwritten.",
        ],
    }
    if args.clock_ocr:
        try:
            manifest["clock_ocr"] = enrich_manifest_with_clock_ocr(
                manifest=manifest,
                evidence_dir=output,
                backend=args.clock_ocr_backend,
                roi=clock_roi,
                playback_speed_label=playback_speed_label,
                playback_speed=playback_speed,
                min_confidence=args.clock_min_confidence,
                ffmpeg=ffmpeg,
            )
        except ClockOCRError as exc:
            fail(f"clock OCR failed: {exc}")
        manifest["notes"].append(
            "OCR game times are evidence anchors, not guaranteed facts; review low-confidence and discontinuity rows in clock-map.tsv."
        )
    else:
        manifest["clock_ocr"] = {"enabled": False}
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_ledger(output / "event-ledger.tsv")
    write_index(output / "index.html", manifest)
    print(
        json.dumps(
            {
                "output": str(output),
                "duration": metadata["duration_timecode"],
                "overview_interval_seconds": interval,
                "frame_count": len(frames),
                "contact_sheet_count": len(contacts),
                "clock_ocr": manifest["clock_ocr"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
