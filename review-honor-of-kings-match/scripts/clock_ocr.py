#!/usr/bin/env python3
"""Recognize and stabilize an in-game HUD clock from extracted replay frames."""

from __future__ import annotations

import csv
import json
import math
import platform
import re
import shutil
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any


class ClockOCRError(RuntimeError):
    pass


ROI_DEFAULT = (0.25, 0.0, 0.5, 0.18)
TIMER_WITH_SEPARATOR = re.compile(r"(?<!\d)(\d{1,2})[:：;,.](\d{2})(?!\d)")
TIMER_WITHOUT_SEPARATOR = re.compile(r"(?<!\d)(\d{3,4})(?!\d)")
OCR_TRANSLATION = str.maketrans(
    {
        "O": "0",
        "o": "0",
        "Q": "0",
        "I": "1",
        "l": "1",
        "|": "1",
        "Z": "2",
        "S": "5",
        "s": "5",
        "B": "8",
    }
)
MAP_COLUMNS = [
    "video_time",
    "game_time",
    "raw_game_time",
    "raw_text",
    "confidence",
    "status",
    "segment",
    "source_frame",
]


def parse_roi(value: str) -> tuple[float, float, float, float]:
    try:
        values = tuple(float(item.strip()) for item in value.split(","))
    except ValueError as exc:
        raise ClockOCRError("clock ROI must contain four numbers: x,y,width,height") from exc
    if len(values) != 4:
        raise ClockOCRError("clock ROI must contain four numbers: x,y,width,height")
    x, y, width, height = values
    if min(values) < 0 or width <= 0 or height <= 0 or x + width > 1 or y + height > 1:
        raise ClockOCRError("clock ROI must fit inside normalized top-left coordinates 0..1")
    return x, y, width, height


def parse_playback_speed(value: str) -> tuple[str, float | None]:
    normalized = value.strip().lower().replace("×", "x")
    if normalized in {"variable", "var", "unknown", "变速", "未知"}:
        return value, None
    if normalized.endswith("x"):
        normalized = normalized[:-1]
    try:
        speed = float(normalized)
    except ValueError as exc:
        raise ClockOCRError("playback speed must be a positive number such as 1, 2x, or variable") from exc
    if not math.isfinite(speed) or speed <= 0:
        raise ClockOCRError("playback speed must be greater than zero")
    return value, speed


def format_game_time(seconds: float | None) -> str:
    if seconds is None:
        return ""
    total = max(0, int(round(seconds)))
    minutes, remainder = divmod(total, 60)
    return f"{minutes:02d}:{remainder:02d}"


def resolve_backend(requested: str) -> str:
    if requested not in {"auto", "vision", "tesseract"}:
        raise ClockOCRError(f"unsupported OCR backend: {requested}")
    if requested == "vision":
        if platform.system() != "Darwin" or not shutil.which("clang"):
            raise ClockOCRError("Vision OCR requires macOS and clang")
        return "vision"
    if requested == "tesseract":
        if not shutil.which("tesseract"):
            raise ClockOCRError("Tesseract OCR was requested but tesseract is not on PATH")
        return "tesseract"
    if platform.system() == "Darwin" and shutil.which("clang"):
        return "vision"
    if shutil.which("tesseract"):
        return "tesseract"
    raise ClockOCRError("no OCR backend is available; install Tesseract or run on macOS with Vision")


def run_process(command: list[str], input_text: str | None = None) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(command, input=input_text, text=True, capture_output=True)
    if completed.returncode != 0:
        details = completed.stderr.strip() or completed.stdout.strip()
        raise ClockOCRError(f"command failed ({command[0]}): {details}")
    return completed


def recognize_with_vision(
    frames: list[Path], roi: tuple[float, float, float, float]
) -> dict[str, list[dict[str, Any]]]:
    helper_source = Path(__file__).with_name("vision_text_ocr.m")
    if not helper_source.is_file():
        raise ClockOCRError(f"Vision helper is missing: {helper_source}")
    clang = shutil.which("clang")
    if not clang:
        raise ClockOCRError("clang is required for Vision OCR")
    with tempfile.TemporaryDirectory(prefix="hok-clock-vision-") as temp_dir:
        binary = Path(temp_dir) / "vision-text-ocr"
        run_process(
            [
                clang,
                "-fobjc-arc",
                f"-fmodules-cache-path={Path(temp_dir) / 'module-cache'}",
                str(helper_source),
                "-o",
                str(binary),
                "-framework",
                "Vision",
            ]
        )
        payload = "".join(f"{frame}\n" for frame in frames)
        completed = run_process([str(binary), *(f"{value:.8f}" for value in roi)], payload)
    recognized: dict[str, list[dict[str, Any]]] = {}
    errors = []
    for line in completed.stdout.splitlines():
        if not line.strip():
            continue
        result = json.loads(line)
        if result.get("error"):
            errors.append(str(result["error"]))
        recognized[str(Path(result["path"]).resolve())] = result.get("candidates", [])
    if errors and len(errors) == len(frames):
        raise ClockOCRError(
            f"Vision failed for every frame: {errors[0]}. "
            "The macOS Vision service may require running this local command outside the sandbox."
        )
    return recognized


def crop_for_tesseract(
    ffmpeg: str,
    frame: Path,
    output: Path,
    roi: tuple[float, float, float, float],
) -> None:
    x, y, width, height = roi
    video_filter = (
        f"crop=iw*{width:.8f}:ih*{height:.8f}:iw*{x:.8f}:ih*{y:.8f},"
        "scale=1600:-2:flags=lanczos,format=gray,eq=contrast=1.8:brightness=0.04"
    )
    run_process(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(frame),
            "-vf",
            video_filter,
            "-frames:v",
            "1",
            str(output),
        ]
    )


def recognize_with_tesseract(
    frames: list[Path], roi: tuple[float, float, float, float], ffmpeg: str
) -> dict[str, list[dict[str, Any]]]:
    tesseract = shutil.which("tesseract")
    if not tesseract:
        raise ClockOCRError("tesseract is required for the Tesseract OCR backend")
    recognized: dict[str, list[dict[str, Any]]] = {}
    with tempfile.TemporaryDirectory(prefix="hok-clock-tesseract-") as temp_dir:
        crop = Path(temp_dir) / "clock.png"
        for frame in frames:
            crop_for_tesseract(ffmpeg, frame, crop, roi)
            completed = run_process(
                [
                    tesseract,
                    str(crop),
                    "stdout",
                    "--psm",
                    "6",
                    "-l",
                    "eng",
                    "-c",
                    "tessedit_char_whitelist=0123456789:",
                    "tsv",
                ]
            )
            lines: dict[tuple[str, str, str], list[tuple[str, float]]] = defaultdict(list)
            for row in csv.DictReader(completed.stdout.splitlines(), delimiter="\t"):
                text = (row.get("text") or "").strip()
                try:
                    confidence = float(row.get("conf") or -1)
                except ValueError:
                    confidence = -1
                if not text or confidence < 0:
                    continue
                key = (row.get("block_num", ""), row.get("par_num", ""), row.get("line_num", ""))
                lines[key].append((text, confidence / 100))
            candidates = []
            for tokens in lines.values():
                text = "".join(token for token, _ in tokens)
                confidence = sum(score for _, score in tokens) / len(tokens)
                candidates.append({"text": text, "confidence": confidence, "boundingBox": []})
            recognized[str(frame.resolve())] = candidates
    return recognized


def parse_timer_candidate(text: str) -> tuple[float, str, float] | None:
    normalized = re.sub(r"\s+", "", text.translate(OCR_TRANSLATION))
    match = TIMER_WITH_SEPARATOR.search(normalized)
    penalty = 0.0
    if match:
        minutes, seconds = int(match.group(1)), int(match.group(2))
    else:
        match = TIMER_WITHOUT_SEPARATOR.fullmatch(normalized)
        if not match:
            return None
        digits = match.group(1)
        minutes, seconds = int(digits[:-2]), int(digits[-2:])
        penalty = 0.18
    if seconds >= 60 or minutes > 99:
        return None
    total = float(minutes * 60 + seconds)
    return total, f"{minutes:02d}:{seconds:02d}", penalty


def select_timer(candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
    selected = None
    for candidate in candidates:
        text = str(candidate.get("text", ""))
        parsed = parse_timer_candidate(text)
        if not parsed:
            continue
        seconds, timecode, penalty = parsed
        confidence = max(0.0, min(1.0, float(candidate.get("confidence", 0.0)) - penalty))
        value = {
            "raw_text": text,
            "raw_game_time_seconds": seconds,
            "raw_game_time": timecode,
            "confidence": confidence,
        }
        if selected is None or value["confidence"] > selected["confidence"]:
            selected = value
    return selected


def records_are_continuous(
    earlier: dict[str, Any], later: dict[str, Any], playback_speed: float | None
) -> bool:
    video_delta = later["video_time_seconds"] - earlier["video_time_seconds"]
    game_delta = later["raw_game_time_seconds"] - earlier["raw_game_time_seconds"]
    if video_delta <= 0:
        return abs(game_delta) <= 1
    if playback_speed is None:
        return -1 <= game_delta <= max(30.0, video_delta * 8.0)
    expected = video_delta * playback_speed
    tolerance = max(3.0, expected * 0.35 + 1.0)
    return abs(game_delta - expected) <= tolerance


def stabilize_records(
    records: list[dict[str, Any]], playback_speed: float | None, min_confidence: float
) -> None:
    candidates = [
        index
        for index, record in enumerate(records)
        if record.get("raw_game_time_seconds") is not None and record["confidence"] >= min_confidence
    ]
    previous_index: int | None = None
    segment = 0
    for candidate_position, index in enumerate(candidates):
        record = records[index]
        if previous_index is None:
            record.update(
                {
                    "game_time_seconds": record["raw_game_time_seconds"],
                    "game_time": record["raw_game_time"],
                    "status": "anchor",
                    "segment": segment,
                }
            )
            previous_index = index
            continue
        previous = records[previous_index]
        if records_are_continuous(previous, record, playback_speed):
            record.update(
                {
                    "game_time_seconds": record["raw_game_time_seconds"],
                    "game_time": record["raw_game_time"],
                    "status": "accepted",
                    "segment": segment,
                }
            )
            previous_index = index
            continue
        next_index = candidates[candidate_position + 1] if candidate_position + 1 < len(candidates) else None
        confirmed_jump = (
            next_index is not None
            and record["confidence"] >= min(1.0, min_confidence + 0.1)
            and records_are_continuous(record, records[next_index], playback_speed)
        )
        if confirmed_jump:
            segment += 1
            record.update(
                {
                    "game_time_seconds": record["raw_game_time_seconds"],
                    "game_time": record["raw_game_time"],
                    "status": "discontinuity-anchor",
                    "segment": segment,
                }
            )
            previous_index = index
        else:
            record.update({"status": "needs-review", "segment": segment})

    for record in records:
        if record.get("raw_game_time_seconds") is None:
            record.setdefault("status", "missing")
            record.setdefault("segment", None)
        elif record["confidence"] < min_confidence:
            record.setdefault("status", "low-confidence")
            record.setdefault("segment", None)

    anchor_indices = [
        index for index, record in enumerate(records) if record.get("game_time_seconds") is not None
    ]
    for left_index, right_index in zip(anchor_indices, anchor_indices[1:]):
        left, right = records[left_index], records[right_index]
        if left["segment"] != right["segment"]:
            continue
        video_span = right["video_time_seconds"] - left["video_time_seconds"]
        game_span = right["game_time_seconds"] - left["game_time_seconds"]
        if video_span <= 0 or game_span < 0:
            continue
        for index in range(left_index + 1, right_index):
            record = records[index]
            if record.get("game_time_seconds") is not None:
                continue
            ratio = (record["video_time_seconds"] - left["video_time_seconds"]) / video_span
            estimated = left["game_time_seconds"] + ratio * game_span
            record.update(
                {
                    "game_time_seconds": round(estimated, 3),
                    "game_time": format_game_time(estimated),
                    "status": "interpolated",
                    "segment": left["segment"],
                    "confidence": round(min(left["confidence"], right["confidence"]) * 0.7, 4),
                }
            )


def make_roi_preview(
    ffmpeg: str,
    frame: Path,
    output: Path,
    roi: tuple[float, float, float, float],
) -> None:
    x, y, width, height = roi
    video_filter = f"crop=iw*{width:.8f}:ih*{height:.8f}:iw*{x:.8f}:ih*{y:.8f},scale=1600:-2"
    run_process(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(frame),
            "-vf",
            video_filter,
            "-frames:v",
            "1",
            str(output),
        ]
    )


def write_clock_map(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MAP_COLUMNS, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        for record in records:
            row = dict(record)
            row["confidence"] = f"{record.get('confidence', 0.0):.4f}"
            writer.writerow(row)


def enrich_manifest_with_clock_ocr(
    manifest: dict[str, Any],
    evidence_dir: Path,
    backend: str,
    roi: tuple[float, float, float, float],
    playback_speed_label: str,
    playback_speed: float | None,
    min_confidence: float,
    ffmpeg: str,
) -> dict[str, Any]:
    resolved_backend = resolve_backend(backend)
    frame_entries = manifest.get("frames", [])
    if not frame_entries:
        raise ClockOCRError("manifest contains no frames to OCR")
    frame_paths = [(evidence_dir / entry["file"]).resolve() for entry in frame_entries]
    missing = [path for path in frame_paths if not path.is_file()]
    if missing:
        raise ClockOCRError(f"OCR source frame is missing: {missing[0]}")

    if resolved_backend == "vision":
        recognized = recognize_with_vision(frame_paths, roi)
    else:
        recognized = recognize_with_tesseract(frame_paths, roi, ffmpeg)

    grouped: dict[float, list[dict[str, Any]]] = defaultdict(list)
    for entry, frame_path in zip(frame_entries, frame_paths):
        selected = select_timer(recognized.get(str(frame_path), []))
        grouped[round(float(entry["video_time_seconds"]), 3)].append(
            {
                "video_time_seconds": float(entry["video_time_seconds"]),
                "video_time": entry["video_timecode"],
                "source_frame": entry["file"],
                **(
                    selected
                    or {
                        "raw_text": "",
                        "raw_game_time_seconds": None,
                        "raw_game_time": "",
                        "confidence": 0.0,
                    }
                ),
            }
        )
    records = []
    for video_time in sorted(grouped):
        alternatives = grouped[video_time]
        records.append(max(alternatives, key=lambda item: item["confidence"]))
    stabilize_records(records, playback_speed, min_confidence)

    map_by_time = {round(record["video_time_seconds"], 3): record for record in records}
    for entry in frame_entries:
        result = map_by_time[round(float(entry["video_time_seconds"]), 3)]
        entry.update(
            {
                "game_time_seconds": result.get("game_time_seconds"),
                "game_timecode": result.get("game_time"),
                "clock_ocr_confidence": round(float(result.get("confidence", 0.0)), 4),
                "clock_ocr_status": result["status"],
                "clock_ocr_segment": result.get("segment"),
                "clock_ocr_raw_text": result.get("raw_text", ""),
            }
        )

    map_path = evidence_dir / "clock-map.tsv"
    preview_path = evidence_dir / "clock-roi-preview.jpg"
    write_clock_map(map_path, records)
    preview_record = next(
        (record for record in records if record.get("raw_game_time_seconds") is not None), records[0]
    )
    make_roi_preview(ffmpeg, evidence_dir / preview_record["source_frame"], preview_path, roi)
    resolved = sum(record.get("game_time_seconds") is not None for record in records)
    needs_review = sum(record["status"] in {"needs-review", "low-confidence"} for record in records)
    return {
        "enabled": True,
        "backend": resolved_backend,
        "roi_normalized_top_left": {
            "x": roi[0],
            "y": roi[1],
            "width": roi[2],
            "height": roi[3],
        },
        "declared_playback_speed": playback_speed_label,
        "min_confidence": min_confidence,
        "record_count": len(records),
        "resolved_count": resolved,
        "coverage": round(resolved / len(records), 4),
        "needs_review_count": needs_review,
        "map_file": map_path.relative_to(evidence_dir).as_posix(),
        "roi_preview": preview_path.relative_to(evidence_dir).as_posix(),
    }
