from __future__ import annotations

import json
import subprocess
from pathlib import Path

import cv2
import numpy as np
from rosbags.highlevel import AnyReader

CAMERA_TOPIC = "/camera_front/image_raw/compressed"


def convert_to_h264(video_path: Path) -> None:
    temporary = video_path.with_name(f"{video_path.stem}_tmp{video_path.suffix}")
    command = [
        "ffmpeg", "-y", "-i", str(video_path),
        "-c:v", "libx264", "-tag:v", "avc1",
        "-profile:v", "high", "-level", "4.0",
        "-pix_fmt", "yuv420p", "-crf", "20",
        "-preset", "fast", "-an", str(temporary),
    ]
    print('[INFO] [ROUTE/VIDEO] Converting MP4 video to H.264...', flush=True)
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
        temporary.replace(video_path)
        print('[INFO] [ROUTE/VIDEO] H.264 conversion completed.', flush=True)
    except subprocess.CalledProcessError as error:
        raise RuntimeError(f"FFmpeg conversion failed: {error.stderr}") from error
    finally:
        temporary.unlink(missing_ok=True)


def extract_video() -> Path:
    project_root = Path(__file__).resolve().parents[2]
    with (project_root / "config.json").open(encoding="utf-8") as file:
        config = json.load(file)

    destination = project_root / config["data_dir"] / config["route_mp4"]
    destination.parent.mkdir(parents=True, exist_ok=True)
    source_dir = project_root / config["others_dir"]
    mcaps = sorted(source_dir.glob("*.mcap"))
    if len(mcaps) != 1:
        raise RuntimeError(f"Expected exactly one MCAP in {source_dir}, found {len(mcaps)}")
    mcap = mcaps[0]
    print(f"[INFO] [ROUTE/VIDEO] MCAP source located: {mcap.name}", flush=True)
    if not mcap.is_file():
        raise FileNotFoundError(f"MCAP file not found: {mcap}")

    print('[INFO] [ROUTE/VIDEO] Starting video extraction...', flush=True)
    writer = None
    frame_count = 0
    try:
        with AnyReader([mcap]) as reader:
            connections = [c for c in reader.connections if c.topic == CAMERA_TOPIC]
            if not connections:
                raise RuntimeError(f"Topic not found: {CAMERA_TOPIC}")

            print('[INFO] [ROUTE/VIDEO] Camera topic located.', flush=True)
            for connection, _, rawdata in reader.messages(connections=connections):
                message = reader.deserialize(rawdata, connection.msgtype)
                frame = cv2.imdecode(np.frombuffer(message.data, dtype=np.uint8), cv2.IMREAD_COLOR)
                if frame is None:
                    continue
                if writer is None:
                    height, width = frame.shape[:2]
                    print(f"[INFO] [ROUTE/VIDEO] Video resolution: {width}x{height}", flush=True)
                    writer = cv2.VideoWriter(
                        str(destination), cv2.VideoWriter_fourcc(*"mp4v"), 20.0, (width, height)
                    )
                    if not writer.isOpened():
                        raise RuntimeError("Could not initialize video writer")
                writer.write(frame)
                frame_count += 1
                if frame_count % 500 == 0:
                    print(f"[INFO] [ROUTE/VIDEO] Processed frames: {frame_count}", flush=True)
    finally:
        if writer is not None:
            writer.release()

    if frame_count == 0:
        raise RuntimeError("No valid video frames found")

    print(f"[INFO] [ROUTE/VIDEO] Video extraction completed: {frame_count} frames", flush=True)
    convert_to_h264(destination)
    print(f"[INFO] [ROUTE/VIDEO] MP4 file exported to: {destination} ({frame_count} frames)", flush=True)
    return destination
