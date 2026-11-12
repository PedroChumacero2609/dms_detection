"""Auto GPS Video Extractor
========================
Extracts collision video clips from ROS 2 MCAP recordings using GPS coordinates
and converts the resulting videos to H.264-compatible MP4 files.

Workflow:
1.  Load config.json and resolve the MCAP recording
2.  Discover and parse collision_gps_*.txt files
3.  Read the GPS trajectory from the MCAP recording
4.  Match collision START/END coordinates to GPS timestamps
5.  Apply the configured GPS tolerance and time padding
6.  Extract JPEG frames for all collision time ranges in FAST mode
7.  Assemble each MP4 using FFmpeg stream copy
8.  Convert each MP4 to H.264 and overwrite the final output atomically
9.  Remove temporary files after processing

Extraction mode is permanently FAST. Video conversion mode is permanently
OVERWRITE. Original GPS matching, timestamp selection, frame extraction,
FFmpeg assembly and H.264 encoding parameters are preserved.

Run:
    python -m dms.auto_gps
    python -m dms.auto_gps --config config.json
    python -m dms.auto_gps --rosbag /path/to/recording.mcap
"""

import bisect
import json
import logging
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from tqdm import tqdm
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore


def get_logger(name):
    return logging.getLogger(name)


def setup_logging(level="INFO"):
    logging.basicConfig(level=getattr(logging, level.upper(), logging.INFO),
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                        stream=sys.stdout)


SHOW_PROGRESS = False  # Optional detailed frame-processing progress
EARTH_RADIUS_M = 6371000.0


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance between two GPS points (meters). O(1)."""
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dl / 2) ** 2
    return EARTH_RADIUS_M * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

def validate_gps_coordinates(lat: float, lon: float) -> bool:
    return -90 <= lat <= 90 and -180 <= lon <= 180

@dataclass
class GPSPoint:
    timestamp: int  # nanoseconds
    lat: float
    lon: float
    alt: float = 0.0

class GPSTrajectory:
    """
    GPS trajectory with O(log n) lookup and O(n) segment matching.
    Uses batch insertion (sort once) for O(n log n) build cost.
    """

    def __init__(self) -> None:
        self.points: List[GPSPoint] = []
        self._timestamps: List[int] = []   # kept in sync, sorted
        self._batch: List[GPSPoint] = []
        self.logger = get_logger(__name__)

    # ------------------------------------------------------------------
    # Building the trajectory
    # ------------------------------------------------------------------

    def add_point(self, timestamp: int, lat: float, lon: float, alt: float = 0.0) -> bool:
        """Accumulate a point (call finalize_batch() when done)."""
        if not validate_gps_coordinates(lat, lon):
            return False
        self._batch.append(GPSPoint(timestamp, lat, lon, alt))
        return True

    def finalize_batch(self) -> None:
        """Sort accumulated points once and merge into main list. O(n log n)."""
        if not self._batch:
            return
        self._batch.sort(key=lambda p: p.timestamp)
        self.points.extend(self._batch)
        self._timestamps.extend(p.timestamp for p in self._batch)
        self._batch = []

    # ------------------------------------------------------------------
    # Querying
    # ------------------------------------------------------------------

    def get_timestamp_range(
        self,
        start_lat: float,
        start_lon: float,
        end_lat: float,
        end_lon: float,
        tolerance_m: float = 50.0,
        padding_start_ns: int = 0,
        padding_end_ns: int = 0,
    ) -> Optional[Tuple[int, int]]:
        """
        Find (start_ns, end_ns) for the trajectory segment closest to the
        given GPS coordinates, within tolerance_m.

        Returns None if no match found. Applies padding before/after.
        """
        if not self.points:
            return None

        result = self._find_segment_indices(start_lat, start_lon, end_lat, end_lon, tolerance_m)
        if result is None:
            return None

        start_idx, end_idx = result
        start_ns = max(0, self.points[start_idx].timestamp - padding_start_ns)
        end_ns = self.points[end_idx].timestamp + padding_end_ns
        return start_ns, end_ns

    def _find_segment_indices(
        self,
        start_lat: float,
        start_lon: float,
        end_lat: float,
        end_lon: float,
        tolerance_m: float,
    ) -> Optional[Tuple[int, int]]:
        """Find the best-match segment indices (start, end) within tolerance."""
        # 1. Find the best match for the start coordinate in the WHOLE trajectory
        start_idx: Optional[int] = None
        min_start_dist = float("inf")

        for i, pt in enumerate(self.points):
            d = haversine_distance(pt.lat, pt.lon, start_lat, start_lon)
            if d < min_start_dist:
                min_start_dist = d
                if d <= tolerance_m:
                    start_idx = i

        if start_idx is None:
            self.logger.debug(
                'Start coordinate not found within %.1f m; minimum distance: %.1f m',
                tolerance_m, min_start_dist
            )
            return None

        # 2. Find the best match for the end coordinate AFTER the start index
        end_idx: Optional[int] = None
        min_end_dist = float("inf")

        for i in range(start_idx, len(self.points)):
            pt = self.points[i]
            d = haversine_distance(pt.lat, pt.lon, end_lat, end_lon)
            if d < min_end_dist:
                min_end_dist = d
                if d <= tolerance_m:
                    end_idx = i

        if end_idx is None:
            self.logger.debug(
                'End coordinate not found after the start index within %.1f m; minimum distance: %.1f m',
                tolerance_m, min_end_dist
            )
            return None

        return start_idx, end_idx

    # ------------------------------------------------------------------
    # Info
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.points)

    def get_info(self) -> dict:
        if not self.points:
            return {"num_points": 0}
        duration_s = (self.points[-1].timestamp - self.points[0].timestamp) / 1e9
        return {
            "num_points": len(self.points),
            "duration_s": round(duration_s, 2),
            "start_ns": self.points[0].timestamp,
            "end_ns": self.points[-1].timestamp,
        }

    def get_points_in_range(self, start_ns: int, end_ns: int) -> List[GPSPoint]:
        """Return all points with start_ns <= timestamp <= end_ns."""
        idx_start = bisect.bisect_left(self._timestamps, start_ns)
        idx_end = bisect.bisect_right(self._timestamps, end_ns)
        return self.points[idx_start:idx_end]

@dataclass
class CollisionSegment:
    collision_id: int
    start_lat: float
    start_lon: float
    end_lat: float
    end_lon: float
    source_file: str  # full path to collision_gps_XX.txt
    source_folder: str  # parent folder of source_file

logger = get_logger(__name__)
_RE_COLLISION_ID = re.compile(r"COLLISION\s+ID\s*:\s*(\d+)", re.IGNORECASE)
_RE_SECTION_NUMBER = re.compile(r"Collision\s+section\s+number\s*:\s*(\d+)", re.IGNORECASE)
_RE_FILENAME_ID = re.compile(r"collision_gps_(\d+)\.txt$", re.IGNORECASE)
_RE_GEO_COORDS = re.compile(
    r"Geographic\s+Coordinates\s*:\s*([+-]?\d+(?:\.\d+)?)\s*,\s*([+-]?\d+(?:\.\d+)?)",
    re.IGNORECASE,
)
_RE_START_POLE = re.compile(r"^START\s*(?:POLE|:)", re.IGNORECASE)
_RE_END_POLE = re.compile(r"^END\s*(?:POLE|:)", re.IGNORECASE)


def parse_collision_file(txt_path: Path) -> List[CollisionSegment]:
    """Read both legacy GPS reports and current Export_GPS_TXT section reports.

    Only START POLE and END POLE coordinates define the video range.
    Geographic coordinates of intermediate DMS/critical points are ignored.
    """
    try:
        lines = txt_path.read_text(encoding="utf-8-sig", errors="replace").splitlines()
    except OSError as exc:
        logger.warning('Unable to read collision file %s: %s', txt_path, exc)
        return []

    fallback_match = _RE_FILENAME_ID.search(txt_path.name)
    fallback_id = int(fallback_match.group(1)) if fallback_match else None
    collision_id = None
    state = "idle"
    start = None
    end = None
    segments: List[CollisionSegment] = []

    def finish():
        nonlocal start, end, state
        if start is not None and end is not None:
            cid = collision_id if collision_id is not None else fallback_id
            if cid is not None:
                segments.append(CollisionSegment(
                    collision_id=cid, start_lat=start[0], start_lon=start[1],
                    end_lat=end[0], end_lon=end[1],
                    source_file=str(txt_path), source_folder=str(txt_path.parent),
                ))
            else:
                logger.warning('Missing collision ID in %s', txt_path)
        start, end, state = None, None, "idle"

    for line in lines:
        line = line.strip()
        m = _RE_COLLISION_ID.search(line) or _RE_SECTION_NUMBER.search(line)
        if m:
            # A new record starts: complete any preceding one first.
            if start is not None and end is not None:
                finish()
            collision_id = int(m.group(1))
            continue

        if _RE_START_POLE.search(line):
            if start is not None and end is not None:
                finish()
            start, end, state = None, None, "start"
            continue
        if _RE_END_POLE.search(line):
            if start is not None:
                state = "end"
            continue

        coord = _RE_GEO_COORDS.search(line)
        if not coord:
            continue
        lat, lon = float(coord.group(1)), float(coord.group(2))
        if not validate_gps_coordinates(lat, lon):
            logger.warning('Invalid GPS coordinates in %s: %s', txt_path, line)
            continue
        if state == "start" and start is None:
            start = (lat, lon)
            state = "between"
        elif state == "end" and end is None:
            end = (lat, lon)
            finish()

    if start is not None or end is not None:
        logger.warning('Incomplete START/END GPS record in %s', txt_path)
    return segments


def discover_collision_files(collisions_folder: Path) -> List[Path]:
    """Discover GPS reports from the current and legacy exporter."""
    return sorted(p for p in collisions_folder.rglob('*.txt')
                  if _RE_FILENAME_ID.fullmatch(p.name))

@dataclass
class TimeRange:
    collision_id: int
    camera_topic: str
    start_ns: int
    end_ns: int
    output_path: Path

def _slugify(topic: str) -> str:
    """Convert a ROS topic to a safe filename part."""
    return topic.replace("/", "_").strip("_")

class McapReader:
    """Original FAST implementation without unused video extraction modes."""

    def __init__(self, rosbag_path: Path) -> None:
        self.rosbag_path = rosbag_path
        self.typestore = get_typestore(Stores.ROS2_HUMBLE)
        self.logger = get_logger(__name__)

    def extract_gps_trajectory(self, gps_topic: str) -> GPSTrajectory:
        """
        Read all GPS messages and return a GPSTrajectory.
        Single pass, only reads the GPS topic (fast, messages are tiny).
        """
        trajectory = GPSTrajectory()
        count = 0

        with AnyReader([self.rosbag_path], default_typestore=self.typestore) as reader:
            conns = [c for c in reader.connections if c.topic == gps_topic]
            if not conns:
                raise ValueError(
                    f"GPS topic '{gps_topic}' not found in rosbag.\n"
                    f"Available topics: {self._list_topics(reader)}"
                )

            total = sum(c.msgcount for c in conns)
            for connection, timestamp, rawdata in tqdm(
                reader.messages(connections=conns),
                disable=not SHOW_PROGRESS,
                total=total,
                desc="Reading GPS",
                unit="msg",
                leave=False,
            ):
                try:
                    msg = reader.deserialize(rawdata, connection.msgtype)
                    trajectory.add_point(
                        timestamp=timestamp,
                        lat=float(msg.latitude),
                        lon=float(msg.longitude),
                        alt=float(msg.altitude) if hasattr(msg, "altitude") else 0.0,
                    )
                    count += 1
                except Exception as e:
                    self.logger.debug('GPS message deserialization failed at timestamp %d: %s', timestamp, e)

        trajectory.finalize_batch()
        self.logger.info('GPS trajectory loaded: %d points', count)
        return trajectory

    def extract_jpeg_segments(
        self,
        time_ranges: List[TimeRange],
        temp_dir: Path,
    ) -> Dict[int, Dict[str, List[Path]]]:
        """
        Single pass through the .mcap, extracting raw JPEG bytes for all
        time ranges simultaneously. No decoding, no numpy — pure file I/O.

        Returns: {collision_id: {camera_topic: [sorted list of jpeg paths]}}
        """
        if not time_ranges:
            return {}

        # Sort ranges by start for efficient sweep + early termination
        sorted_ranges = sorted(time_ranges, key=lambda r: r.start_ns)
        needed_topics: Set[str] = {tr.camera_topic for tr in sorted_ranges}
        global_start = sorted_ranges[0].start_ns
        global_end = max(r.end_ns for r in sorted_ranges)

        # Pre-create temp subdirectories and output structure
        jpeg_collections: Dict[int, Dict[str, List[Path]]] = {}
        frame_counters: Dict[Tuple[int, str], int] = {}
        for tr in sorted_ranges:
            jpeg_collections.setdefault(tr.collision_id, {}).setdefault(tr.camera_topic, [])
            frame_counters[(tr.collision_id, tr.camera_topic)] = 0
            d = temp_dir / f"col{tr.collision_id:03d}_{_slugify(tr.camera_topic)}"
            d.mkdir(parents=True, exist_ok=True)

        with AnyReader([self.rosbag_path], default_typestore=self.typestore) as reader:
            conns = [c for c in reader.connections if c.topic in needed_topics]
            if not conns:
                raise ValueError(
                    f"None of the camera topics found in rosbag: {needed_topics}\n"
                    f"Available: {self._list_topics(reader)}"
                )

            total = sum(c.msgcount for c in conns)
            with tqdm(total=total, desc="Extracting frames", unit="frame", disable=not SHOW_PROGRESS) as pbar:
                for connection, timestamp, rawdata in reader.messages(
                    connections=conns,
                    start=global_start,
                    stop=global_end,
                ):
                    pbar.update(1)
                    topic = connection.topic

                    for tr in sorted_ranges:
                        if tr.camera_topic != topic:
                            continue
                        if timestamp < tr.start_ns or timestamp > tr.end_ns:
                            continue

                        # In range — extract JPEG bytes without full deserialize
                        jpeg = self._extract_jpeg_bytes_raw(rawdata)
                        if jpeg is None:
                            # Fall back to deserialize
                            try:
                                msg = reader.deserialize(rawdata, connection.msgtype)
                                jpeg = bytes(msg.data)
                            except Exception as e:
                                self.logger.debug('Camera frame deserialization failed: %s', e)
                                continue # Skip this range, try others

                        key = (tr.collision_id, topic)
                        idx = frame_counters[key]
                        frame_counters[key] = idx + 1

                        out_dir = temp_dir / f"col{tr.collision_id:03d}_{_slugify(topic)}"
                        frame_path = out_dir / f"frame_{idx:08d}.jpg"
                        frame_path.write_bytes(jpeg)
                        jpeg_collections[tr.collision_id][topic].append(frame_path)

        # Log frame counts
        for tr in sorted_ranges:
            n = len(jpeg_collections.get(tr.collision_id, {}).get(tr.camera_topic, []))
            self.logger.info('Collision %d | Camera %s | Extracted frames: %d', tr.collision_id, tr.camera_topic, n)

        return jpeg_collections

    @staticmethod
    def _list_topics(reader) -> List[str]:
        return sorted({c.topic for c in reader.connections})

    @staticmethod
    def _extract_jpeg_bytes_raw(rawdata: bytes) -> Optional[bytes]:
        """
        Fast CDR extraction of JPEG bytes from a CompressedImage message.
        CDR layout (little-endian):
          [0:4]   CDR header
          [4:8]   sec  (uint32)
          [8:12]  nanosec (uint32)
          [12:16] frame_id length (uint32)
          [16 + frame_id_len + pad] format string length + bytes
          [...] data array length (uint32) + JPEG bytes
        This is heuristic — if it fails, the caller falls back to full deserialize.
        """
        try:
            import struct
            pos = 4  # skip CDR header

            # Header.stamp (sec + nanosec)
            pos += 8

            # Header.frame_id: uint32 length + chars + null + padding to 4
            fid_len = struct.unpack_from("<I", rawdata, pos)[0]
            pos += 4 + fid_len
            # align to 4 bytes
            if pos % 4:
                pos += 4 - (pos % 4)

            # format string: uint32 length + chars + null + padding
            fmt_len = struct.unpack_from("<I", rawdata, pos)[0]
            pos += 4 + fmt_len
            if pos % 4:
                pos += 4 - (pos % 4)

            # data array: uint32 count + bytes
            data_len = struct.unpack_from("<I", rawdata, pos)[0]
            pos += 4
            return rawdata[pos: pos + data_len]
        except Exception:
            return None

def generate_collision_kml(
    output_path: Path,
    collision_id: int,
    start_pole: Tuple[float, float],
    end_pole: Tuple[float, float],
    trajectory_points: List[Tuple[float, float, float]],
) -> None:
    """Export original Auto-GPS diagnostic KML: START, END and video GPS track."""
    from xml.etree import ElementTree as ET

    ns = "http://www.opengis.net/kml/2.2"
    ET.register_namespace("", ns)
    def element(parent, name, text=None, **attributes):
        node = ET.SubElement(parent, f"{{{ns}}}{name}", attributes)
        if text is not None:
            node.text = str(text)
        return node

    root = ET.Element(f"{{{ns}}}kml")
    document = element(root, "Document")
    element(document, "name", f"Collision {collision_id}")
    style = element(document, "Style", id="pathStyle")
    line_style = element(style, "LineStyle")
    element(line_style, "color", "ff0000ff")  # KML AABBGGRR: red
    element(line_style, "width", "3")

    for label, (lat, lon) in (("Pole 1 (START)", start_pole),
                              ("Pole 2 (END)", end_pole)):
        placemark = element(document, "Placemark")
        element(placemark, "name", label)
        point = element(placemark, "Point")
        element(point, "coordinates", f"{lon},{lat},0")

    placemark = element(document, "Placemark")
    element(placemark, "name", "Video Trajectory")
    element(placemark, "styleUrl", "#pathStyle")
    line = element(placemark, "LineString")
    element(line, "tessellate", "1")
    element(line, "coordinates", "\n".join(
        f"{lon},{lat},{alt}" for lat, lon, alt in trajectory_points
    ))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(root).write(output_path, encoding="utf-8", xml_declaration=True)


def build_concat_list(jpeg_paths: List[Path], fps: float, concat_file: Path) -> None:
    """Write an FFmpeg concat demuxer input list."""
    duration = 1.0 / fps
    lines = ["ffconcat version 1.0\n"]
    for p in jpeg_paths:
        # Use forward slashes; works on both Windows and Linux with -safe 0
        lines.append(f"file '{p.as_posix()}'\n")
        lines.append(f"duration {duration:.8f}\n")
    concat_file.write_text("".join(lines), encoding="utf-8")

def run_stream_copy(concat_file: Path, output_mp4: Path) -> bool:
    """
    Mux JPEG frames into MP4 using FFmpeg concat demuxer with stream copy.
    No re-encoding — fastest possible output.
    """
    cmd = [
        "ffmpeg", "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", str(concat_file),
        "-c:v", "copy",
        "-movflags", "+faststart",
        str(output_mp4),
    ]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        logger.debug('FFmpeg diagnostic output:\n%s', result.stderr.decode(errors="replace"))
    return result.returncode == 0

def require_ffmpeg():
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found on PATH. Install FFmpeg before running DMS.")


def resolve_rosbag_source(value: str) -> Path:
    """Resolve a standalone MCAP, a rosbag2 directory, or a folder containing an MCAP.

    Never choose arbitrarily between recordings: using the wrong recording
    would produce incorrect cuts even if the timestamp logic remains intact.
    """
    source = Path(value).expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(
            f"MCAP path does not exist: {source}\n"
            "Specify the actual recording with --rosbag /path/recording.mcap."
        )
    if source.is_file():
        if source.suffix.lower() == ".mcap":
            return source
        if source.suffix.lower() == ".db3":
            parent = source.parent
            if (parent / "metadata.yaml").is_file():
                return parent
            raise ValueError(f"The .db3 file requires a rosbag2 directory with metadata.yaml: {parent}")
        raise ValueError(f"Expected an .mcap file: {source}")
    if not source.is_dir():
        raise ValueError(f"Invalid rosbag path: {source}")

    # Open a complete rosbag2 directory as a directory, not as a standalone MCAP.
    if (source / "metadata.yaml").is_file():
        return source

    candidates = sorted(p for p in source.rglob("*.mcap") if p.is_file())
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise FileNotFoundError(
            f"No .mcap file or metadata.yaml found inside: {source}\n"
            "Find your recording with: find . -type f -iname '*.mcap'\n"
            "Then run: python -m dms.auto_gps --rosbag /path/to/recording.mcap"
        )
    options = "\n".join(f"  - {p}" for p in candidates[:20])
    raise ValueError(
        f"Found {len(candidates)} .mcap files inside {source}; "
        "cannot select one safely without risking extraction from the wrong recording.\n"
        f"{options}\nSpecify one with --rosbag /path/to/recording.mcap, "
        "or specify a rosbag2 directory containing metadata.yaml."
    )


@dataclass
class Config:
    # Read configuration from config.json; extraction and conversion modes are fixed.
    rosbag_path: str
    collisions_folder: str
    output_video_template: str
    camera_topics: List[str]
    gps_topic: str
    gps_tolerance_m: float
    padding_seconds: float
    output_fps: float

    @classmethod
    def from_json(cls, config_path, rosbag_path=None):
        """Load either the original DMS JSON or a JSON with an auto_gps section.

        Defaults match the original Auto-GPS FAST configuration; GPS matching
        and extraction behavior are unchanged. The optional --rosbag argument
        overrides the recording path defined in the configuration.
        """
        path = Path(config_path).resolve()
        data = json.loads(path.read_text(encoding="utf-8"))
        # Algorithm settings are intentionally stored in this module, not config.json.
        gps = {}

        def resolve(value):
            p = Path(value).expanduser()
            return str(p if p.is_absolute() else (path.parent / p).resolve())

        return cls(
            rosbag_path=resolve(
                rosbag_path if rosbag_path is not None
                else data["others_dir"]
            ),
            collisions_folder=resolve(data["collision_dir"]),
            output_video_template=data["collision_video_template"],
            camera_topics=list(gps.get("camera_topics", ["/camera_front/image_raw/compressed"])),
            gps_topic=gps.get("gps_topic", "/swift/navsat_fix"),
            gps_tolerance_m=float(gps.get("gps_tolerance_m", 35.0)),
            padding_seconds=float(gps.get("padding_seconds", 5.0)),
            output_fps=float(gps.get("output_fps", 24)),
        )

    def validate(self):
        # Resolve the recording source before instantiating AnyReader; keep GPS logic unchanged.
        self.rosbag_path = str(resolve_rosbag_source(self.rosbag_path))
        if not Path(self.collisions_folder).is_dir():
            raise FileNotFoundError(f"Collision folder not found: {self.collisions_folder}")
        if not self.camera_topics:
            raise ValueError("auto_gps.camera_topics must contain at least one topic")
        if self.output_fps <= 0:
            raise ValueError("auto_gps.output_fps must be greater than zero")


def convert_to_compatible_mp4(input_path: str, output_path: str) -> bool:

    if not os.path.exists(input_path):
        print(f"[ERROR] [DMS/VIDEO] Input video file not found: {input_path}")
        return False

    command = [
        'ffmpeg',
        '-y',
        '-i', input_path,
        '-c:v', 'libx264',
        '-tag:v', 'avc1',
        '-profile:v', 'high',
        '-level', '4.0',
        '-pix_fmt', 'yuv420p',
        '-crf', '15',
        '-preset', 'fast',
        '-c:a', 'copy',
        output_path
    ]

    try:
        subprocess.run(
            command,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE
        )
        print(f"[INFO] [DMS/VIDEO] Video converted successfully: {output_path}")
        return True

    except subprocess.CalledProcessError as e:
        print(f"[ERROR] [DMS/VIDEO] FFmpeg failed for input: {input_path}")
        print(e.stderr.decode())
        return False

    except FileNotFoundError:
        print('[ERROR] [DMS/VIDEO] FFmpeg executable not found.')
        return False

def convert_overwrite(video: Path, output_path: Optional[Path] = None) -> bool:
    """Convert and replace the output only after successful video conversion.

    Keep the final MP4 filename unchanged, and clean up temporary files even
    when FFmpeg fails.
    """
    target = output_path if output_path is not None else video
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=".auto_gps_convert_", suffix=".mp4", dir=target.parent)
    os.close(fd)
    temp = Path(temp_name)
    try:
        if not convert_to_compatible_mp4(str(video), str(temp)):
            return False
        if not temp.is_file() or temp.stat().st_size == 0:
            return False
        temp.replace(target)
        return True
    finally:
        temp.unlink(missing_ok=True)


class SegmentExtractor:
    """Original FAST implementation without unused video extraction modes."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.reader = McapReader(Path(config.rosbag_path))
        self.logger = get_logger(__name__)

    def run(self) -> int:
        """
        Full pipeline. Returns number of successfully generated videos.

        Step 1: Validate + check ffmpeg
        Step 2: Discover + parse collision .txt files
        Step 3: Extract GPS trajectory (MCAP pass 1)
        Step 4: Resolve GPS coords → timestamp ranges
        Step 5: Extract video data (MCAP pass 2, single pass)
        Step 6: Assemble MP4 per collision via FFmpeg
        """
        require_ffmpeg()

        # Step 2
        segments = self._load_collision_segments()
        if not segments:
            self.logger.error('No collision segments found; verify collisions_folder in the configuration.')
            return 0
        self.logger.info('Collision segments loaded: %d across all files', len(segments))

        # Step 3
        self.logger.info('Extracting GPS trajectory from the rosbag...')
        trajectory = self.reader.extract_gps_trajectory(self.config.gps_topic)
        info = trajectory.get_info()
        self.logger.info(
            'GPS trajectory ready: %d points over %.1f s',
            info["num_points"], info.get("duration_s", 0),
        )

        # Step 4
        padding_ns = int(self.config.padding_seconds * 1e9)
        time_ranges: List[TimeRange] = []
        for seg in segments:
            for topic in self.config.camera_topics:
                ts_range = trajectory.get_timestamp_range(
                    seg.start_lat, seg.start_lon,
                    seg.end_lat, seg.end_lon,
                    tolerance_m=self.config.gps_tolerance_m,
                    padding_start_ns=padding_ns,
                    padding_end_ns=padding_ns,
                )
                if ts_range is None:
                    self.logger.warning(
                        'Collision %d: GPS coordinates do not match the trajectory '
                        "(tolerance=%.1fm). Skipping.",
                        seg.collision_id, self.config.gps_tolerance_m,
                    )
                    continue
                output_path = self._build_output_path(seg, topic)
                time_ranges.append(TimeRange(
                    collision_id=seg.collision_id,
                    camera_topic=topic,
                    start_ns=ts_range[0],
                    end_ns=ts_range[1],
                    output_path=output_path,
                ))
                self.logger.debug(
                    'Collision %d | Camera %s | Duration: %.2f s | Output: %s',
                    seg.collision_id, topic,
                    (ts_range[1] - ts_range[0]) / 1e9,
                    output_path.name,
                )

        if not time_ranges:
            self.logger.error('No valid extraction time ranges could be resolved; video generation skipped.')
            return 0
        self.logger.info('Resolved extraction time ranges: %d/%d', len(time_ranges), len(segments) * len(self.config.camera_topics))

        # Step 4.5: Diagnostic KML for each matched collision (original Auto-GPS feature).
        self._generate_kmls(segments, trajectory, time_ranges)

        # Step 5 + 6
        # FAST only. No re-encode extraction path or mode selector.
        return self._run_fast_path(time_ranges)

    def _generate_kmls(
        self,
        segments: List[CollisionSegment],
        trajectory: GPSTrajectory,
        time_ranges: List[TimeRange],
    ) -> None:
        """Create one KML per collision using its matched, padded video GPS range."""
        generated_collisions: Set[int] = set()
        for tr in time_ranges:
            if tr.collision_id in generated_collisions:
                continue
            seg = next((s for s in segments if s.collision_id == tr.collision_id), None)
            if seg is None:
                continue
            points = trajectory.get_points_in_range(tr.start_ns, tr.end_ns)
            trajectory_coords = [(p.lat, p.lon, p.alt) for p in points]
            kml_path = Path(seg.source_folder) / f"collision_{seg.collision_id:02d}_debug.kml"
            try:
                generate_collision_kml(
                    kml_path,
                    seg.collision_id,
                    (seg.start_lat, seg.start_lon),
                    (seg.end_lat, seg.end_lon),
                    trajectory_coords,
                )
            except OSError as exc:
                self.logger.warning(
                    'Collision %d: KML export failed: %s', seg.collision_id, exc
                )
                continue
            self.logger.info(
                'Collision %d: diagnostic KML exported to %s',
                seg.collision_id, kml_path,
            )
            generated_collisions.add(tr.collision_id)

    def _run_fast_path(self, time_ranges: List[TimeRange]) -> int:
        self.logger.info('Video extraction mode: fast (JPEG stream copy; no re-encoding)')
        success = 0

        with tempfile.TemporaryDirectory() as tmp:
            temp_dir = Path(tmp)

            # MCAP pass 2: extract all JPEG frames in one sweep
            jpeg_collections = self.reader.extract_jpeg_segments(time_ranges, temp_dir)

            # FFmpeg assembly per collision
            for tr in time_ranges:
                jpegs = jpeg_collections.get(tr.collision_id, {}).get(tr.camera_topic, [])
                if not jpegs:
                    self.logger.warning(
                        'Collision %d [%s]: no frames extracted; skipping.',
                        tr.collision_id, tr.camera_topic,
                    )
                    continue

                concat_file = temp_dir / f"concat_{tr.collision_id:03d}_{tr.camera_topic.replace('/', '_')}.txt"
                build_concat_list(jpegs, self.config.output_fps, concat_file)

                tr.output_path.parent.mkdir(parents=True, exist_ok=True)
                self.logger.info(
                    'Collision %d: encoding %d frames to %s',
                    tr.collision_id, len(jpegs), tr.output_path,
                )
                # Preserve the existing MP4 until H.264 conversion succeeds.
                # FAST keeps the original run_stream_copy implementation and concat list.
                fd, fast_name = tempfile.mkstemp(
                    prefix=".auto_gps_fast_", suffix=".mp4", dir=tr.output_path.parent,
                )
                os.close(fd)
                fast_mp4 = Path(fast_name)
                try:
                    ok = run_stream_copy(concat_file, fast_mp4)
                    if ok:
                        # Always overwrite the final video with an H.264 / avc1-compatible MP4.
                        if convert_overwrite(fast_mp4, tr.output_path):
                            success += 1
                            self.logger.info('Collision %d: compatible video exported to %s', tr.collision_id, tr.output_path)
                        else:
                            self.logger.error('Collision %d: H.264 conversion failed; previous MP4 preserved.', tr.collision_id)
                    else:
                        self.logger.error('Collision %d: FFmpeg encoding failed; previous MP4 preserved.', tr.collision_id)
                finally:
                    fast_mp4.unlink(missing_ok=True)

        return success

    def _load_collision_segments(self) -> List[CollisionSegment]:
        collision_files = discover_collision_files(Path(self.config.collisions_folder))
        if not collision_files:
            self.logger.warning('No collision coordinate files found in: %s', self.config.collisions_folder)
            return []
        self.logger.info('Collision coordinate files found: %d', len(collision_files))
        segments = []
        for f in collision_files:
            segs = parse_collision_file(f)
            if segs:
                self.logger.debug('Collision file %s: %d segments', f.name, len(segs))
            else:
                self.logger.warning('Collision file %s: no valid segments found', f.name)
            segments.extend(segs)
        return segments

    def _build_output_path(self, seg: CollisionSegment, topic: str) -> Path:
        # Use the parent folder of the source .txt file
        folder = Path(seg.source_folder)
        # Replace template variables
        filename = self.config.output_video_template.format(
            id=seg.collision_id,
            id02d=f"{seg.collision_id:02d}",
            topic=_slugify(topic),
        )
        return folder / filename


def process_collision_videos(config_path="config.json", rosbag_path=None) -> int:
    """Run FAST GPS video extraction and OVERWRITE H.264 conversion.

    Inputs are config.json, collision GPS text files and an MCAP recording.
    Return the number of successfully extracted and converted MP4 videos.
    """
    cfg = Config.from_json(config_path, rosbag_path)
    cfg.validate()
    setup_logging()
    logging.getLogger(__name__).info("Auto-GPS FAST + video converter OVERWRITE | MCAP: %s", cfg.rosbag_path)
    return SegmentExtractor(cfg).run()


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="DMS: Auto-GPS FAST + video converter OVERWRITE")
    p.add_argument("--config", default="config.json", help="Path to the DMS config.json file")
    p.add_argument("--rosbag", default=None, help="Optional MCAP/rosbag path override for config.json")
    args = p.parse_args()
    count = process_collision_videos(args.config, args.rosbag)
    print(f"[INFO] [DMS/AUTO_GPS] Videos completed: {count}")
    if count == 0:
        sys.exit(1)
