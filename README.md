# LiDAR MV Reconstruction and DMS Collision Detection

A two-stage Python pipeline for medium-voltage (MV) LiDAR infrastructure analysis. The first stage identifies poles, labels three MV conductors, reconstructs line components, and produces a labeled point cloud. The second stage detects possible DMS (minimum safety-distance) incidents in pole-to-pole sections, extracts local clouds, prepares collision reports, and retrieves camera video clips from a GPS-enabled ROS 2 recording. A separate route utility creates a route image and video.

## Requirements

- **Miniconda** (or another compatible Conda distribution) with a terminal capable of running `conda`.
- Python **3.11**, installed automatically by the environment file.
- An operating system with support for Open3D and OpenCV. The reconstruction viewer needs a graphical desktop; the cable-labeling interface opens in a web browser.
- **FFmpeg**, installed by Conda. Check with `ffmpeg -version`.
- A sufficiently large disk and memory budget for the input PLY, intermediate PLYs, and ROS 2 MCAP recordings.
- Internet access for the optional map tile backgrounds. When tiles cannot be downloaded, map background functionality may be limited.
- **Optional NVIDIA GPU:** a compatible CUDA 12 driver and the CuPy packages mentioned below. CPU processing remains available.

### Install Miniconda and create the `lab` environment

1. Download the official **Miniconda** installer for your operating system from [Anaconda's Miniconda installation page](https://www.anaconda.com/docs/getting-started/miniconda/install).
2. Install Miniconda, then open **Miniconda Prompt**, or a terminal with Conda initialized (`conda init` may be required for your shell).
3. Clone this repository and change into its root directory, where `environment.yml` and `pyproject.toml` are located.
4. Create and activate the environment:

   ```bash
   conda env create -f environment.yml
   conda activate lab
   ```

5. Confirm that Python and FFmpeg are available:

   ```bash
   python --version
   ffmpeg -version
   ```

`environment.yml` creates an environment called **lab**, installs Python 3.11 and FFmpeg from Conda, and installs the project and its Python dependencies from **pyproject.toml** with pip. These dependencies cover NumPy, SciPy, Pandas, Matplotlib, plyfile, Open3D, PyProj, Shapely, PyYAML, websockets, rosbags, OpenCV, contextily, tqdm, and Plotly.

If you already have a `lab` environment, update it from the repository root using `conda env update -n lab -f environment.yml --prune`, then activate it. If the project's Python dependencies change, reinstall with `python -m pip install -e .`.

**Optional GPU setup:** with a compatible NVIDIA driver/CUDA 12 environment, activate `lab` and run:

```bash
python -m pip install -e ".[gpu]"
```

Then set `"GPU": true` in `config.json`. For CPU-only processing, keep `"GPU": false` (the default); CuPy is not required. DMS uses GPU only for candidate prefiltering; final geometry is evaluated on the CPU. A GPU initialization or batch failure causes a CPU fallback. GPU wheels and CUDA compatibility depend on your OS and hardware.

## Inputs and configuration

Place your original labeled LiDAR cloud in `data/` using the filename in `input_ply_raw`. Put the KML route and the related MCAP recording under `others/`. If more than one candidate recording is present, specify it explicitly when running the DMS stage.

**Extractor georeferencing is required:** supply the ENU/WGS84 origin YAML associated with your source PLY and a complete GPS trajectory (for example, the configured TUM trajectory in `others/`). The Extractor uses these sources, together with the KML route, to place and order poles geographically. Confirm the origin and trajectory belong to the same survey as the PLY. The DMS video step also requires a recording with the expected GPS and compressed camera topics. Their default names (`/swift/navsat_fix` and `/camera_front/image_raw/compressed`) are configured as Python constants in the video module.

The project uses two configuration files:

- **`config.json`** contains **only directory paths, filenames or filename templates, and the `GPU` boolean**. Use JSON `true`/`false`, not quoted strings. The relevant DMS keys are:

  ```json
  {
    "data_dir": "data",
    "output_dir": "02_reports",
    "collision_dir": "03_collisions",
    "input_ply_reconstructed": "GlobalMap_labeled_3.ply",
    "input_ply_new": "GlobalMap_labeled_4.ply",
    "zones_report_json": "zones_report.json",
    "collisions": "collisions.png",
    "collision_video_template": "collision_video_{id02d}.mp4",
    "labels_json": "labels.json",
    "GPU": false
  }
  ```

  This excerpt shows the DMS-related settings; the included configuration contains the complete project file inventory. Paths are resolved relative to the project root. The route's optional `kml` filename may be specified when multiple KML files exist.

- **`labels.json`** is the **single source of semantic class IDs**. The extractor, wire labeling, reconstruction, and DMS use the catalog. DMS maps original, DMS, and critical-point (CP) classes by their descriptive titles rather than hard-coded numeric IDs. Keep the required class titles consistent when updating the catalog.

Geometric thresholds, batch sizes, camera/GPS processing parameters, and display settings remain in the corresponding Python source modules, **not** in `config.json`.

## Run the project

Open a terminal in the repository root and activate the environment (`conda activate lab`). There are **two independent entry points** and one optional route utility:

| Executable | Command | Purpose |
| --- | --- | --- |
| `main_1.py` | `python main_1.py` | Runs the six-stage reconstruction pipeline. |
| `main_2.py` | `python main_2.py` | Runs the four-stage DMS collision analysis and video pipeline **after** reconstruction. |
| `route/base_route.py` | `python -m route.base_route` | Creates the route map and route video independently of both entry points. |
| `cable_mt/rebuild.py` | `python -m cable_mt.rebuild` | Restores the cable-labeled PLY from an existing Cable MT backup without opening the labeling interface. |

### Stage 1 — Reconstruct the MV infrastructure

Run `python main_1.py` and complete the interactive stages when prompted:

1. **Extractor:** find MV poles and save a labeled PLY with pole identifiers and raw pole metadata.
2. **Cable MT:** select two poles and label the three conductors in the browser; finalize the session to save wire labels and pole connections.
3. **Connector:** consolidate raw pole data with the cable connections without overwriting the raw report.
4. **Points:** fit conductor equations, calculate support points, and refine the wire equations.
5. **Classifier:** classify poles and wires; supply any requested bipole identifiers at the terminal.
6. **Rebuild:** create a reconstructed PLY. The Open3D viewer can save a reconstruction image with **S**.

The main labeled intermediate files are written under `data/` and the JSON results under `02_reports/`. The final reconstructed cloud is `data/GlobalMap_labeled_3.ply` with the default configuration. Cable labeling backups are stored in `data/backup/`.

### Stage 2 — Detect and document DMS collisions

Run `python main_2.py` after Stage 1. To choose a recording explicitly, use:

```bash
python main_2.py --rosbag others/recording.mcap
```

1. **Zones:** assign points to pole-to-pole sections, segment physical objects, detect DMS/CP points, and create the annotated cloud plus section-level JSON report and collision overview map.
2. **Split:** export a local PLY cloud for every confirmed collision section, preserving all vertex attributes.
3. **Export info:** generate section JSON/TXT records and top, profile, and transverse graphics.
4. **Auto GPS:** match collision GPS coordinates to recording timestamps and create H.264 MP4 clips through FFmpeg.

The default output directory is `03_collisions/`. The annotated PLY is `GlobalMap_labeled_4.ply`, and the master report is `zones_report.json`. Individual collision folders are named `collision_01/`, `collision_02/`, and so on, with their respective clouds, maps, text reports, and videos.

**Object identification:** an `object_id` is **local to its section**. Collision objects are numbered consecutively starting at **1** in each section, regardless of physical objects that did not qualify as DMS collisions. Other segmented objects within a collision-bearing section retain local IDs after the collision objects, so they can still appear as gray background context. Sections without reported collisions use `object_id=0` in the annotated PLY. Interpret an object by the pair (`section_id`, `object_id`), not by `object_id` alone. The same pair is used in the JSON and PLY. The annotated PLY uses the reported collision-section numbering; points outside reported collision sections use `section_id=0`.

**Distance fields:** `distance_m` is **not exported as a PLY vertex property**. The detection algorithm still computes separation internally, and the JSON retains the per-level summary `min_distance_m`.

**GPU:** set `"GPU": true` or `false` in `config.json`. Other numerical parameters are edited directly in the DMS source. The default console output reports pipeline stages, key exports and warnings; detailed clustering progress is suppressed by default.

### Optional route extraction

Run `python -m route.base_route` separately. This utility reads a KML route and a ROS 2 MCAP recording and writes a PNG route map and an MP4 route video under the configured `data_dir` (`route.png` and `route.mp4` by default). It is **not** invoked by either main pipeline.

## Repository structure

```text
.
├── README.md                  # Installation and usage
├── environment.yml            # Miniconda environment: lab
├── pyproject.toml             # Python dependencies and optional CUDA packages
├── config.json                # Directories, filenames, GPU toggle
├── labels.json                # Semantic class IDs
├── main_1.py                  # Reconstruction entry point
├── main_2.py                  # DMS entry point
├── extractor/                 # Pole extraction
├── cable_mt/                  # Interactive wire labeling and backup reconstruction
├── connector/                 # Consolidated pole metadata
├── points/                    # Wire equations and points
├── classifier/                # Final pole and wire classification
├── rebuild/                   # Reconstructed point cloud
├── dms/                       # Collision detection and exports
└── route/                     # Independent route map and video
```

The directories `data/`, `others/`, `02_reports/`, and `03_collisions/` are intentionally excluded from Git by `.gitignore` because they contain large inputs, generated files, or potentially sensitive location/camera data. Create these directories locally as needed. Do not publish source MCAP recordings or exact GPS positions without appropriate authorization.

## Known limitations and validation

- The pipeline has interactive stages and requires real labeled PLY, pole metadata, KML, and MCAP inputs to complete an end-to-end run.
- The visualizer and Open3D may need a working desktop/graphics configuration. Map backgrounds need network connectivity.
- Static Python checks and small synthetic checks are useful before running a new dataset, but **full end-to-end processing must be validated using the actual point cloud and ROS 2 recording**. This repository does not bundle those datasets.
- DMS collision detection and reconstruction are analytical aids and should not be treated as a substitute for physical clearance checks or engineering inspection.
