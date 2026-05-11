# gen_view_eval Package Layout

- `config.py`: Global configuration and metric-group definitions.
- `optional_deps.py`: Best-effort imports for optional third-party dependencies.
- `utils.py`: Numerical, geometry, and image helper functions.
- `visualization.py`: Qualitative visualization utilities.
- `evaluator.py`: Thin orchestration layer (`ScanNetEvaluator`) that coordinates backends and metrics.
- `cli.py`: Command-line parsing, evaluation loop, and summary output.

Subpackages:
- `backends/colmap_backend.py`: COLMAP SfM pose/point backend.
- `backends/vggt_backend.py`: VGGT pose/point backend.
- `metrics/object_metrics.py`: Annotation/detector object and relation metrics.

Compatibility:
- `../eval_pose_metric.py` remains the public entrypoint and now forwards to `gen_view_eval.cli.main`.
