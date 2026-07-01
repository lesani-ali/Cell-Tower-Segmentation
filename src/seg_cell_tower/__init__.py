"""
seg_cell_tower — Cell-Tower Antenna Segmentation Package.

Modules
-------
pipeline        End-to-end inference pipeline and orchestration.
models          Individual model wrappers (depth, saliency, detection, SAM).
postprocessing  Bounding-box filtering and depth-guided recovery utilities.
evaluation      COCO-style segmentation metrics (IoU, Dice) and evaluator.
utils           Image I/O and visualization helpers.
config          Pydantic-based YAML configuration schema.
logging         Logger setup (console + file, tqdm-safe).
cli             Command-line entry-point (segct).
"""

__version__ = "0.1.0"