"""
seg_cell_tower — Cell-Tower Antenna Segmentation Package.

Sub-packages
------------
pipeline        End-to-end inference pipeline and orchestration.
models          Individual model wrappers (depth, saliency, detection, SAM).
postprocessing  Bounding-box filtering and depth-guided utilities.
evaluation      Segmentation metrics (IoU, Dice) and evaluator.
utils           I/O, config, logging, and visualization helpers.
cli             Command-line entry-points (segct).
"""

__version__ = "0.1.0"