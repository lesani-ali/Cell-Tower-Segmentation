import argparse
import warnings

from ..pipeline.pipeline import SegmentationPipeline
from ..config import load_config
from ..logging import setup_logger, get_logger

warnings.filterwarnings("ignore")

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the cell-tower segmentation pipeline."
    )
    parser.add_argument(
        "-c", "--config-dir", type=str, default="./config/config.yaml",
        help="Path to the YAML configuration file.",
    )
    parser.add_argument(
        "-i", "--input-img-dir", type=str, default="./data/input_images",
        help="Directory of input images.",
    )
    parser.add_argument(
        "-o", "--output-img-dir", type=str, default="./data/output_images",
        help="Directory to save overlay images.",
    )
    parser.add_argument(
        "-m", "--output-mask-dir", type=str, default="./data/output_masks",
        help="Directory to save binary masks.",
    )
    parser.add_argument(
        "-e", "--eval", action="store_true",
        help="Enable evaluation against COCO-format ground truth.",
    )
    parser.add_argument(
        "-g", "--gt-path", type=str, default="./data/annotation/instances.json",
        help="Path to COCO-format GT annotation JSON (used only when --eval is set).",
    )
    parser.add_argument(
        "-r", "--output-report", type=str, default=None,
        help="(Optional) Path to save the evaluation report CSV (used only when --eval is set).",
    )
    return parser.parse_args()


def main() -> None:
    """Entry-point for the `segct` command."""
    args = parse_args()

    config = load_config(args.config_dir)
    setup_logger(log_file=config.log_dir)

    logger.info("=" * 60)
    logger.info("  Cell-Tower Segmentation Pipeline")
    logger.info("=" * 60)
    logger.info(f"  Config      : {args.config_dir}")
    logger.info(f"  Input dir   : {args.input_img_dir}")
    logger.info(f"  Output dir  : {args.output_img_dir}")
    logger.info(f"  Mask dir    : {args.output_mask_dir}")
    if args.eval:
        logger.info(f"  Eval        : ON  (GT Path: {args.gt_path})")
    else:
        logger.info("  Eval        : OFF")
    logger.info("")
    logger.info("Instantiating pipeline…")
    pipeline = SegmentationPipeline(config)

    pipeline.process_directory(
        args.input_img_dir,
        args.output_img_dir,
        args.output_mask_dir,
        gt_path=args.gt_path if args.eval else None,
        output_report=args.output_report if args.eval else None,
    )

    logger.info("Pipeline completed successfully.")


if __name__ == "__main__":
    main()

