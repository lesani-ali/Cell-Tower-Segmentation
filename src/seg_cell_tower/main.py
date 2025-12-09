import argparse
import logging
import warnings

from .pipeline.pipeline import SegmentationPipeline
from .utils.logging import setup_logger
from .utils.config import load_config

warnings.filterwarnings("ignore")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Run segmentation pipeline.')
    parser.add_argument(
        '--config-dir', type=str, default='./config/config.yaml',
        help='Path to the configuration file.'
    )
    parser.add_argument(
        '--input-img-dir', type=str, default='./data/input_images',
        help='Path to the input image directory.'
    )
    parser.add_argument(
        '--output-img-dir', type=str, default='./data/output_images',
        help='Path to save the output images.'
    )
    parser.add_argument(
        '--output-mask-dir', type=str, default='./data/output_masks',
        help='Path to save the output masks.'
    )
    return parser.parse_args()


def main() -> None:
    """Main entry point for the CLI."""
    args = parse_args()

    config = load_config(args.config_dir)

    setup_logger(config.log_dir)

    logging.info('Starting segmentation pipeline...')
    logging.info(f'Configuration file: {args.config_dir}')
    logging.info(f'Input image directory: {args.input_img_dir}')
    logging.info(f'Output image directory: {args.output_img_dir}')
    logging.info(f'Output mask directory: {args.output_mask_dir}')

    logging.info('\nInstantiating the segmentation pipeline object...')
    pipeline = SegmentationPipeline(config)

    # Use the process_directory method directly
    pipeline.process_directory(
        args.input_img_dir,
        args.output_img_dir,
        args.output_mask_dir
    )

    logging.info('\nSegmentation pipeline completed successfully.')


if __name__ == '__main__':
    main()