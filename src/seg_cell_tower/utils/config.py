
import yaml
from pydantic import BaseModel, Field
from typing import Literal


class SaliencyConfig(BaseModel):
    mode: str = "base"
    device: str = "cuda:0"


class DepthConfig(BaseModel):
    ckpt: str
    device: str = "cuda:0"


class ObjectDetectionConfig(BaseModel):
    ckpt: str
    config_path: str
    text_prompt: str = "small bright rectangles attached to tower"
    box_threshold: float = Field(ge=0.0, le=1.0, default=0.14)
    text_threshold: float = Field(ge=0.0, le=1.0, default=0.25)
    device: str = "cuda:0"


class SegmentationConfig(BaseModel):
    ckpt: str
    model_type: Literal["vit_h", "vit_l", "vit_b"] = "vit_h"
    device: str = "cuda:0"


class ModelsConfig(BaseModel):
    saliency: SaliencyConfig
    depth: DepthConfig
    object_detection: ObjectDetectionConfig
    segmentation: SegmentationConfig


class Config(BaseModel):
    models: ModelsConfig
    recover_info_threshold: int = Field(ge=0, default=140, description="Depth threshold for recoverable information")
    ignore_info_threshold: int = Field(ge=0, default=80, description="Depth threshold for ignorable information")
    log_dir: str = "./logs"


def load_config(config_path: str) -> Config:
    """Load and validate configuration from YAML file."""
    with open(config_path, 'r') as file:
        config_dict = yaml.safe_load(file)
    
    return Config(**config_dict)