from typing import Literal, Optional
from pydantic import BaseModel, Field


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
    saliency: SaliencyConfig = Field(default_factory=SaliencyConfig)
    depth: DepthConfig = Field(default_factory=DepthConfig)
    object_detection: ObjectDetectionConfig = Field(
        default_factory=ObjectDetectionConfig
    )
    segmentation: SegmentationConfig = Field(
        default_factory=SegmentationConfig
    )


class Config(BaseModel):
    models: ModelsConfig = Field(default_factory=ModelsConfig)
    recover_info_threshold: int = Field(ge=0, default=140)
    ignore_info_threshold: int = Field(ge=0, default=80)
    log_dir: Optional[str] = None