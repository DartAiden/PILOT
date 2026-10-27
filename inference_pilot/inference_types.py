from typing import List

from pydantic import BaseModel, ConfigDict, Field

from qbio.types import BaseModelInput


class InferenceInput(BaseModelInput):
    input_files: List[str]


class PilotConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    num_samples: int = Field(default=10, ge=1, le=100)
    batch_size: int = Field(default=10, ge=1, le=40)
    max_sample_iter: int = Field(default=20, ge=1, le=50)
    pocket_cutoff: float = Field(default=7.0, gt=0, le=12)
    filter_by_lipinski: bool = False
