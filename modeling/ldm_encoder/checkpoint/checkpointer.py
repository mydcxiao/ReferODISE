
import os.path as osp
from collections import defaultdict
from typing import List
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.checkpoint.c2_model_loading import align_and_update_state_dicts
from fvcore.common.checkpoint import Checkpointer

from ..utils.file_io import PathManager


class LdmCheckpointer(Checkpointer):
    def __init__(self, model, save_dir="", *, save_to_disk=None, **checkpointables):
        super().__init__(
            model=model, save_dir=save_dir, save_to_disk=save_to_disk, **checkpointables
        )
        self.path_manager = PathManager

    def _load_model(self, checkpoint):
        # rename the keys in checkpoint
        checkpoint["model"] = checkpoint.pop("state_dict")
        return super()._load_model(checkpoint)
