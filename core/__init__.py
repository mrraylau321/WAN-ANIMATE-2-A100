# Modified for the V100 (sm70) port, 2026-10 - see README.md / NOTES_V100.md.
import importlib
from omegaconf import OmegaConf
from .base_inference_pipeline import BaseInferencePipeline

def build_object_from_dict(config):
    class_path = config.pop("type")
    module_path, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    obj = cls(**config)
    
    return obj


def build_object_from_config_file(config_path, overrides=None):
    config = OmegaConf.load(config_path)
    if overrides:  # V100 port: dotlist overrides, e.g. ["test_cfg.dit_dtype=fp16"]
        config = OmegaConf.merge(config, OmegaConf.from_dotlist(list(overrides)))
    class_path = config.pop("type")
    module_path, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    obj = cls(config)
    return obj
