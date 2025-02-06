import gc

from audiovlm_demo.core.config import Config

import torch


class AudioVLM:
    def __init__(self, *, config: Config, model_store: dict | None = None):
        self.config = config
        model_store_keys = {"Loaded", "History", "Model", "Processor"}
        if model_store is not None:
            if not model_store_keys <= model_store.keys():
                raise ValueError(
                    "Argument `model_store` is missing the following "
                    f"keys: {model_store_keys - model_store.keys()}"
                )
            self.model_store = model_store
        else:
            self.model_store = {
                "Loaded": False,
                "History": [],
                "Model": None,
                "Processor": None,
            }

    def model_cleanup(self):
        # global model_info_pane # Placeholder for Panel UI
        if self.model_store["Model"]:
            # Placeholder for Panel UI
            # model_info_pane.object = "<p><b>No Model Loaded</b></p>"
            del self.model_store["Model"]
            del self.model_store["Processor"]
            gc.collect()
            torch.cuda.empty_cache()
            self.model_store["Model"] = None
            self.model_store["Processor"] = None
            self.model_store["Loaded"] = False
