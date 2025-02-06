import gc
import re

from audiovlm_demo.core.config import Config

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    BitsAndBytesConfig,
    Qwen2AudioForConditionalGeneration,
)


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

    def load_model(self, model_selection: str):
        # Placeholder
        # global # model_info_pane
        if self.model_store["Model"]:
            self.model_cleanup()

        match model_selection:
            case "Molmo-7B-D-0924":
                # Placeholder for Panel UI
                # model_info_pane.object = f"<p>Loading {model_selection}...</p>"
                self.model_store["Processor"] = AutoProcessor.from_pretrained(
                    self.config.model_path,
                    trust_remote_code=True,
                    torch_dtype=torch.bfloat16,
                    device_map="auto",
                )
                self.model_store["Model"] = AutoModelForCausalLM.from_pretrained(
                    self.config.model_path,
                    trust_remote_code=True,
                    torch_dtype=torch.bfloat16,
                    device_map="auto",
                )
                # Placeholder for Panel UI
                # model_info_pane.object = f"<p>{model_selection} loaded.</p>"
                self.model_store["Loaded"] = True
            case "Molmo-7B-D-0924-4bit":
                # Placeholder for Panel UI
                # model_info_pane.object = f"<p>Loading {model_selection}...</p>"
                self.model_store["Processor"] = AutoProcessor.from_pretrained(
                    self.config.model_path,
                    trust_remote_code=True,
                    torch_dtype=torch.bfloat16,
                    device_map="auto",
                )
                arguments = {
                    "device_map": "auto",
                    "torch_dtype": "auto",
                    "trust_remote_code": True,
                }
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="fp4",  # or nf4
                    bnb_4bit_use_double_quant=False,
                )
                arguments["quantization_config"] = quantization_config
                self.model_store["Model"] = AutoModelForCausalLM.from_pretrained(
                    self.config.model_path,
                    **arguments,
                )
                # Placeholder for Panel UI
                # model_info_pane.object = f"<p>{model_selection} loaded.</p>"
                self.model_store["Loaded"] = True
            case "Aria":
                # Placeholder for Panel UI
                # model_info_pane.object = f"<p>Loading {model_selection}...</p>"

                model_id_or_path = self.config.aria_model_path
                self.model_store["Processor"] = AutoProcessor.from_pretrained(
                    model_id_or_path, trust_remote_code=True
                )
                self.model_store["Model"] = AutoModelForCausalLM.from_pretrained(
                    model_id_or_path,
                    device_map="auto",
                    torch_dtype=torch.bfloat16,
                    trust_remote_code=True,
                )
                # Placeholder for Panel UI
                # model_info_pane.object = f"<p>{model_selection} loaded.</p>"
                self.model_store["Loaded"] = True
            case "Qwen2-Audio":
                # Placeholder for Panel UI
                # model_info_pane.object = f"<p>Loading {model_selection}...</p>"

                model_id_or_path = self.config.qwen_audio_model_path
                self.model_store["Processor"] = AutoProcessor.from_pretrained(
                    model_id_or_path
                )
                self.model_store["Model"] = (
                    Qwen2AudioForConditionalGeneration.from_pretrained(
                        model_id_or_path, device_map="auto"
                    )
                )
                # Placeholder for Panel UI
                # model_info_pane.object = f"<p>{model_selection} loaded.</p>"
                self.model_store["Loaded"] = True
            case _:
                pass

    @classmethod
    def parse_points(cls, points_str: str):
        # Regex to extract each <points> tag with multiple x and y pairs
        point_tags = re.findall(r"<points (.*?)>(.*?)</points>", points_str)
        if len(point_tags) == 0:
            point_tags = re.findall(r"<point (.*?)>(.*?)</point>", points_str)
        parsed_points = []
        if len(point_tags) == 0:
            return None

        for attributes, label in point_tags:
            coordinates = re.findall(r'x\d+="(.*?)" y\d+="(.*?)"', attributes)
            if not coordinates:
                single_coordinate = re.findall(r'x="(.*?)" y="(.*?)"', attributes)
                if single_coordinate:
                    coordinates = [single_coordinate[0]]
            parsed_points.append(
                {
                    "label": label,
                    "coordinates": [(float(x), float(y)) for x, y in coordinates],
                }
            )
        return parsed_points
