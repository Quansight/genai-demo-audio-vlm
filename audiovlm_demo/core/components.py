from __future__ import annotations
import gc
import io
import re
import time
from pathlib import Path
from typing import Annotated, Any

import librosa
import tomlkit
import torch
from pydantic import AfterValidator
from pydantic_settings import BaseSettings
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    BitsAndBytesConfig,
    GenerationConfig,
    Qwen2AudioForConditionalGeneration,
)

from audiovlm_demo.core.utils import resolve_path

_ResolvedPath = Annotated[Path, AfterValidator(resolve_path)]


class Config(BaseSettings):
    """
    Class to store configuration for the demo.

    Includes paths to downloaded models, among other thnigs.
    """

    model_path: _ResolvedPath
    aria_model_path: _ResolvedPath
    qwen_audio_model_path: _ResolvedPath

    @classmethod
    def from_file(cls, path: str | Path) -> Config:
        path = resolve_path(path)
        if not path.is_file():
            raise FileNotFoundError(f"{path} does not exist.")

        with open(path) as file:
            return cls.model_validate(tomlkit.load(file).unwrap())


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
                self.model_store["Loaded"] = True
            case "Molmo-7B-D-0924-4bit":
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
                self.model_store["Loaded"] = True
            case "Aria":
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
                self.model_store["Loaded"] = True
            case "Qwen2-Audio":
                model_id_or_path = self.config.qwen_audio_model_path
                self.model_store["Processor"] = AutoProcessor.from_pretrained(
                    model_id_or_path
                )
                self.model_store["Model"] = (
                    Qwen2AudioForConditionalGeneration.from_pretrained(
                        model_id_or_path, device_map="auto"
                    )
                )
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

    # This function signature is up for discussion
    def build_chat_history(self, users_and_objects: dict):
        return [
            {
                "role": user_and_object["user"],
                "content": user_and_object["object"],
            }
            for user_and_object in users_and_objects
        ]

    _default_system_prompt = "You are an unbiased, helpful assistant."

    def compile_prompt_gguf(
        self,
        # TODO: Change `Any` to the correct type
        history: list[dict[str, Any]],
        user_name: str,
        assistant_name: str,
        system_prompt: str | None = None,
    ):
        if system_prompt is None:
            system_prompt = self._default_system_prompt

        messages = []

        for i in history:
            if i["role"] == user_name:
                messages.append(
                    {
                        "role": "user",
                        "content": [{"text": i["content"], "type": "text"}],
                    }
                )
            elif i["role"] == assistant_name:
                messages.append(
                    {
                        "role": "assistant",
                        "content": [{"text": i["content"], "type": "text"}],
                    }
                )
            else:
                pass

        if messages[-1]["role"] == "user":
            messages[-1]["content"].append({"text": None, "type": "image"})
        return messages

    def compile_prompt(
        self,
        # TODO: Change `Any` to the correct type
        history: list[dict[str, Any]],
        user_name: str,
        assistant_name: str,
        system_prompt: str | None = None,
    ):
        if system_prompt is None:
            system_prompt = self._default_system_prompt

        texts = [""]
        for i in history:
            if i["role"] == user_name:
                texts.append(f'<|startoftext|>USER: {i["content"]}\nASSISTANT:')
            elif i["role"] == assistant_name:
                if i["content"][-13:] == "<|endoftext|>":
                    texts.append(f'{i["content"]}\n')
                elif i["content"][-15:] == "<|endoftext|>\n":
                    texts.append(f'{i["content"]}')
                else:
                    texts.append(f'{i["content"]}<|endoftext|>\n')
            else:
                pass
        return "".join(texts)

    # TODO: Add type annotations
    def molmo_callback(self, *, image, chat_history):
        prompt_full = self.compile_prompt(
            self.build_chat_history(chat_history), "User", "Assistant"
        )

        inputs = self.model_store["Processor"].process(images=[image], text=prompt_full)

        inputs = {
            k: v.to(self.model_store["Model"].device).unsqueeze(0)
            for k, v in inputs.items()
        }

        with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
            output = self.model_store["Model"].generate_from_batch(
                inputs,
                GenerationConfig(max_new_tokens=1250, stop_strings="<|endoftext|>"),
                tokenizer=self.model_store["Processor"].tokenizer,
            )

        generated_tokens = output[0, inputs["input_ids"].size(1) :]
        self.model_store["History"].append(generated_tokens)
        generated_text = self.model_store["Processor"].tokenizer.decode(
            generated_tokens, skip_special_tokens=True
        )

        points_data = self.parse_points(generated_text)
        if points_data:
            self.overlay_points(points_data)
        time.sleep(0.1)
        return generated_text

    # TODO: Add type annotations
    def aria_callback(self, *, image, chat_history):
        messages = self.engine.compile_prompt_gguf(
            self.engine.build_chat_history(chat_history), "User", "Assistant"
        )
        text = self.engine.model_store["Processor"].apply_chat_template(
            messages, add_generation_prompt=True
        )
        inputs = self.engine.model_store["Processor"](
            text=text, images=image, return_tensors="pt"
        )
        inputs["pixel_values"] = inputs["pixel_values"].to(
            self.engine.model_store["Model"].dtype
        )
        inputs = {
            k: v.to(self.engine.model_store["Model"].device) for k, v in inputs.items()
        }

        with torch.inference_mode(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
            output = self.engine.model_store["Model"].generate(
                **inputs,
                max_new_tokens=500,
                stop_strings=["<|im_end|>"],
                tokenizer=self.engine.model_store["Processor"].tokenizer,
                do_sample=True,
                temperature=0.7,
            )
            output_ids = output[0][inputs["input_ids"].shape[1] :]
            result = self.engine.model_store["Processor"].decode(
                output_ids, skip_special_tokens=True
            )
            result = result.replace("<|im_end|>", "")
        time.sleep(0.1)
        return result

    # TODO: Add type annotations
    def qwen_callback(self, *, audio_file_content, chat_history):
        messages = self.engine.build_chat_history(chat_history)[-1]
        if messages["role"] == "User":
            text_input = messages["content"]
        else:
            return "Error handling input content - please restart application and try again."

        conversation = [
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio_url": "Filler.wav"},
                    {"type": "text", "text": text_input},
                ],
            },
        ]
        text = self.engine.model_store["Processor"].apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=False
        )
        audios = []
        for message in conversation:
            if isinstance(message["content"], list):
                for ele in message["content"]:
                    if ele["type"] == "audio":
                        try:
                            audios.append(
                                librosa.load(
                                    io.BytesIO(audio_file_content),
                                    sr=self.engine.model_store[
                                        "Processor"
                                    ].feature_extractor.sampling_rate,
                                )[0]
                            )
                        except:  # noqa: E722
                            return "Error loading audio file, please change file dropper content to appropriate file format"

        inputs = self.engine.model_store["Processor"](
            text=text, audios=audios, return_tensors="pt", padding=True
        )
        inputs.input_ids = inputs.input_ids.to("cuda")
        inputs["input_ids"] = inputs["input_ids"].to("cuda")

        generate_ids = self.engine.model_store["Model"].generate(
            **inputs, max_length=256
        )
        generate_ids = generate_ids[:, inputs.input_ids.size(1) :]

        response = self.engine.model_store["Processor"].batch_decode(
            generate_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        time.sleep(0.1)
        return response
