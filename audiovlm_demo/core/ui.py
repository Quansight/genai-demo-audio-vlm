import io
import time

import librosa
import numpy as np
import panel as pn
from PIL import Image, ImageDraw, ImageFile

from audiovlm_demo.core.components import AudioVLM

pn.extension("filedropper")


class AudioVLMUI:
    def __init__(self, *, engine: AudioVLM):
        self.engine = engine

        self.file_dropper = pn.widgets.FileDropper(
            accepted_filetypes=["image/*", "audio/*"],
            multiple=False,
            max_file_size="10MB",
            width=300,
            height=95,
        )

        self.toggle_group = pn.widgets.ToggleGroup(
            name="Model Select",
            options=["Molmo-7B-D-0924", "Molmo-7B-D-0924-4bit", "Aria", "Qwen2-Audio"],
            behavior="radio",
        )
        self.load_button = pn.widgets.Button(name="Load Model", button_type="primary")
        self.load_button.on_click(self.engine.load_model)

        self.model_info_pane = pn.pane.HTML("<p><b>No Model Loaded</b></p>")

        self.image_pane = pn.pane.Image(sizing_mode="scale_width", max_width=550)
        self.audio_pane = pn.pane.Audio(
            sizing_mode="scale_width", max_width=550, visible=False
        )
        self.image_preview_html = pn.pane.HTML("<p></p>")
        self.file_dropper.param.watch(self.display_image, "value")

        with open("main.html", "r") as f:
            header_html = f.read().replace("\n", "")

        self.header_pane = pn.pane.HTML(
            header_html,
            width_policy="max",
            sizing_mode="stretch_width",
        )

        self.image_load = pn.Column(
            self.file_dropper,
            pn.Column(
                self.image_preview_html,
                self.audio_pane,
                self.image_pane,
            ),
        )

        self.left_bar = pn.Column(
            self.toggle_group,
            pn.Row(self.load_button, self.model_info_pane),
            self.image_load,
            width=600,
            height=800,
        )

        self.chat_interface = pn.chat.ChatInterface(
            # callback=callback_dispatcher,
            callback=lambda *args, **kwargs: "needs more reverb",
            callback_exception="verbose",
        )

        self.full_interface = pn.Column(
            self.header_pane,
            pn.Row(
                self.left_bar,
                self.chat_interface,
            ),
        ).servable()

    def display_image(self, event):
        if self.file_dropper.value:
            if list(self.file_dropper.mime_type.values())[0].split("/")[0] == "image":
                self.audio_pane.object = None
                self.audio_pane.visible = False
                file_name, file_content = next(iter(self.file_dropper.value.items()))
                image = Image.open(io.BytesIO(file_content))
                self.image_preview_html.object = "<p>Scaled Image Preview:</p>"
                self.image_pane.object = image
            elif list(self.file_dropper.mime_type.values())[0].split("/")[0] == "audio":
                self.image_pane.object = None
                file_name, file_content = next(iter(self.file_dropper.value.items()))
                self.image_preview_html.object = "<p>Audio Track:</p>"
                audio = librosa.load(io.BytesIO(file_content))
                self.audio_pane.sample_rate = sample_rate = audio[1]
                self.audio_pane.object = np.int16(
                    np.array(audio[0], dtype=np.float32) * 32767
                )
                self.audio_pane.visible = True
        else:
            self.image_preview_html.object = "<p></p>"
            self.image_pane.object = None
            self.audio_pane.object = None
            self.audio_pane.visible = False

    def overlay_points(self, points_data):
        if self.file_dropper.value:
            file_name, file_content = next(iter(self.file_dropper.value.items()))
            image = Image.open(io.BytesIO(file_content))
        else:
            return

        draw = ImageDraw.Draw(image)
        width, height = image.size

        for point_data in points_data:
            label = point_data["label"]
            for x_percent, y_percent in point_data["coordinates"]:
                x = (x_percent / 100) * width
                y = (y_percent / 100) * height
                radius = int(height / 55)
                draw.ellipse(
                    (x - radius, y - radius, x + radius, y + radius), fill="blue"
                )

            # Optionally, add label text next to the first coordinate
            # if point_data["coordinates"]:
            #     x, y = point_data["coordinates"][0]
            #     draw.text((x, y - 10), label, fill="yellow")

        self.image_pane.object = image

    # TODO: Improve type annotations
    @classmethod
    def validate_image_input(
        cls, file_dropper: pn.widgets.FileDropper
    ) -> ImageFile.ImageFile | str:
        if (
            file_dropper
            and next(iter(file_dropper.mime_type.values())).split("/")[0] == "image"
        ):
            file_name, file_content = next(iter(file_dropper.value.items()))
            image = Image.open(io.BytesIO(file_content))
            return image
        return "Please upload an image using the file dropper in order to talk over that image."

    # TODO: Improve type annotation
    @classmethod
    def validate_audio_input(cls, file_dropper: pn.widgets.FileDropper):
        if (
            file_dropper.value
            and next(iter(file_dropper.mime_type.values())).split("/")[0] == "audio"
        ):
            _, audio_file_content = next(iter(file_dropper.value.items()))
            return audio_file_content
        else:
            return "Please attach an audio sample of the appropriate file format"

    def callback_dispatcher(
        self, contents: str, user: str, instance: pn.chat.ChatInterface
    ):
        if not self.engine.model_store["Loaded"]:
            instance.send(
                "Loading model; one moment please...",
                user="System",
                respond=False,
            )
            self.engine.load_model(None)
            null_and_void = instance.objects.pop()

        if self.toggle_group.value in ["Molmo-7B-D-0924", "Molmo-7B-D-0924-4bit"]:
            image_or_error_message = AudioVLMUI.validate_image_input(self.file_dropper)
            if isinstance(image_or_error_message, str):
                return image_or_error_message
            else:
                image = image_or_error_message
                del image_or_error_message

            generated_text = self.engine.molmo_callback(
                image=image,
                chat_history=[
                    {
                        "role": utterance.user,
                        "content": utterance.object,
                    }
                    for utterance in instance.objects
                ],
            )
            return generated_text
        elif self.toggle_group.value == "Aria":
            image_or_error_message = AudioVLMUI.validate_image_input(self.file_dropper)
            if isinstance(image_or_error_message, str):
                return image_or_error_message
            else:
                image = image_or_error_message
                del image_or_error_message

            result = self.engine.aria_callback(
                image=image,
                chat_history=[
                    {
                        "role": utterance.user,
                        "content": utterance.object,
                    }
                    for utterance in instance.objects
                ],
            )
            return result
        elif self.toggle_group.value == "Qwen2-Audio":
            audio_or_error_message = AudioVLMUI.validate_audio_input(self.file_dropper)
            if isinstance(audio_or_error_message, str):
                return audio_or_error_message
            else:
                audio_file_content = audio_or_error_message
                del audio_or_error_message

            messages = self.engine.build_chat_history(instance)[-1]
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
                            except:
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
