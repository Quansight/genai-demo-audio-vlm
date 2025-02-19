import io
import re

import librosa
import numpy as np
import panel as pn
from PIL import Image, ImageDraw, ImageFile

from audiovlm_demo.core.components import AudioVLM
from audiovlm_demo.core.utils import resolve_path

pn.extension("filedropper")


class AudioVLMPanel:
    _main_html_path = resolve_path(__file__).parent / "main.html"

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
        self.load_button.on_click(self._load_model_wrapper)

        self.model_info_pane = pn.pane.HTML("<p><b>No Model Loaded</b></p>")

        self.image_pane = pn.pane.Image(sizing_mode="scale_width", max_width=550)
        self.audio_pane = pn.pane.Audio(
            sizing_mode="scale_width", max_width=550, visible=False
        )
        self.image_preview_html = pn.pane.HTML("<p></p>")
        self.file_dropper.param.watch(self.display_image, "value")

        with open(AudioVLMPanel._main_html_path, "r") as f:
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
            callback=self.callback_dispatcher,
            callback_exception="verbose",
        )

        self.full_interface = pn.Column(
            self.header_pane,
            pn.Row(
                self.left_bar,
                self.chat_interface,
            ),
        ).servable()

    def __panel__(self):
        return self.full_interface

    def _load_model_wrapper(self, event):
        self.model_info_pane.object = f"<p>Loading {self.toggle_group.value}...</p>"
        self.engine.load_model(self.toggle_group.value)
        self.model_info_pane.object = f"<p>{self.toggle_group.value} loaded.</p>"

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
                self.audio_pane.sample_rate = audio[1]
                self.audio_pane.object = np.int16(
                    np.array(audio[0], dtype=np.float32) * 32767
                )
                self.audio_pane.visible = True
        else:
            self.image_preview_html.object = "<p></p>"
            self.image_pane.object = None
            self.audio_pane.object = None
            self.audio_pane.visible = False

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

    def overlay_points(self, points_data):
        if self.file_dropper.value:
            file_name, file_content = next(iter(self.file_dropper.value.items()))
            image = Image.open(io.BytesIO(file_content))
        else:
            return

        draw = ImageDraw.Draw(image)
        width, height = image.size

        for point_data in points_data:
            label = point_data["label"]  # noqa: F841
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

    def build_chat_history(self, instance: pn.chat.ChatInterface):
        return [
            {
                "role": utterance.user,
                "content": utterance.object,
            }
            for utterance in instance.objects
        ]

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
            null_and_void = instance.objects.pop()  # noqa: F841

        if self.toggle_group.value in ["Molmo-7B-D-0924", "Molmo-7B-D-0924-4bit"]:
            image_or_error_message = AudioVLMPanel.validate_image_input(
                self.file_dropper
            )
            if isinstance(image_or_error_message, str):
                return image_or_error_message
            else:
                image = image_or_error_message
                del image_or_error_message

            generated_text = self.engine.molmo_callback(
                image=image,
                chat_history=self.build_chat_history(instance),
            )
            return generated_text
        elif self.toggle_group.value == "Aria":
            image_or_error_message = AudioVLMPanel.validate_image_input(
                self.file_dropper
            )
            if isinstance(image_or_error_message, str):
                return image_or_error_message
            else:
                image = image_or_error_message
                del image_or_error_message

            result = self.engine.aria_callback(
                image=image,
                chat_history=self.build_chat_history(instance),
            )
            return result
        elif self.toggle_group.value == "Qwen2-Audio":
            audio_or_error_message = AudioVLMPanel.validate_audio_input(
                self.file_dropper
            )
            if isinstance(audio_or_error_message, str):
                return audio_or_error_message
            else:
                audio_file_content = audio_or_error_message
                del audio_or_error_message
            response = self.engine.aria_callback(
                audio_file_content,
                chat_history=self.build_chat_history(instance),
            )
            return response
