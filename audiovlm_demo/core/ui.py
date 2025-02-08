import io

import librosa
import numpy as np
import panel as pn
from PIL import Image, ImageDraw

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
