import panel as pn
from audiovlm_demo.core.components import AudioVLM

import io

from PIL import Image
import librosa
import numpy as np

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
