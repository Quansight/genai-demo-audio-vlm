import panel as pn
from audiovlm_demo.core.components import AudioVLM

pn.extension("filedropper")


class AudioVLMUI:
    def __init__(self, *, engine: AudioVLM):
        self.engine = engine
