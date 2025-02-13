from audiovlm_demo import AudioVLM, AudioVLMPanel, Config


def main():
    config = Config(
        model_path="allenai/Molmo-7B-D-0924",
        aria_model_path="rhymes-ai/Aria",
        qwen_audio_model_path="Qwen/Qwen2-Audio-7B-Instruct",
    )
    A = AudioVLM(config=config)
    UI = AudioVLMPanel(engine=A)  # noqa: F841


main()
