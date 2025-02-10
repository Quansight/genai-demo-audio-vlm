from audiovlm_demo import AudioVLM, AudioVLMPanel, Config


def main():
    config = Config.from_file("config.toml")
    A = AudioVLM(config=config)
    UI = AudioVLMPanel(engine=A)  # noqa: F841


main()
