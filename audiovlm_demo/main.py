from audiovlm_demo import AudioVLM, AudioVLMUI, Config


def main():
    config = Config.from_file("config.toml")
    A = AudioVLM(config=config)
    UI = AudioVLMUI(engine=A)  # noqa: F841


main()
