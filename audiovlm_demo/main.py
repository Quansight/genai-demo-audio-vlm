from audiovlm_demo import AudioVLM, AudioVLMUI, Config


def main():
    config = Config.from_file("config.toml")
    A = AudioVLM(config=config)
    A.model_cleanup()
    UI = AudioVLMUI(engine=A)
    print("needs more reverb")


if __name__ == "__main__":
    main()
