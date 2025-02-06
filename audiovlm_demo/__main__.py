from audiovlm_demo import AudioVLM, Config


def main():
    config = Config.from_file("config.toml")
    A = AudioVLM(config=config)
    A.model_cleanup()
    print("needs more reverb")


if __name__ == "__main__":
    main()
