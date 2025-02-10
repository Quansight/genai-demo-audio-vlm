from pathlib import Path

import tomlkit
from pydantic import Field
from pydantic_settings import BaseSettings


class Config(BaseSettings):
    """
    Class to store configuration for the demo.

    Includes paths to downloaded models, among other thnigs.
    """

    model_path: Path = Field(default_factory=Path)
    aria_model_path: Path = Field(default_factory=Path)
    qwen_audio_model_path: Path = Field(default_factory=Path)

    # I would like to remove the quotes in the return type
    # annotation without getting a NameError
    @classmethod
    def from_file(cls, path: str | Path) -> "Config":
        path = Path(path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"{path} does not exist.")

        with open(path) as file:
            return cls.model_validate(tomlkit.load(file).unwrap())
