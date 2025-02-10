from pathlib import Path
from typing import Annotated

import tomlkit
from pydantic import AfterValidator
from pydantic_settings import BaseSettings

from audiovlm_demo.core.utils import resolve_path

_ResolvedPath = Annotated[Path, AfterValidator(resolve_path)]


class Config(BaseSettings):
    """
    Class to store configuration for the demo.

    Includes paths to downloaded models, among other thnigs.
    """

    model_path: _ResolvedPath
    aria_model_path: _ResolvedPath
    qwen_audio_model_path: _ResolvedPath

    # TODO: I would like to remove the quotes in the return type
    # annotation without getting a NameError
    @classmethod
    def from_file(cls, path: str | Path) -> "Config":
        path = resolve_path(path)
        if not path.is_file():
            raise FileNotFoundError(f"{path} does not exist.")

        with open(path) as file:
            return cls.model_validate(tomlkit.load(file).unwrap())
