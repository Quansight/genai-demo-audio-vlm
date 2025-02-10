from pathlib import Path


def resolve_path(path: Path | str) -> Path:
    return Path(path).expanduser().resolve()
