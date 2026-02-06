from pathlib import Path
from typing import Optional

import warnings
def _resolve_include_path(fn: str, *, base_dir: Optional[Path]) -> Path:
    """
    Resolve include("x.pl") path.
    - If fn is absolute: use it.
    - Else: resolve relative to base_dir (preferred), fallback to cwd.
    """
    p = Path(fn)
    if p.is_absolute():
        return p.resolve()
    if base_dir is None:
        return (Path.cwd() / p).resolve()
    return (base_dir / p).resolve()


def _resolve_include_path(file_dir: Optional[Path]) -> Path:
    """
    Resolve include("x.pl") path.
    - If file_dir is absolute: get its base directoy.
    - If file_dir is relative: resolve against base_dir or cwd.
    - If file_dir is None: return cwd.
    """
    p = Path(file_dir)
    cwd = Path.cwd()
    if p.is_absolute():
        return p.parent
    else:
        base_dir = cwd
        if file_dir is not None:
            base_dir = (cwd / p).parent
        return base_dir

class PathSolver:
    """
    the input path_or_src can be:
    - a file path (absolute or relative)
    - a raw source code string
    - or None

    """
    def get_dir(self, base_dir:Optional[str]=None) -> Path:
        """Get the base directory of the current context."""
        if base_dir is None:
            return Path.cwd()
        if Path(base_dir).is_dir():
            return Path(base_dir).resolve()
        elif self.get_path(base_dir):
            return self.get_path(base_dir).parent
        else:
            raise ValueError(f"Invalid base directory: {base_dir}")

    def get_filename(self, file_path:str) -> str:
        """Get the file name from the given file path."""
        p = self.get_path(file_path)
        return p.name

    def get_path(self, file_path:str, *, give_base_dir:str) -> Path:
        """Resolve the file path."""
        if give_base_dir:
            base_dir = Path(give_base_dir)
            p = (base_dir / file_path).resolve()
            if p.is_file():
                return p
        p = Path(file_path).resolve()
        if p.is_file():
            return p
        raise FileNotFoundError(f"File not found: {file_path}")

    def get_base_dir(self, file_path: Optional[str]=None) -> Path:
        """Get the base directory of the given file path."""
        if file_path is None:
            return Path.cwd()
        p = self.get_path(file_path)
        return p.parent

    def read_content(self, file_path:str) -> Optional[str]:
        """Read the content from the file if path is set."""
        p = self.get_path(file_path)
        return p.read_text(encoding="utf-8")

    def is_valid_path(self, path_or_src: str) -> bool:
        """Check if the given string is a valid file path."""
        try:
            _ = self.get_path(path_or_src)
            return True
        except:
            return False
            