"""Environment resolution shared by every entry point: real environment, then `.env`, then the
legacy `~/.config/jvn/env`.

The lookup order is the family convention (`jvn`, `jvr`, comment-tool): a variable already set
in the process environment always wins; the tool's own checkout `.env` fills what is missing; the
legacy config file fills what is still missing. A file never overrides anything that is
already set, so shell exports and CI secrets keep precedence everywhere.

Two rules keep an untrusted repository from configuring the tool through a `.env` it ships:

- The `.env` is read only from this tool's own source checkout (`checkout_root`), never from the
  directory a search happens to run in, so a repository under analysis cannot point the API key at
  another host with its own `.env`. Where the tool's code is installed decides whether there is a
  checkout, not the directory it runs in: `uv run jvn` in the checkout, or an editable install,
  reads that checkout's `.env`; a package installed elsewhere (`uv tool install`, `pipx`) reads no
  `.env` and uses the real environment and `~/.config/jvn/env`.
- A file may set only the tool's own recognised settings (`TYPESAFE_*`, `JEV_NAVIGATOR_*` and
  `SYSTEM_ONE_*`); every other name is ignored and named on stderr. So a config file cannot inject
  an unrelated variable such as `PATH`, `LD_PRELOAD` or `RIPGREP_CONFIG_PATH` into the tool or into
  the `rg`, `git` and `ast-grep` subprocesses it runs.
"""

from __future__ import annotations

import os
import sys
import tomllib
from collections.abc import MutableMapping
from pathlib import Path

# The tool's own settings namespace. Only names under these prefixes are honoured from a file, so a
# file can never inject an operating-system or subprocess variable. `JEV_NAVIGATOR_*` holds the
# search thresholds and budget; `SYSTEM_ONE_*` names the decision-model routes and their keys.
SETTING_PREFIXES = ("TYPESAFE_", "JEV_NAVIGATOR_", "SYSTEM_ONE_")
LEGACY_CONFIG = Path.home() / ".config/jvn/env"


def checkout_root() -> Path | None:
    """This tool's own source checkout, the first directory above this file whose `pyproject.toml`
    names the `jev-navigator` project, or None when `jvn` is installed outside such a checkout.

    The `.env` is trusted only from here. Returning None rather than the current directory is what
    stops an arbitrary working directory's `.env` from configuring the tool.
    """
    for parent in Path(__file__).resolve().parents:
        pyproject = parent / "pyproject.toml"
        if pyproject.is_file() and _names_this_project(pyproject):
            return parent
    return None


def load_typesafe_environment(
    environment: MutableMapping[str, str] | None = None,
    root: Path | None = None,
    legacy: Path | None = None,
) -> dict[str, str]:
    """Fill jvn's settings (`TYPESAFE_*`, `JEV_NAVIGATOR_*` and `SYSTEM_ONE_*`) from the tool's
    checkout `.env`, then the legacy `~/.config/jvn/env`; return what the files contributed.

    Real environment variables win over both files, and a file may set only the tool's own
    settings (`SETTING_PREFIXES`). Every other name in a file is named on stderr, never its value,
    and so is a `.env` in the working directory when there is no checkout to read one from. Raises
    when no source provides an API key. ``root`` overrides the checkout the `.env` is read from;
    passing it opts into reading that directory's `.env`.
    """
    environment = os.environ if environment is None else environment
    root = checkout_root() if root is None else root
    legacy = legacy or LEGACY_CONFIG
    if root is None:
        _note_an_unread_working_directory_env(legacy)
    contributed: dict[str, str] = {}
    sources = ([root / ".env"] if root is not None else []) + [legacy]
    for source in sources:
        for name, value in _settings_in(source).items():
            if not environment.get(name, "").strip() and value:
                environment[name] = value
                contributed[name] = value
    if not environment.get("TYPESAFE_API_KEY", "").strip():
        raise RuntimeError(_missing_key_message(root, legacy))
    return contributed


def _settings_in(source: Path) -> dict[str, str]:
    """The file's names under `SETTING_PREFIXES`; the others are named on stderr and left out."""
    values = _env_file(source)
    ignored = [name for name in values if not _is_setting(name)]
    if ignored:
        prefixes = ", ".join(f"{prefix}*" for prefix in SETTING_PREFIXES)
        _notice(f"ignored {', '.join(ignored)} in {source}: a settings file may set only {prefixes} names")
    return {name: value for name, value in values.items() if _is_setting(name)}


def _note_an_unread_working_directory_env(legacy: Path) -> None:
    unread = Path.cwd() / ".env"
    if unread.is_file():
        _notice(f"not read: {unread}, because jvn reads only its own checkout's .env and {legacy}")


def _notice(message: str) -> None:
    print(f"jvn: {message}", file=sys.stderr)


def _is_setting(name: str) -> bool:
    return name.startswith(SETTING_PREFIXES)


def _missing_key_message(root: Path | None, legacy: Path) -> str:
    """Where to put the key: only the files this run reads, the checkout `.env` when there is one."""
    checkout = f"{root / '.env'} (see .env.example) or " if root is not None else ""
    return f"TYPESAFE_API_KEY is unset: export it, or set it in {checkout}{legacy}"


def _names_this_project(pyproject: Path) -> bool:
    """Whether ``pyproject``'s own `[project]` name is jev-navigator, so a parent project's
    `pyproject.toml` (and its `.env`) higher up the tree is never taken for the tool's checkout. A
    file that cannot be read or parsed as TOML, which must be UTF-8, names no project."""
    try:
        project = tomllib.loads(pyproject.read_bytes().decode()).get("project")
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError):
        return False
    return isinstance(project, dict) and project.get("name") == "jev-navigator"


def _env_file(path: Path) -> dict[str, str]:
    """`KEY=value` lines; `#` comments, `export ` prefixes and quotes handled, a line without a
    name or `=` skipped; nothing overrides."""
    if not path.is_file():
        return {}
    values: dict[str, str] = {}
    for line in _settings_text(path).splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, _, value = line.removeprefix("export ").partition("=")
        if name := name.strip():
            values[name] = value.strip().strip("'\"")
    return values


def _settings_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError as error:
        raise RuntimeError(f"cannot read settings file {path}: {error}") from error
