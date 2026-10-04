"""jvn requires no package, and the typesafe extra is optional, so every module must import and
discovery must work on the standard library alone, as after an install without the extra."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

from isolated_jvn import JVN

SOURCE = Path(__file__).resolve().parents[1] / "src"

# Runs before `isolated_jvn`'s own code: an interpreter started with -I -S sees no site-packages,
# so only jvn's source and the standard library can be imported.
IMPORT_EVERY_MODULE = (
    "import importlib, pkgutil, sys\n"
    f"sys.path.insert(0, {str(SOURCE)!r})\n"
    "import jev_navigator\n"
    "for module in pkgutil.walk_packages(jev_navigator.__path__, 'jev_navigator.'):\n"
    "    importlib.import_module(module.name)\n"
)


def _jvn_on_the_standard_library(tmp_path: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    python, _, code = JVN
    return subprocess.run(
        [python, "-I", "-S", "-c", IMPORT_EVERY_MODULE + code, *arguments],
        capture_output=True,
        text=True,
        cwd=tmp_path,
        env={"HOME": str(tmp_path), "PATH": os.environ["PATH"]},
        check=False,
    )


def test_every_module_imports_and_discovery_works_without_any_installed_package(tmp_path):
    # Act
    schema = _jvn_on_the_standard_library(tmp_path, "schema", "find")
    help_text = _jvn_on_the_standard_library(tmp_path, "--help")

    # Assert
    assert schema.returncode == 0, schema.stderr
    assert json.loads(schema.stdout)["title"] == "jvn find request"
    assert help_text.returncode == 0, help_text.stderr
    assert "jvn find" in help_text.stdout
