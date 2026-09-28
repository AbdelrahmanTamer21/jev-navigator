from __future__ import annotations

import subprocess
from pathlib import Path

from jev_navigator.index.code_index import CodeIndex
from jev_navigator.index.imports import imported_names
from jev_navigator.index.spans import Span

ROOT_TSCONFIG = """\
{
  // Aliases for the web app; comments and trailing commas are allowed here.
  "compilerOptions": {
    "baseUrl": ".",
    "paths": {
      "@/*": ["src/*"],
      "@config": ["src/config/index.ts"],
    },
  },
}
"""


def write_files(root: Path, files: dict[str, str]) -> CodeIndex:
    for name, text in files.items():
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_text(text)
    return CodeIndex(root, [name for name in files if not name.endswith(".json")])


def test_an_import_spanning_several_lines_resolves_its_names(tmp_path: Path) -> None:
    # Arrange
    index = write_files(
        tmp_path,
        {
            "src/app/format.ts": "export function formatPrice(cents: number) {\n  return cents / 100;\n}\n",
            "src/app/page.ts": (
                "import {\n  formatPrice,\n  type Price,\n} from './format';\n\n"
                "export function show() {\n  return formatPrice(1);\n}\n"
            ),
        },
    )

    # Act
    imports = index.imports("src/app/page.ts")
    call = index.find_callers("formatPrice")[0]

    # Assert
    assert imports == ("src/app/format.ts",)
    assert call.binding.status == "resolved"
    assert call.binding.target == Span("src/app/format.ts", 1, 3, "formatPrice")


def test_a_path_alias_from_the_nearest_tsconfig_resolves_to_a_scope_file(tmp_path: Path) -> None:
    # Arrange
    index = write_files(
        tmp_path,
        {
            "tsconfig.json": ROOT_TSCONFIG,
            "src/lib/utils.ts": "export function cn(...parts: string[]) {\n  return parts.join(' ');\n}\n",
            "src/config/index.ts": "export const settings = {};\n",
            "src/components/button.ts": (
                'import { cn } from "@/lib/utils";\nimport { settings } from "@config";\n\n'
                "export function button() {\n  return cn('a', 'b');\n}\n"
            ),
        },
    )

    # Act
    imports = index.imports("src/components/button.ts")
    call = index.find_callers("cn")[0]

    # Assert
    assert imports == ("src/lib/utils.ts", "src/config/index.ts")
    assert call.binding.status == "resolved"


def test_the_nearest_tsconfig_wins_and_extends_inherits_paths(tmp_path: Path) -> None:
    # Arrange
    index = write_files(
        tmp_path,
        {
            "tsconfig.json": ROOT_TSCONFIG,
            "src/lib/utils.ts": "export const shared = 1;\n",
            "packages/web/tsconfig.json": '{"compilerOptions": {"paths": {"@/*": ["./app/*"]}}}',
            "packages/web/app/util.ts": "export const local = 1;\n",
            "packages/web/app/page.ts": 'import { local } from "@/util";\n',
            "packages/api/tsconfig.json": '{"extends": "../../tsconfig.json"}',
            "packages/api/handler.ts": 'import { shared } from "@/lib/utils";\n',
        },
    )

    # Act
    web = index.imports("packages/web/app/page.ts")
    api = index.imports("packages/api/handler.ts")

    # Assert
    assert web == ("packages/web/app/util.ts",)
    assert api == ("src/lib/utils.ts",)


def test_a_parenthesised_python_import_over_several_lines_lists_every_name() -> None:
    # Arrange
    source = "from app.jobs import (\n    send_invoice,  # the monthly run\n    refund as give_back,\n)\n"

    # Act
    names = imported_names(source, "app/routes.py")

    # Assert
    assert names == {"send_invoice": "app.jobs", "give_back": "app.jobs"}


def test_an_index_at_an_old_commit_still_reads_the_tsconfig_outside_its_scope(tmp_path: Path) -> None:
    # Arrange
    write_files(
        tmp_path,
        {
            "tsconfig.json": ROOT_TSCONFIG,
            "src/lib/utils.ts": "export const shared = 1;\n",
            "src/app/page.ts": 'import { shared } from "@/lib/utils";\n',
        },
    )
    for command in (
        ["init", "-q"],
        ["add", "."],
        ["-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "c"],
    ):
        subprocess.run(["git", *command], cwd=tmp_path, check=True)

    # Act
    index = CodeIndex.at_commit(tmp_path, "HEAD", prefixes=("src/",))

    # Assert
    assert index.imports("src/app/page.ts") == ("src/lib/utils.ts",)
    assert "tsconfig.json" not in index.files
