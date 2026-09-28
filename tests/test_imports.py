from __future__ import annotations

from pathlib import Path

from git_repos import commit_all, write_files

from jev_navigator.index.code_index import CodeIndex
from jev_navigator.index.imports import (
    directly_exported_names,
    imported_modules,
    imported_names,
    reexported_names,
)
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


def indexed(root: Path, files: dict[str, str]) -> CodeIndex:
    """Writes ``files`` and indexes every one except the JSON configs."""
    write_files(root, files)
    return CodeIndex(root, [name for name in files if not name.endswith(".json")])


def test_an_import_spanning_several_lines_resolves_its_names(tmp_path: Path) -> None:
    # Arrange
    index = indexed(
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


def test_script_reexports_name_the_source_module_and_exported_names() -> None:
    # Arrange
    source = """\
export * from "./orders";
export { refund, createOrder as placeOrder } from "./commands";
import { ignored } from "./ignored";
"""

    # Act
    exports = reexported_names(source, "src/services/index.ts")

    # Assert
    assert exports == (
        (None, "./orders"),
        (frozenset({"refund", "placeOrder"}), "./commands"),
    )
    assert reexported_names("from .orders import create_order", "app/__init__.py") == ()


def test_script_export_surface_excludes_private_and_default_declarations() -> None:
    # Arrange
    source = """\
function privateRun() {}
export function run() {}
export const READY = true;
const local = true;
export { local as publicLocal };
export default function defaultRun() {}
"""

    # Act and assert
    assert directly_exported_names(source, "src/service.ts") == frozenset({"run", "READY", "publicLocal"})


def test_a_call_imported_through_a_barrel_has_a_proven_target(tmp_path: Path) -> None:
    # Arrange
    index = indexed(
        tmp_path,
        {
            "src/services/orders.ts": "export function createOrder() { return 1; }\n",
            "src/services/index.ts": 'export * from "./orders";\n',
            "src/routes.ts": (
                'import { createOrder } from "./services";\n'
                "export function postOrder() { return createOrder(); }\n"
            ),
        },
    )

    # Act
    call = index.find_callers("createOrder")[0]

    # Assert
    assert call.binding.status == "resolved"
    assert call.binding.target == Span("src/services/orders.ts", 1, 1, "createOrder")


def test_two_wildcard_reexports_with_the_same_name_stay_ambiguous(tmp_path: Path) -> None:
    # Arrange
    index = indexed(
        tmp_path,
        {
            "src/services/one.ts": "export function run() { return 1; }\n",
            "src/services/two.ts": "export function run() { return 2; }\n",
            "src/services/index.ts": 'export * from "./one";\nexport * from "./two";\n',
            "src/page.ts": 'import { run } from "./services";\nrun();\n',
        },
    )

    # Act
    call = index.find_callers("run")[0]

    # Assert
    assert call.binding.status == "candidate"
    assert call.binding.target is None


def test_a_private_name_behind_a_wildcard_barrel_is_not_a_proven_import(tmp_path: Path) -> None:
    # Arrange
    index = indexed(
        tmp_path,
        {
            "src/services/one.ts": "function run() { return 1; }\n",
            "src/services/index.ts": 'export * from "./one";\n',
            "src/page.ts": 'import { run } from "./services";\nrun();\n',
        },
    )

    # Act
    call = index.find_callers("run")[0]

    # Assert
    assert call.binding.status == "candidate"
    assert call.binding.target is None


def test_a_path_alias_from_the_nearest_tsconfig_resolves_to_a_scope_file(tmp_path: Path) -> None:
    # Arrange
    index = indexed(
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
    index = indexed(
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
    commit_all(tmp_path)

    # Act
    index = CodeIndex.at_commit(tmp_path, "HEAD", prefixes=("src/",))

    # Assert
    assert index.imports("src/app/page.ts") == ("src/lib/utils.ts",)
    assert "tsconfig.json" not in index.files


def test_a_multi_line_import_with_comments_inside_keeps_its_module_and_names() -> None:
    # Arrange
    source = "import {\n  a, // the first\n  /* the second */ b,\n} from './x';\n"

    # Act
    modules = imported_modules(source, "src/p.ts")
    names = imported_names(source, "src/p.ts")

    # Assert
    assert modules == ["./x"]
    assert names == {"a": "./x", "b": "./x"}


def test_an_import_after_a_statement_without_semicolon_keeps_its_names() -> None:
    # Arrange
    source = "export default Foo\nimport { a } from './a'\n"

    # Act
    names = imported_names(source, "src/p.ts")

    # Assert
    assert names == {"a": "./a"}


def test_the_alias_with_the_longest_prefix_wins_like_typescript(tmp_path: Path) -> None:
    # Arrange
    index = indexed(
        tmp_path,
        {
            "tsconfig.json": (
                '{"compilerOptions": {"baseUrl": ".", "paths": '
                '{"@/*": ["src/*"], "@/components/*": ["src/ui/*"]}}}'
            ),
            "src/components/button.ts": "export const wrong = 1;\n",
            "src/ui/button.ts": "export const right = 1;\n",
            "src/page.ts": 'import { right } from "@/components/button";\n',
        },
    )

    # Act
    imports = index.imports("src/page.ts")

    # Assert
    assert imports == ("src/ui/button.ts",)


def test_an_exact_alias_wins_over_a_wildcard_like_typescript(tmp_path: Path) -> None:
    # Arrange
    index = indexed(
        tmp_path,
        {
            "tsconfig.json": (
                '{"compilerOptions": {"paths": {"@*": ["./lib/*"], "@config": ["./cfg/main.ts"]}}}'
            ),
            "lib/config.ts": "export const wrong = 1;\n",
            "cfg/main.ts": "export const right = 1;\n",
            "page.ts": 'import { right } from "@config";\n',
        },
    )

    # Act
    imports = index.imports("page.ts")

    # Assert
    assert imports == ("cfg/main.ts",)


def test_a_config_whose_base_lies_above_the_index_root_leaves_aliases_unknown(tmp_path: Path) -> None:
    # Arrange
    write_files(
        tmp_path,
        {
            "tsconfig.base.json": (
                '{"compilerOptions": {"baseUrl": ".", "paths": {"@/*": ["packages/web/src/*"]}}}'
            ),
            "packages/web/tsconfig.json": '{"extends": "../../tsconfig.base.json"}',
            "packages/web/src/util.ts": "export function u() {\n  return 1;\n}\n",
            "packages/web/src/page.ts": (
                'import { u } from "@/util";\n\nexport function p() {\n  return u();\n}\n'
            ),
        },
    )
    index = CodeIndex(tmp_path / "packages/web", ["src/util.ts", "src/page.ts"])

    # Act
    imports = index.imports("src/page.ts")
    call = index.find_callers("u")[0]

    # Assert
    assert imports == ()
    assert call.binding.status == "candidate"


def test_configs_outside_the_root_or_behind_a_symbolic_link_are_not_read(tmp_path: Path) -> None:
    # Arrange
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "evil.json").write_text(
        '{"compilerOptions": {"baseUrl": ".", "paths": {"@/*": ["../repo/src/*"]}}}'
    )
    (outside / "tsconfig.json").write_text(
        '{"compilerOptions": {"baseUrl": ".", "paths": {"@/*": ["../repo/src/*"]}}}'
    )
    root = tmp_path / "repo"
    index = indexed(
        root,
        {
            "tsconfig.json": '{"extends": "../outside/evil.json"}',
            "src/util.ts": "export const u = 1;\n",
            "src/page.ts": 'import { u } from "@/util";\n',
        },
    )
    (root / "linked").mkdir()
    (root / "linked/tsconfig.json").symlink_to(outside / "tsconfig.json")
    (root / "linked/page.ts").write_text('import { u } from "@/util";\n')
    linked = CodeIndex(root, ["src/util.ts", "src/page.ts", "linked/page.ts"])

    # Act
    through_extends = index.imports("src/page.ts")
    through_link = linked.imports("linked/page.ts")

    # Assert
    assert through_extends == ()
    assert through_link == ()
