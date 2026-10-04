"""The environment chain: real environment wins, then the checkout `.env`, then the legacy config."""

from __future__ import annotations

import os

import pytest

from jev_navigator import environment
from jev_navigator.directives.find_code import SearchBudget
from jev_navigator.environment import (
    _env_file,
    checkout_root,
    load_typesafe_environment,
)
from jev_navigator.judgments.thresholds import Thresholds

# Names an untrusted file might try to inject; none is a `jvn` setting.
INJECTED = ("RIPGREP_CONFIG_PATH", "LD_PRELOAD", "EVIL_MARKER")


@pytest.fixture(autouse=True)
def no_exported_injected_names(monkeypatch):
    # conftest clears the tool's own settings; these names are cleared so that a developer's
    # exported RIPGREP_CONFIG_PATH cannot make the injection checks fail.
    for name in INJECTED:
        monkeypatch.delenv(name, raising=False)


def test_the_real_environment_wins_over_every_file(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text("TYPESAFE_API_KEY=from-env-file\n")
    monkeypatch.setenv("TYPESAFE_API_KEY", "from-shell")

    load_typesafe_environment(root=tmp_path)

    assert os.environ["TYPESAFE_API_KEY"] == "from-shell"


def test_the_checkout_env_file_fills_what_the_environment_lacks(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text(
        "# config\n"
        "TYPESAFE_BASE_URL=https://drex.nace.ai\n"
        'TYPESAFE_DEFAULT_MODEL="drex-latest"\n'
        "export TYPESAFE_API_KEY=nace-key\n"
    )

    contributed = load_typesafe_environment(root=tmp_path)

    assert os.environ["TYPESAFE_BASE_URL"] == "https://drex.nace.ai"
    assert os.environ["TYPESAFE_DEFAULT_MODEL"] == "drex-latest"
    assert os.environ["TYPESAFE_API_KEY"] == "nace-key"
    assert contributed == {
        "TYPESAFE_BASE_URL": "https://drex.nace.ai",
        "TYPESAFE_DEFAULT_MODEL": "drex-latest",
        "TYPESAFE_API_KEY": "nace-key",
    }


def test_the_legacy_config_fills_only_what_both_left_open(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text("TYPESAFE_BASE_URL=https://drex.nace.ai\n")
    legacy = _written(
        tmp_path, {"TYPESAFE_API_KEY": "legacy-key", "TYPESAFE_BASE_URL": "https://legacy.example"}
    )
    monkeypatch.setattr(environment, "LEGACY_CONFIG", legacy)

    load_typesafe_environment(root=tmp_path)

    assert os.environ["TYPESAFE_API_KEY"] == "legacy-key"
    assert os.environ["TYPESAFE_BASE_URL"] == "https://drex.nace.ai"  # the checkout `.env` comes first


@pytest.mark.parametrize("has_checkout", [True, False], ids=["run-from-a-checkout", "installed"])
def test_a_missing_key_points_only_at_the_files_jvn_reads(tmp_path, monkeypatch, has_checkout: bool):
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    legacy = tmp_path / "absent-legacy-env"
    monkeypatch.setattr(environment, "checkout_root", lambda: checkout if has_checkout else None)

    with pytest.raises(RuntimeError) as raised:
        load_typesafe_environment({}, legacy=legacy)

    message = str(raised.value)
    assert str(legacy) in message
    assert (str(checkout / ".env") in message) is has_checkout
    assert (".env.example" in message) is has_checkout


def test_a_checkout_env_file_is_read_through_checkout_root(tmp_path, monkeypatch):
    # The default path (no explicit root) reads the `.env` from the tool's own checkout.
    (tmp_path / ".env").write_text("TYPESAFE_API_KEY=checkout-key\n")
    monkeypatch.setattr("jev_navigator.environment.checkout_root", lambda: tmp_path)
    monkeypatch.setattr("jev_navigator.environment.LEGACY_CONFIG", tmp_path / "absent-legacy-env")

    load_typesafe_environment()

    assert os.environ["TYPESAFE_API_KEY"] == "checkout-key"


def test_a_dotenv_outside_a_checkout_is_never_read(tmp_path, monkeypatch):
    # `jvn` installed in the virtual environment of a repository it searches, which ships its own
    # `pyproject.toml` and `.env`: neither the working directory nor that project above the
    # installed module is jev-navigator's checkout, so the repository's `.env` must not configure jvn.
    repository = tmp_path / "repository"
    repository.mkdir()
    (repository / "pyproject.toml").write_text('[project]\nname = "their-app"\n')
    (repository / ".env").write_text("TYPESAFE_API_KEY=attacker-key\nTYPESAFE_BASE_URL=http://attacker\n")
    installed = repository / ".venv/lib/site-packages/jev_navigator/environment.py"
    monkeypatch.chdir(repository)
    monkeypatch.setattr(environment, "__file__", str(installed))
    monkeypatch.setattr(environment, "checkout_root", checkout_root)  # the real lookup, not conftest's
    monkeypatch.setattr(environment, "LEGACY_CONFIG", tmp_path / "absent-legacy-env")

    with pytest.raises(RuntimeError, match=r"TYPESAFE_API_KEY is unset"):
        load_typesafe_environment()

    assert "TYPESAFE_BASE_URL" not in os.environ


def test_a_file_may_set_only_the_tools_own_settings(tmp_path):
    # A key is still loaded, but names outside the tool's namespace are dropped, so a file cannot
    # inject a variable into the tool or the subprocesses (rg, git, ast-grep) it launches.
    (tmp_path / ".env").write_text(
        "TYPESAFE_API_KEY=real-key\n"
        "RIPGREP_CONFIG_PATH=/tmp/evil-rg-config\n"
        "LD_PRELOAD=/tmp/evil.so\n"
        "EVIL_MARKER=owned\n"
    )

    contributed = load_typesafe_environment(root=tmp_path)

    assert os.environ["TYPESAFE_API_KEY"] == "real-key"
    assert contributed == {"TYPESAFE_API_KEY": "real-key"}
    for name in INJECTED:
        assert name not in os.environ


def test_each_file_names_the_names_it_could_not_set_but_never_their_values(tmp_path, capsys):
    checkout_env = tmp_path / ".env"
    checkout_env.write_text("TYPESAFE_API_KEY=real-key\nLD_PRELOAD=/tmp/evil.so\nEVIL_MARKER=owned\n")
    legacy = _written(tmp_path, {"TYPESAFE_BASE_URL": "https://drex.nace.ai", "PATH": "/tmp/evil-bin"})

    load_typesafe_environment({}, root=tmp_path, legacy=legacy)

    notices = capsys.readouterr().err.splitlines()
    assert len(notices) == 2
    checkout_notice, legacy_notice = notices
    assert str(checkout_env) in checkout_notice
    assert "LD_PRELOAD, EVIL_MARKER" in checkout_notice
    assert str(legacy) in legacy_notice
    assert "PATH" in legacy_notice
    for value in ("/tmp/evil.so", "owned", "/tmp/evil-bin", "real-key", "drex.nace.ai"):
        assert value not in "\n".join(notices)


def test_files_holding_only_settings_print_nothing(tmp_path, capsys):
    (tmp_path / ".env").write_text("TYPESAFE_API_KEY=k\n# a comment\n\n")

    load_typesafe_environment({}, root=tmp_path, legacy=_written(tmp_path, {"SYSTEM_ONE_ROUTES": "jev"}))

    assert capsys.readouterr().err == ""


@pytest.mark.parametrize("has_checkout", [True, False], ids=["run-from-a-checkout", "installed"])
def test_a_working_directory_env_is_named_when_jvn_has_no_checkout_to_read(
    tmp_path, monkeypatch, capsys, has_checkout: bool
):
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    searched = tmp_path / "searched"
    searched.mkdir()
    (searched / ".env").write_text("TYPESAFE_API_KEY=their-key\n")
    legacy = _written(tmp_path, {"TYPESAFE_API_KEY": "legacy-key"})
    monkeypatch.setattr(environment, "checkout_root", lambda: checkout if has_checkout else None)
    monkeypatch.chdir(searched)

    load_typesafe_environment({}, legacy=legacy)

    notice = capsys.readouterr().err
    assert (f"not read: {searched / '.env'}" in notice) is not has_checkout
    assert (str(legacy) in notice) is not has_checkout


def test_a_route_setting_is_still_honoured_from_a_file(tmp_path):
    # The allowlist is a namespace, not a fixed list, so decision-model route settings load too.
    (tmp_path / ".env").write_text(
        "TYPESAFE_API_KEY=k\nSYSTEM_ONE_ROUTES=decider\nSYSTEM_ONE_DECIDER_API_KEY=decider-key\n"
    )

    contributed = load_typesafe_environment({}, root=tmp_path, legacy=tmp_path / "absent")

    assert contributed["SYSTEM_ONE_ROUTES"] == "decider"
    assert contributed["SYSTEM_ONE_DECIDER_API_KEY"] == "decider-key"


def test_the_search_thresholds_and_budget_are_still_honoured_from_a_file(tmp_path):
    # `JEV_NAVIGATOR_*` is the tool's own namespace too: the yes/no bars and the search budget
    # keep loading from the checkout `.env`.
    (tmp_path / ".env").write_text(
        "TYPESAFE_API_KEY=k\nJEV_NAVIGATOR_NOUL_YES_AT=0.9\nJEV_NAVIGATOR_MAX_CALLS=40\n"
    )
    environment: dict[str, str] = {}

    load_typesafe_environment(environment, root=tmp_path, legacy=tmp_path / "absent")

    assert Thresholds.from_env(environment).noul_yes_at == 0.9
    assert SearchBudget.from_env(environment).max_calls == 40


@pytest.mark.parametrize(
    ("pyproject", "is_checkout"),
    [
        pytest.param(b'[project]\nname = "jev-navigator"\nversion = "0.1.0"\n', True, id="this-project"),
        pytest.param(b'[project]\nname = "some-other-tool"\n', False, id="another-project"),
        pytest.param(
            b'[project]\nname = "analysis-engine"\n\n[[tool.uv.index]]\nname = "jev-navigator"\n',
            False,
            id="another-project-naming-jev-navigator-elsewhere",
        ),
        pytest.param(b'[project\nname = "jev-navigator"\n', False, id="unparseable-pyproject"),
        pytest.param(b'[project]\nname = "jev-navigator"\n# caf\xe9\n', False, id="pyproject-not-utf-8"),
    ],
)
def test_checkout_root_is_a_project_whose_own_name_is_jev_navigator(
    tmp_path, monkeypatch, pyproject: bytes, is_checkout: bool
):
    # `jvn` installed in a virtual environment inside a project that ships its own `.env`: only
    # jev-navigator's own `[project]` name, in a file that parses as TOML, makes that project the
    # tool's checkout.
    project = tmp_path / "project"
    installed = project / ".venv/lib/site-packages/jev_navigator/environment.py"
    installed.parent.mkdir(parents=True)
    (project / "pyproject.toml").write_bytes(pyproject)
    (project / ".env").write_text("TYPESAFE_BASE_URL=http://attacker\n")
    monkeypatch.setattr(environment, "__file__", str(installed))

    assert checkout_root() == (project if is_checkout else None)


def test_an_installed_jvn_has_no_checkout_whatever_directory_it_runs_in(tmp_path, monkeypatch):
    # `jvn` installed outside a checkout, run inside a repository under analysis that ships its own
    # `.env` and a `pyproject.toml` claiming jev-navigator's name: the working directory is never
    # taken for the tool's checkout, whatever its files say.
    repository = tmp_path / "repository"
    repository.mkdir()
    (repository / ".env").write_text("TYPESAFE_BASE_URL=http://attacker\n")
    (repository / "pyproject.toml").write_text('[project]\nname = "jev-navigator"\n')
    installed = tmp_path / "site-packages/jev_navigator/environment.py"
    monkeypatch.setattr(environment, "__file__", str(installed))
    monkeypatch.chdir(repository)

    assert checkout_root() is None


def test_a_settings_file_that_is_not_utf_8_is_named_in_the_error(tmp_path):
    settings = tmp_path / ".env"
    settings.write_bytes(b"TYPESAFE_API_KEY=caf\xe9\n")

    with pytest.raises(RuntimeError) as raised:
        load_typesafe_environment({}, root=tmp_path, legacy=tmp_path / "absent")

    assert str(settings) in str(raised.value)
    assert isinstance(raised.value.__cause__, UnicodeDecodeError)


def test_env_file_parsing_is_tolerant(tmp_path):
    path = _written(
        tmp_path, {"A": "plain", "B": "quoted"}, extra=["", "# comment", "no equals sign", "=novalue"]
    )
    assert _env_file(path) == {"A": "plain", "B": "quoted"}


def _written(tmp_path, values: dict[str, str], extra: list[str] | None = None):
    path = tmp_path / "legacy-env"
    path.write_text("\n".join([*(f"{k}={v}" for k, v in values.items()), *(extra or [])]) + "\n")
    return path
