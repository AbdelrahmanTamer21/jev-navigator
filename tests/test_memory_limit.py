"""JVN's memory limit, proved with real processes: a child that really grows, real ast-grep parses,
and real JVN processes contending for real slot files."""

from __future__ import annotations

import mmap
import os
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest
from git_repos import commit_files

from jev_navigator import memory_limit
from jev_navigator.index import tools
from jev_navigator.index.code_index import CodeIndex
from jev_navigator.index.fact_cache import FactCache
from jev_navigator.memory_limit import MemoryLimit, MemoryLimitReachedError

MB = 2**20

GROWS_TO_600_MB = """\
import time
held = []
for _ in range(20):
    held.append(b"x" * (30 * 2**20))
    time.sleep(0.03)
print("grew to 600 MB")
"""

TAKES_A_SLOT_AND_HOLDS_IT = """\
import sys
from pathlib import Path
from jev_navigator.index import tools
tools.git(["--version"], Path.cwd())
print("holding", flush=True)
sys.stdin.read()
"""

TAKES_A_SLOT = """\
import time
from pathlib import Path
from jev_navigator.index import tools
started = time.monotonic()
tools.git(["--version"], Path.cwd())
print(f"took a slot after {time.monotonic() - started:.1f} s", flush=True)
"""

WAITS_FOR_A_SLOT = """\
import time
from pathlib import Path
from jev_navigator.index import tools
print("asking for a slot", flush=True)
started = time.monotonic()
tools.git(["--version"], Path.cwd())
print(f"took a slot after {time.monotonic() - started:.1f} s", flush=True)
"""

STATEMENT = "export function launch(){return 1};"


@pytest.fixture
def commands(monkeypatch: pytest.MonkeyPatch) -> list[list[str]]:
    """Every command a process was started with; each one still runs."""
    started: list[list[str]] = []

    class RecordedPopen(subprocess.Popen):
        def __init__(self, arguments, *args, **kwargs) -> None:
            started.append(list(arguments))
            super().__init__(arguments, *args, **kwargs)

    monkeypatch.setattr(subprocess, "Popen", RecordedPopen)
    return started


def _scans(commands: list[list[str]]) -> list[list[str]]:
    return [command for command in commands if command[:2] == [tools.AST_GREP, "scan"]]


@contextmanager
def _holding(megabytes: int) -> Iterator[None]:
    """This process grows by ``megabytes`` of touched memory, returned to the system on exit. A freed
    Python object can stay in the footprint and be reused, which would hide a later test's growth."""
    held = mmap.mmap(-1, megabytes * MB)
    for page in range(0, megabytes * MB, mmap.PAGESIZE):
        held[page] = 1
    try:
        yield
    finally:
        held.close()


def _limit_the_process(
    monkeypatch: pytest.MonkeyPatch, slots: Path, allowance_mb: int, ceiling_mb: int, wait_seconds: float = 0
) -> None:
    monkeypatch.setenv("JEV_NAVIGATOR_MEMORY_ALLOWANCE_MB", str(allowance_mb))
    monkeypatch.setenv("JEV_NAVIGATOR_MEMORY_CEILING_MB", str(ceiling_mb))
    monkeypatch.setenv("JEV_NAVIGATOR_MEMORY_WAIT_SECONDS", str(wait_seconds))
    monkeypatch.setenv("JEV_NAVIGATOR_MEMORY_SLOTS_DIR", str(slots))


def _jvn_process(script: str, slots: Path, **settings: str) -> subprocess.Popen:
    environment = {
        **os.environ,
        "JEV_NAVIGATOR_MEMORY_SLOTS_DIR": str(slots),
        **{f"JEV_NAVIGATOR_MEMORY_{name.upper()}": value for name, value in settings.items()},
    }
    return subprocess.Popen(
        [sys.executable, "-c", script],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=environment,
    )


def test_a_child_that_grows_past_the_allowance_is_stopped_and_named(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Arrange
    _limit_the_process(monkeypatch, tmp_path / "slots", allowance_mb=150, ceiling_mb=150)
    started = time.monotonic()

    # Act
    with pytest.raises(MemoryLimitReachedError) as stopped:
        tools.run_command([sys.executable, "-c", GROWS_TO_600_MB], tmp_path)

    # Assert
    message = str(stopped.value)
    assert "memory allowance of 150 MB" in message
    assert f"process {os.getpid()}" in message
    assert "stopped" in message and os.path.basename(sys.executable) in message
    assert time.monotonic() - started < 5


def test_a_real_parse_over_the_allowance_is_stopped_and_caches_no_facts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Arrange: one line of 66,000 characters is under the parse guard's estimate, so ast-grep parses
    # it, and its real parse peak is far over a 40 MB allowance.
    bundle = "dist/bundle.js"
    commit_files(
        tmp_path / "repo", {bundle: (STATEMENT * 2000)[:66_000], "src/small.py": "def a():\n    return 1\n"}
    )
    facts = tmp_path / "facts"
    _limit_the_process(monkeypatch, tmp_path / "slots", allowance_mb=40, ceiling_mb=40)
    index = CodeIndex.from_git(tmp_path / "repo", fact_cache_dir=facts)

    # Act
    with pytest.raises(MemoryLimitReachedError, match="ast-grep"):
        index.functions_in(bundle)

    # Assert
    content = (tmp_path / "repo" / bundle).read_bytes()
    assert FactCache(facts).load(bundle, content) is None
    assert bundle not in index.parsed_files
    assert bundle not in index.unavailable_files


def test_growth_past_the_allowance_starts_no_further_process(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, commands: list[list[str]]
) -> None:
    # Arrange
    _limit_the_process(monkeypatch, tmp_path / "slots", allowance_mb=100, ceiling_mb=100)
    tools.git(["--version"], tmp_path)

    # Act
    with _holding(200), pytest.raises(MemoryLimitReachedError, match="memory allowance of 100 MB"):
        tools.git(["--version"], tmp_path)

    # Assert
    assert commands == [["git", "--version"]]


def test_a_second_jvn_process_waits_for_a_full_ceiling_then_refuses_naming_the_holder(tmp_path: Path) -> None:
    # Arrange: one slot, held by a real JVN process.
    slots = tmp_path / "slots"
    one_slot = {"allowance_mb": "512", "ceiling_mb": "512"}
    holder = _jvn_process(TAKES_A_SLOT_AND_HOLDS_IT, slots, **one_slot)
    assert holder.stdout.readline() == "holding\n"

    # Act
    started = time.monotonic()
    refused = _jvn_process(TAKES_A_SLOT, slots, wait_seconds="2", **one_slot)
    _, refusal = refused.communicate(timeout=30)
    waited = time.monotonic() - started
    holder.kill()
    holder.wait()

    # Assert
    assert refused.returncode == 1
    assert "MemoryLimitReachedError" in refusal
    assert f"process {holder.pid} holds" in refusal
    assert "ceiling of 512 MB" in refusal
    assert 2 <= waited < 20


def test_a_waiting_process_takes_the_slot_once_the_holder_ends(tmp_path: Path) -> None:
    # Arrange
    slots = tmp_path / "slots"
    one_slot = {"allowance_mb": "512", "ceiling_mb": "512"}
    holder = _jvn_process(TAKES_A_SLOT_AND_HOLDS_IT, slots, **one_slot)
    assert holder.stdout.readline() == "holding\n"
    waiting = _jvn_process(WAITS_FOR_A_SLOT, slots, wait_seconds="60", **one_slot)
    assert waiting.stdout.readline() == "asking for a slot\n"
    time.sleep(0.5)

    # Act
    holder.stdin.close()
    holder.wait()
    output, errors = waiting.communicate(timeout=60)

    # Assert
    assert waiting.returncode == 0, errors
    assert output.startswith("took a slot after ")
    assert float(output.split()[4]) >= 0.4


def test_a_killed_holder_leaves_no_stale_slot(tmp_path: Path) -> None:
    # Arrange
    slots = tmp_path / "slots"
    one_slot = {"allowance_mb": "512", "ceiling_mb": "512"}
    holder = _jvn_process(TAKES_A_SLOT_AND_HOLDS_IT, slots, **one_slot)
    assert holder.stdout.readline() == "holding\n"

    # Act
    os.kill(holder.pid, signal.SIGKILL)
    holder.wait()
    successor = _jvn_process(TAKES_A_SLOT, slots, wait_seconds="0", **one_slot)
    output, errors = successor.communicate(timeout=30)

    # Assert
    assert successor.returncode == 0, errors
    assert output == "took a slot after 0.0 s\n"


def test_the_ceiling_admits_as_many_processes_as_it_has_slots(tmp_path: Path) -> None:
    # Arrange: two slots of 512 MB.
    slots = tmp_path / "slots"
    two_slots = {"allowance_mb": "512", "ceiling_mb": "1024"}
    first = _jvn_process(TAKES_A_SLOT_AND_HOLDS_IT, slots, **two_slots)
    assert first.stdout.readline() == "holding\n"

    # Act
    second = _jvn_process(TAKES_A_SLOT, slots, wait_seconds="0", **two_slots)
    output, errors = second.communicate(timeout=30)
    first.kill()
    first.wait()

    # Assert
    assert second.returncode == 0, errors
    assert output == "took a slot after 0.0 s\n"


def test_a_holder_whose_slot_file_was_deleted_takes_a_slot_again_before_more_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Arrange: this process holds the only slot, then a /tmp cleaner deletes its file, and a real JVN
    # process takes the slot at the new file.
    slots = tmp_path / "slots"
    _limit_the_process(monkeypatch, slots, allowance_mb=512, ceiling_mb=512, wait_seconds=1)
    tools.git(["--version"], tmp_path)
    (slots / "slot-0").unlink()
    newcomer = _jvn_process(TAKES_A_SLOT_AND_HOLDS_IT, slots, allowance_mb="512", ceiling_mb="512")
    assert newcomer.stdout.readline() == "holding\n"

    # Act
    try:
        with pytest.raises(MemoryLimitReachedError, match=f"process {newcomer.pid} holds") as refused:
            tools.git(["--version"], tmp_path)
    finally:
        newcomer.kill()
        newcomer.wait()

    # Assert
    assert "ceiling of 512 MB" in str(refused.value)


def test_memory_held_before_the_slot_is_not_charged_to_jvn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, commands: list[list[str]]
) -> None:
    # Arrange: a host that already holds 300 MB when it first uses JVN.
    _limit_the_process(monkeypatch, tmp_path / "slots", allowance_mb=100, ceiling_mb=100)

    # Act
    with _holding(300):
        output = tools.git(["--version"], tmp_path)
        again = tools.git(["--version"], tmp_path)

    # Assert
    assert output.startswith("git version") and again == output
    assert commands == [["git", "--version"], ["git", "--version"]]


def test_growth_of_this_process_while_a_child_runs_stops_the_child(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Arrange: the child only sleeps; this process grows past the allowance while it waits for it.
    _limit_the_process(monkeypatch, tmp_path / "slots", allowance_mb=100, ceiling_mb=100)
    tools.git(["--version"], tmp_path)
    grown = threading.Event()

    def grow_while_waiting() -> None:
        time.sleep(0.3)
        with _holding(200):
            grown.set()
            time.sleep(2)

    grower = threading.Thread(target=grow_while_waiting)
    grower.start()
    started = time.monotonic()

    # Act
    with pytest.raises(MemoryLimitReachedError, match="it stopped") as stopped:
        tools.run_command([sys.executable, "-c", "import time; time.sleep(10)"], tmp_path)
    grower.join()

    # Assert
    assert grown.is_set()
    assert time.monotonic() - started < 8
    assert f"process {os.getpid()} grew" in str(stopped.value)


def test_a_slot_folder_that_is_a_symbolic_link_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Arrange: the folder is a predictable path under /tmp, so another user could plant a link there.
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (tmp_path / "slots").symlink_to(elsewhere)
    _limit_the_process(monkeypatch, tmp_path / "slots", allowance_mb=512, ceiling_mb=512)

    # Act
    with pytest.raises(PermissionError, match="symbolic link"):
        tools.git(["--version"], tmp_path)

    # Assert
    assert list(elsewhere.iterdir()) == []


@pytest.mark.parametrize(("allowance_mb", "threads"), [(1024, "3"), (512, "1"), (2048, "7")])
def test_ast_grep_gets_the_threads_its_allowance_affords(
    sample_repo: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    commands: list[list[str]],
    allowance_mb: int,
    threads: str,
) -> None:
    # Arrange
    _limit_the_process(monkeypatch, tmp_path / "slots", allowance_mb=allowance_mb, ceiling_mb=allowance_mb)
    index = CodeIndex.from_git(sample_repo, fact_cache_dir=tmp_path / "facts")

    # Act
    functions = index.functions_in("app/orders.py")

    # Assert
    assert [command[command.index("--threads") + 1] for command in _scans(commands)] == [threads]
    assert [span.name for span in functions] == ["place", "cancel"]


def test_a_ceiling_below_the_allowance_is_refused() -> None:
    with pytest.raises(ValueError, match="ceiling"):
        MemoryLimit.from_env(
            {"JEV_NAVIGATOR_MEMORY_ALLOWANCE_MB": "1024", "JEV_NAVIGATOR_MEMORY_CEILING_MB": "512"}
        )


def test_the_defaults_are_a_gigabyte_per_process_four_slots_and_two_minutes() -> None:
    limit = MemoryLimit.from_env({})

    assert (limit.allowance_mb, limit.ceiling_mb, limit.slots, limit.wait_seconds) == (1024, 4096, 4, 120.0)


def test_the_process_guard_follows_the_settings_it_is_read_with(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    # Arrange
    _limit_the_process(monkeypatch, tmp_path / "a", allowance_mb=512, ceiling_mb=1024)
    first = memory_limit.process_guard()
    _limit_the_process(monkeypatch, tmp_path / "b", allowance_mb=256, ceiling_mb=256)

    # Act
    second = memory_limit.process_guard()

    # Assert
    assert first is not second
    assert (second.limit.allowance_mb, second.directory) == (256, tmp_path / "b")
