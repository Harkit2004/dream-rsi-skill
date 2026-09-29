"""Installing this repo as a skill, without installing the repo into the host (issue #74).

The README covered ``pip install -e .`` and the quickstart and nothing said how to
install the repo *as a skill*. Tried with OpenCode, the natural move was to clone it
into the host project's ``.opencode/skills/dream-rsi/``, which put ``src/``,
``tests/``, the paper's PDF and ``.github/`` inside that project — and its own
protected-path, format and lint gates then failed on those files. A host needs two
separate things: the skill directory, and the package the agent will call.

Every test installs into a temporary directory. Nothing here touches a real host.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest
from test_skill_md import NAME_PATTERN, SKILL_NAME, frontmatter_of

from dream_rsi.install import InstallError, install

REPO = Path(__file__).resolve().parents[1]

# Where each host looks, relative to a home directory or a project: what each host's
# current documentation says (checked for this issue against Claude Code's, OpenCode's
# and Cursor's), and what the issue's table says.
USER_GLOBAL = {
    "claude": ".claude/skills",
    "opencode": ".config/opencode/skills",
    "cursor": ".cursor/skills",
}
PROJECT = {
    "claude": ".claude/skills",
    "opencode": ".opencode/skills",
    "cursor": ".cursor/skills",
}

SKILL_FILES = {"SKILL.md", "references/method.md"}


@pytest.fixture(autouse=True)
def _hermetic(monkeypatch: pytest.MonkeyPatch) -> None:
    """No test here may depend on, or write to, the environment it happens to run in.

    CI sets ``XDG_CONFIG_HOME``; a developer's shell often does not. A test that passed
    on one and wrote to the real config directory on the other is what this prevents.
    """
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)


def _files(root: Path) -> set[str]:
    return {path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()}


def _snapshot(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }


@pytest.mark.parametrize("host", sorted(USER_GLOBAL))
def test_installing_into_a_home_writes_the_skill_at_the_hosts_path_and_nothing_else(
    tmp_path: Path, host: str
) -> None:
    """Issue #74's first "tests first"."""
    written = install(host, home=tmp_path)

    assert written == tmp_path / USER_GLOBAL[host] / SKILL_NAME
    assert _files(tmp_path) == {f"{USER_GLOBAL[host]}/{SKILL_NAME}/{name}" for name in SKILL_FILES}


@pytest.mark.parametrize("host", sorted(PROJECT))
def test_installing_into_a_project_writes_only_inside_the_hosts_skill_directory(
    tmp_path: Path, host: str
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    (project / "app.py").write_text("print('the host project')\n", encoding="utf-8")

    written = install(host, project=project)

    assert written == project / PROJECT[host] / SKILL_NAME
    assert _files(project) == {
        "app.py",
        *(f"{PROJECT[host]}/{SKILL_NAME}/{name}" for name in SKILL_FILES),
    }


def test_opencode_follows_xdg_config_home_for_the_real_user_global_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = tmp_path / "elsewhere"
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(config))

    written = install("opencode")

    assert written == config / "opencode" / "skills" / SKILL_NAME


def test_an_explicit_home_is_not_overridden_by_the_environment_the_process_runs_in(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "ambient"))

    written = install("opencode", home=tmp_path / "home")

    assert written == tmp_path / "home" / ".config" / "opencode" / "skills" / SKILL_NAME
    assert not (tmp_path / "ambient").exists()


def test_the_installed_skill_is_the_one_source_file_and_loads_by_its_directory_name(
    tmp_path: Path,
) -> None:
    written = install("claude", home=tmp_path)

    text = (written / "SKILL.md").read_text(encoding="utf-8")
    assert text == (REPO / "SKILL.md").read_text(encoding="utf-8")
    assert (written / "references" / "method.md").read_bytes() == (
        REPO / "references" / "method.md"
    ).read_bytes()
    # #66's check, on the installed copy: it parses, and a host that requires the
    # name to match the folder (Cursor does) finds that it does.
    frontmatter = frontmatter_of(text)
    assert frontmatter["name"] == written.name == SKILL_NAME
    assert NAME_PATTERN.fullmatch(frontmatter["name"])


def test_everything_the_installed_skill_points_at_was_installed_with_it(tmp_path: Path) -> None:
    """A thin skill directory is only usable if it does not link out of itself."""
    written = install("claude", home=tmp_path)
    text = (written / "SKILL.md").read_text(encoding="utf-8")

    links = re.findall(r"\]\(([^)#]+)(?:#[^)]*)?\)", text)
    references = re.findall(r"`(references/[\w./-]+)`", text)
    relative = [target for target in (*links, *references) if "://" not in target]

    assert relative, "the skill points at nothing, which the test would otherwise pass on"
    for target in relative:
        assert (written / target).exists(), f"SKILL.md points at {target}, which was not installed"


def test_installing_over_a_copy_that_differs_is_refused_without_force(tmp_path: Path) -> None:
    """Issue #74's third: including the whole-repo clone the obvious way of installing leaves."""
    written = install("claude", home=tmp_path)
    (written / "SKILL.md").write_text("a local edit\n", encoding="utf-8")
    (written / "src").mkdir()
    (written / "src" / "extra.py").write_text("# whatever else was cloned in\n", encoding="utf-8")
    before = _snapshot(tmp_path)

    with pytest.raises(InstallError, match="--force"):
        install("claude", home=tmp_path)

    assert _snapshot(tmp_path) == before


def test_a_whole_repo_clone_is_not_mistaken_for_the_thin_install(tmp_path: Path) -> None:
    """Its SKILL.md and method.md are identical to ours; it is everything else that differs.

    This is what cloning the repo into the skills directory leaves, and the very
    thing that put ``src/``, ``tests/`` and the PDF inside a host project.
    """
    written = install("claude", home=tmp_path)
    (written / "tests").mkdir()
    (written / "tests" / "test_x.py").write_text("# cloned in\n", encoding="utf-8")

    with pytest.raises(InstallError, match="--force"):
        install("claude", home=tmp_path)


def test_force_replaces_a_copy_that_differs(tmp_path: Path) -> None:
    written = install("claude", home=tmp_path)
    (written / "SKILL.md").write_text("a local edit\n", encoding="utf-8")
    (written / "src").mkdir()
    (written / "src" / "extra.py").write_text("# leftover\n", encoding="utf-8")

    install("claude", home=tmp_path, force=True)

    assert _files(written) == SKILL_FILES
    assert (written / "SKILL.md").read_bytes() == (REPO / "SKILL.md").read_bytes()


def test_installing_twice_is_not_an_error(tmp_path: Path) -> None:
    first = install("cursor", home=tmp_path)

    second = install("cursor", home=tmp_path)

    assert second == first
    assert _files(first) == SKILL_FILES


def test_force_over_a_symlink_replaces_the_link_and_leaves_what_it_pointed_at(
    tmp_path: Path,
) -> None:
    """Skill authors symlink a checkout into place, and ``--force`` must not delete it."""
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    (checkout / "precious.txt").write_text("keep me\n", encoding="utf-8")
    target = tmp_path / USER_GLOBAL["claude"] / SKILL_NAME
    target.parent.mkdir(parents=True)
    target.symlink_to(checkout, target_is_directory=True)

    with pytest.raises(InstallError, match="--force"):
        install("claude", home=tmp_path)
    install("claude", home=tmp_path, force=True)

    assert (checkout / "precious.txt").read_text(encoding="utf-8") == "keep me\n"
    assert not target.is_symlink()
    assert _files(target) == SKILL_FILES


def test_the_command_prints_the_path_it_wrote_and_refuses_in_one_line(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    command = [sys.executable, "-m", "dream_rsi.install", "--host", "opencode"]

    first = subprocess.run(
        [*command, "--project", str(project)], capture_output=True, text=True, check=False
    )
    (project / PROJECT["opencode"] / SKILL_NAME / "SKILL.md").write_text("edited\n")
    second = subprocess.run(
        [*command, "--project", str(project)], capture_output=True, text=True, check=False
    )

    assert first.returncode == 0, first.stderr
    assert str(project / PROJECT["opencode"] / SKILL_NAME) in first.stdout
    assert second.returncode != 0
    assert "--force" in second.stderr
    assert "Traceback" not in second.stderr


def test_a_host_is_required(tmp_path: Path) -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "dream_rsi.install", "--project", str(tmp_path)],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode != 0
    assert "--host" in completed.stderr


def test_a_wheel_built_from_the_repo_carries_the_skill_and_installs_from_it(
    tmp_path: Path,
) -> None:
    """Issue #74's last: ``pip install`` from git, then the install command, with no clone.

    Built with the project's own backend, so what is checked is the wheel this
    repository's ``pyproject.toml`` describes: the skill files are in it, the tests and
    the paper are not, and the installer finds them where a wheel puts them rather than
    by reaching back into a checkout that is not there.
    """
    dist = tmp_path / "dist"
    built = subprocess.run(
        [sys.executable, "-m", "hatchling", "build", "-t", "wheel", "-d", str(dist)],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
    )
    assert built.returncode == 0, built.stdout + built.stderr
    [wheel] = dist.glob("*.whl")

    with zipfile.ZipFile(wheel) as archive:
        names = set(archive.namelist())
        assert "dream_rsi/skill/SKILL.md" in names
        assert "dream_rsi/skill/references/method.md" in names
        assert archive.read("dream_rsi/skill/SKILL.md") == (REPO / "SKILL.md").read_bytes()
        assert not [name for name in names if name.startswith(("tests/", ".github/"))]
        assert not [name for name in names if name.endswith(".pdf")]
        site = tmp_path / "site"
        archive.extractall(site)

    project = tmp_path / "project"
    project.mkdir()
    # Only the unpacked wheel on the path: the repository's own ``src`` must not be
    # what answers, or this would pass with the skill files missing from the wheel.
    environment = {**os.environ, "PYTHONPATH": str(site)}
    installed = subprocess.run(
        [sys.executable, "-m", "dream_rsi.install", "--host", "claude", "--project", str(project)],
        capture_output=True,
        text=True,
        check=False,
        env=environment,
        cwd=tmp_path,
    )

    assert installed.returncode == 0, installed.stderr
    assert _files(project) == {f"{PROJECT['claude']}/{SKILL_NAME}/{name}" for name in SKILL_FILES}
