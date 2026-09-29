"""Install this repository as a skill for a coding-agent host (issue #74).

A host needs two separate things: the *skill directory* — ``SKILL.md`` and whatever
it links to — and the Python the agent will call, which is the ``dream_rsi`` package
(``pip install``). Cloning the repo into the host's skills directory supplies the
first by accident and drags in everything else: ``src/``, ``tests/``, the paper's PDF
and ``.github/``, which the host project's own protected-path, format and lint gates
then fail on. So this writes a *thin* directory instead: ``SKILL.md`` and
``references/method.md``, the one file it links, and nothing else.

    python -m dream_rsi.install --host {claude,opencode,cursor} [--project DIR] [--force]

Where each host looks (checked against each one's current documentation):

=========== ============================== ==================
Host        user-global (default)          project (``--project``)
=========== ============================== ==================
Claude Code ``~/.claude/skills/``          ``.claude/skills/``
OpenCode    ``~/.config/opencode/skills/`` ``.opencode/skills/``
Cursor      ``~/.cursor/skills/``          ``.cursor/skills/``
=========== ============================== ==================

User-global is the default on purpose: a project-local copy sits inside the host
project's own gates, and the skill is not part of that project. OpenCode's
user-global directory follows ``XDG_CONFIG_HOME`` when it is set, as OpenCode does.

**One source.** The files are the repository's own ``SKILL.md`` and
``references/method.md``, never a second copy: in a wheel they are package data
(``dream_rsi/skill/``, put there at build time by ``pyproject.toml``), so ``pip
install git+https://…`` followed by this command works without a clone, and from a
checkout they are read from its root.

**Nothing is overwritten by surprise.** An existing skill directory that differs from
what would be written — a local edit, or the whole-repo clone the obvious install
leaves — is refused unless ``--force``. One that is already identical is left alone.
``--force`` replaces the directory, and replaces a *symlink* by removing the link and
not what it points at: skill authors symlink a checkout into place.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from collections.abc import Sequence
from pathlib import Path

__all__ = ["HOSTS", "SKILL_NAME", "InstallError", "install", "main"]

# What a host loads the skill by, and the directory it is installed into: the same
# word, because Cursor requires the frontmatter name to match its folder.
SKILL_NAME = "dream-rsi"

# What is installed, relative to the source and to the skill directory alike.
_FILES = ("SKILL.md", "references/method.md")

HOSTS = ("claude", "opencode", "cursor")


class InstallError(RuntimeError):
    """The skill could not be installed, and why."""


def install(
    host: str,
    *,
    project: str | Path | None = None,
    home: str | Path | None = None,
    force: bool = False,
) -> Path:
    """Install the skill for ``host`` and return the directory it is in.

    ``project`` installs into that project's skills directory instead of the
    user-global one; ``home`` stands in for the user's home, which is what makes this
    testable without touching a real host.
    """
    if host not in HOSTS:
        raise InstallError(f"unknown host {host!r}: choose one of {', '.join(HOSTS)}")
    source = _source()
    wanted = {name: (source / name).read_bytes() for name in _FILES}
    target = _skill_directory(host, project=project, home=home)

    if target.is_symlink() or target.exists():
        if _holds(target, wanted):
            return target
        if not force:
            raise InstallError(
                f"{target} already exists and differs from what would be installed; "
                "rerun with --force to replace it"
            )
        _remove(target)

    for name, data in wanted.items():
        path = target / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    return target


def main(argv: Sequence[str] | None = None) -> int:
    """The command: install, print where, and refuse in one line."""
    parser = argparse.ArgumentParser(
        prog="python -m dream_rsi.install",
        description=f"Install the {SKILL_NAME} skill (SKILL.md and references/method.md) "
        "for a coding-agent host, without cloning this repository into it.",
    )
    parser.add_argument("--host", required=True, choices=HOSTS, help="the host to install for")
    parser.add_argument(
        "--project",
        type=Path,
        default=None,
        metavar="DIR",
        help="install into DIR's own skills directory instead of the user-global one",
    )
    parser.add_argument(
        "--force", action="store_true", help="replace an existing skill directory that differs"
    )
    args = parser.parse_args(argv)

    try:
        written = install(args.host, project=args.project, force=args.force)
    except InstallError as exc:
        print(exc, file=sys.stderr)
        return 2
    print(f"installed {SKILL_NAME} for {args.host}: {written}")
    return 0


def _skill_directory(host: str, *, project: str | Path | None, home: str | Path | None) -> Path:
    """Where ``host`` looks for the skill called :data:`SKILL_NAME`."""
    if project is not None:
        base = Path(project)
        relative = {
            "claude": (".claude", "skills"),
            "opencode": (".opencode", "skills"),
            "cursor": (".cursor", "skills"),
        }[host]
    else:
        base = Path(home) if home is not None else Path.home()
        relative = {
            "claude": (".claude", "skills"),
            "opencode": (".config", "opencode", "skills"),
            "cursor": (".cursor", "skills"),
        }[host]
        # The ambient environment applies to the real home only: an explicit ``home``
        # is a caller saying where the user's files are, and must not be overridden by
        # wherever this process happens to be running.
        configured = os.environ.get("XDG_CONFIG_HOME")
        if host == "opencode" and configured and home is None:
            base, relative = Path(configured), ("opencode", "skills")
    return base.joinpath(*relative, SKILL_NAME)


def _source() -> Path:
    """The directory holding the skill files: the wheel's package data, else the checkout.

    Package data first, so an installed package never reaches for a repository that is
    not there; the checkout second, which is what an editable install and a clone are.
    """
    packaged = Path(__file__).resolve().parent / "skill"
    checkout = Path(__file__).resolve().parents[2]
    for candidate in (packaged, checkout):
        if all((candidate / name).is_file() for name in _FILES):
            return candidate
    raise InstallError(
        f"cannot find {', '.join(_FILES)}: neither the package data ({packaged}) nor a "
        f"checkout ({checkout}) has them"
    )


def _holds(directory: Path, wanted: dict[str, bytes]) -> bool:
    """Whether ``directory`` is exactly what installing would write: these files, no others."""
    if directory.is_symlink() or not directory.is_dir():
        return False
    present = {
        path.relative_to(directory).as_posix(): path
        for path in directory.rglob("*")
        if path.is_file()
    }
    return set(present) == set(wanted) and all(
        present[name].read_bytes() == data for name, data in wanted.items()
    )


def _remove(target: Path) -> None:
    """Remove ``target``: a link is unlinked, never followed; a directory is deleted."""
    if target.is_symlink() or target.is_file():
        target.unlink()
    else:
        shutil.rmtree(target)


if __name__ == "__main__":  # pragma: no cover - exercised as a subprocess
    raise SystemExit(main())
