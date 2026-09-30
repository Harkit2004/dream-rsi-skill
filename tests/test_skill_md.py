"""``SKILL.md`` must load in a host, and hosts parse its frontmatter strictly (issue #66).

A host lists a skill only after a YAML library has read the block between the
``---`` fences. The description here is long prose that contains ``: ``, which a
plain scalar reads as a nested mapping, so the whole block was refused and the
skill could not be loaded by name even though the file was on disk. Nothing then
noticed: the wording was reviewed as prose, and no test parsed it as YAML.

A regex over the file would have passed the broken version, which is why this
uses a real parser.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
SKILL = REPO / "SKILL.md"

# The Agent Skills limit on ``description``. Hosts truncate or refuse past it.
MAX_DESCRIPTION = 1024

# The name a host loads the skill by, and the directory ``dream_rsi.install`` puts
# it in: lowercase letters and digits, joined by single hyphens.
SKILL_NAME = "dream-rsi"
NAME_PATTERN = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")


def frontmatter_of(text: str) -> Any:
    """The YAML block between the opening ``---`` and the next one, parsed strictly."""
    match = re.match(r"---\r?\n(.*?)\r?\n---\r?\n", text, flags=re.DOTALL)
    assert match is not None, "SKILL.md must open with a --- fenced frontmatter block"
    return yaml.safe_load(match.group(1))


@pytest.fixture(scope="module")
def frontmatter() -> dict[str, Any]:
    parsed = frontmatter_of(SKILL.read_text(encoding="utf-8"))
    assert isinstance(parsed, dict), f"frontmatter must be a mapping, got {type(parsed).__name__}"
    return parsed


def test_the_frontmatter_parses_as_yaml() -> None:
    """The red run for issue #66: at HEAD this raised ``mapping values are not allowed here``."""
    assert isinstance(frontmatter_of(SKILL.read_text(encoding="utf-8")), dict)


def test_the_name_is_the_one_a_host_loads_the_skill_by(frontmatter: dict[str, Any]) -> None:
    name = frontmatter["name"]

    assert name == SKILL_NAME
    assert NAME_PATTERN.fullmatch(name)


def test_the_description_is_a_string_within_the_agent_skills_limit(
    frontmatter: dict[str, Any],
) -> None:
    description = frontmatter["description"]

    assert isinstance(description, str)
    assert description.strip()
    assert len(description) <= MAX_DESCRIPTION
