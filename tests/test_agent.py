"""The coding agent adapter protocol (issue #3)."""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

from dream_rsi.adapters.agent import AgentContext, Artifact, CodingAgent
from dream_rsi.adapters.fake_agent import FakeAgent
from dream_rsi.adapters.toy_evaluator import ToyEvaluator
from dream_rsi.tree import Node

ROOT = Node(id="n000000")


def context(workspace: Path, /, **overrides: object) -> AgentContext:
    """A context resuming from the root, with any field swapped out by keyword."""
    fields: dict[str, object] = {
        "problem": "write the shortest program that solves it",
        "workspace": workspace,
        "history": (ROOT,),
        "observations": (),
        "seed": 0,
    }
    fields.update(overrides)
    return AgentContext(**fields)  # type: ignore[arg-type]


# One variant generator per context field, used to check that the agent's
# output actually depends on that field.
VARIANTS = {
    "problem": lambda i: {"problem": f"problem {i}"},
    "workspace": lambda i: {"workspace": Path(f"/nonexistent/attempt_{i}")},
    "history": lambda i: {
        "history": (
            ROOT,
            Node(id=f"n{i + 1:06d}", parent_id=ROOT.id, artifact=f"attempt {i}", score=float(i)),
        )
    },
    "observations": lambda i: {"observations": (f"observation {i}",)},
    "seed": lambda i: {"seed": i},
}


def test_fake_agent_satisfies_the_protocol(tmp_path):
    # A static type checker accepts the annotated binding; the isinstance check
    # is the runtime half of the same statement.
    agent: CodingAgent = FakeAgent()
    assert isinstance(agent, CodingAgent)
    artifact = agent.propose(context(tmp_path))
    assert isinstance(artifact, Artifact)
    assert artifact.content.strip()


def test_the_protocol_rejects_an_agent_that_cannot_propose():
    class NotAnAgent:
        def generate(self, context: AgentContext) -> Artifact:
            return Artifact(content="")

    assert not isinstance(NotAnAgent(), CodingAgent)


def test_the_protocol_exposes_nothing_but_propose():
    # AGENTS.md rule 6: the orchestration layer improves, the agent stays
    # untouched. A second member here would be the handle to reach around the
    # adapter and change the agent's weights, prompt internals, or decoding.
    assert {name for name in vars(CodingAgent) if not name.startswith("_")} == {"propose"}


def test_the_same_context_gives_the_same_artifact(tmp_path):
    # Replay is only reproducible if a recorded rollout is (AGENTS.md rule 5).
    agent = FakeAgent()
    first = agent.propose(context(tmp_path, seed=7))
    assert agent.propose(context(tmp_path, seed=7)) == first
    assert FakeAgent().propose(context(tmp_path, seed=7)) == first


@pytest.mark.parametrize("field", sorted(VARIANTS))
def test_the_artifact_depends_on_every_part_of_the_context(tmp_path, field):
    # Catches both a stub that ignores its context entirely and one that drops a
    # single field — an agent blind to the seed, or to the parent it resumed
    # from, would hand every worker in a batch the same attempt.
    agent = FakeAgent()
    contents = {agent.propose(context(tmp_path, **VARIANTS[field](i))).content for i in range(12)}
    assert len(contents) > 1, f"{field} does not reach the artifact"


def test_a_context_must_name_the_parent_it_resumes_from(tmp_path):
    # Dream-RSI §3: every attempt begins at exactly one primary parent, whose
    # saved workspace and accumulated observations it inherits.
    with pytest.raises(ValueError, match="history"):
        AgentContext(problem="p", workspace=tmp_path, history=())


def test_fake_agent_rejects_an_empty_script():
    with pytest.raises(ValueError, match="script"):
        FakeAgent(script=())


def test_scripted_artifacts_are_evaluable(tmp_path):
    # The adapters have to meet: what the agent returns as ``content`` is what
    # ``TaskEvaluator.evaluate`` is handed and what a node records.
    evaluator = ToyEvaluator()
    for content in FakeAgent().script:
        result = evaluator.evaluate(content, tmp_path)
        assert result.evaluated is True
        assert result.correct is True


def run_round(agent: CodingAgent, ctx: AgentContext) -> str:
    """Stand-in for the orchestrator's call site (issue #4), typed to the protocol."""
    return agent.propose(ctx).content


def test_any_conforming_adapter_drives_a_round(tmp_path):
    class HandWrittenAgent:
        """Conforms structurally: no base class, no registration, no import of ours."""

        def propose(self, context: AgentContext) -> Artifact:
            return Artifact(content=f"# {context.problem}\ndef solve():\n    return 1\n")

    ctx = context(tmp_path)
    assert run_round(FakeAgent(), ctx).strip()
    assert run_round(HandWrittenAgent(), ctx).endswith("def solve():\n    return 1\n")


def test_src_imports_only_the_standard_library_and_itself():
    # Issue #3, "Done when": nothing in src/ imports a specific model SDK. The
    # package declares no runtime dependencies, so the check is exact rather
    # than a denylist of provider names to keep up to date.
    src = Path(__file__).resolve().parents[1] / "src"
    offenders: dict[str, set[str]] = {}
    for path in sorted(src.rglob("*.py")):
        for statement in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(statement, ast.Import):
                imported = [alias.name for alias in statement.names]
            elif isinstance(statement, ast.ImportFrom):
                # A relative import (level > 0) resolves inside the package.
                imported = [statement.module or ""] if statement.level == 0 else []
            else:
                continue
            for name in imported:
                root = name.split(".")[0]
                if root and root != "dream_rsi" and root not in sys.stdlib_module_names:
                    offenders.setdefault(str(path.relative_to(src)), set()).add(root)
    assert not offenders, f"third-party imports in src/: {offenders}"
