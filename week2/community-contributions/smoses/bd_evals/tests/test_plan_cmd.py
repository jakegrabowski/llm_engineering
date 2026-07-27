"""Tests for the plan command."""

import json
from pathlib import Path

from typer.testing import CliRunner

from rag_evals.cli import app
from rag_evals.planning.plan_cmd import plan_definition

runner = CliRunner()

_EXAMPLE_WORKSPACE = Path(__file__).resolve().parents[2] / "bd_evals_config_example"
_EXAMPLE_DEFINITION = Path("definitions/retrieval-example.yaml")
_EXAMPLE_DEFINITION_PATH = _EXAMPLE_WORKSPACE / "config" / _EXAMPLE_DEFINITION


class TestPlanCommand:
    def test_plan_help(self) -> None:
        result = runner.invoke(app, ["plan", "--help"])
        assert result.exit_code == 0

    def test_plan_retrieval_example(self) -> None:
        result = runner.invoke(
            app,
            ["--workspace", str(_EXAMPLE_WORKSPACE), "plan", str(_EXAMPLE_DEFINITION)],
        )
        assert result.exit_code == 0, result.output
        assert "Total calls" in result.output
        assert "--approve-call-count" in result.output

    def test_plan_json_format(self) -> None:
        result = runner.invoke(
            app,
            [
                "--workspace",
                str(_EXAMPLE_WORKSPACE),
                "plan",
                str(_EXAMPLE_DEFINITION),
                "--format",
                "json",
            ],
        )
        assert result.exit_code == 0
        data = json.loads(result.stdout)
        assert data["definition_type"] == "retrieval"
        assert "total_calls" in data
        assert "rows" in data
        assert len(data["rows"]) == data["total_calls"]

    def test_plan_missing_file(self) -> None:
        result = runner.invoke(
            app,
            ["--workspace", str(_EXAMPLE_WORKSPACE), "plan", "missing.yaml"],
        )
        assert result.exit_code == 3


class TestPlanDefinition:
    def test_retrieval_plan(self) -> None:
        result = plan_definition(
            _EXAMPLE_DEFINITION_PATH,
            project_root=_EXAMPLE_WORKSPACE / "config",
        )
        assert result["definition_type"] == "retrieval"
        assert result["total_calls"] > 0
        assert all("artifact_id" in row for row in result["rows"])

    def test_deterministic_ids(self) -> None:
        r1 = plan_definition(_EXAMPLE_DEFINITION_PATH, _EXAMPLE_WORKSPACE / "config")
        r2 = plan_definition(_EXAMPLE_DEFINITION_PATH, _EXAMPLE_WORKSPACE / "config")
        assert [r["artifact_id"] for r in r1["rows"]] == [r["artifact_id"] for r in r2["rows"]]

    def test_call_count_stable(self) -> None:
        r1 = plan_definition(_EXAMPLE_DEFINITION_PATH, _EXAMPLE_WORKSPACE / "config")
        r2 = plan_definition(_EXAMPLE_DEFINITION_PATH, _EXAMPLE_WORKSPACE / "config")
        assert r1["total_calls"] == r2["total_calls"]

    def test_retrieval_example_dimensions(self) -> None:
        result = plan_definition(
            _EXAMPLE_DEFINITION_PATH,
            project_root=_EXAMPLE_WORKSPACE / "config",
        )
        # 2 questions * 3 KBs * 2 modes * 1 candidate * 1 result = 12
        assert result["total_calls"] == 12
        assert result["dimensions"]["questions"] == 2
        assert result["dimensions"]["knowledge_bases"] == 3
