"""Tests for the plan command."""

import json
from pathlib import Path

from typer.testing import CliRunner

from rag_evals.cli import app
from rag_evals.planning.plan_cmd import plan_definition

runner = CliRunner()


class TestPlanCommand:
    def test_plan_help(self) -> None:
        result = runner.invoke(app, ["plan", "--help"])
        assert result.exit_code == 0

    def test_plan_retrieval_example(self) -> None:
        ws_root = Path(__file__).resolve().parent.parent
        def_path = ws_root / "eval_definitions" / "retrieval-example.yaml"
        if not def_path.exists():
            import pytest

            pytest.skip("Example definition not found")
        result = runner.invoke(app, ["plan", str(def_path)])
        assert result.exit_code == 0, result.output
        assert "Total calls" in result.output
        assert "--approve-call-count" in result.output

    def test_plan_json_format(self) -> None:
        ws_root = Path(__file__).resolve().parent.parent
        def_path = ws_root / "eval_definitions" / "retrieval-example.yaml"
        if not def_path.exists():
            import pytest

            pytest.skip("Example definition not found")
        result = runner.invoke(app, ["plan", str(def_path), "--format", "json"])
        assert result.exit_code == 0
        data = json.loads(result.stdout)
        assert data["definition_type"] == "retrieval"
        assert "total_calls" in data
        assert "rows" in data
        assert len(data["rows"]) == data["total_calls"]

    def test_plan_missing_file(self, tmp_path: Path) -> None:
        result = runner.invoke(app, ["plan", str(tmp_path / "missing.yaml")])
        assert result.exit_code == 3


class TestPlanDefinition:
    def test_retrieval_plan(self) -> None:
        ws_root = Path(__file__).resolve().parent.parent
        def_path = ws_root / "eval_definitions" / "retrieval-example.yaml"
        if not def_path.exists():
            import pytest

            pytest.skip("Example definition not found")
        result = plan_definition(def_path, project_root=ws_root)
        assert result["definition_type"] == "retrieval"
        assert result["total_calls"] > 0
        assert all("artifact_id" in row for row in result["rows"])

    def test_deterministic_ids(self) -> None:
        ws_root = Path(__file__).resolve().parent.parent
        def_path = ws_root / "eval_definitions" / "retrieval-example.yaml"
        if not def_path.exists():
            import pytest

            pytest.skip("Example definition not found")
        r1 = plan_definition(def_path, project_root=ws_root)
        r2 = plan_definition(def_path, project_root=ws_root)
        assert [r["artifact_id"] for r in r1["rows"]] == [r["artifact_id"] for r in r2["rows"]]

    def test_call_count_stable(self) -> None:
        ws_root = Path(__file__).resolve().parent.parent
        def_path = ws_root / "eval_definitions" / "retrieval-example.yaml"
        if not def_path.exists():
            import pytest

            pytest.skip("Example definition not found")
        r1 = plan_definition(def_path, project_root=ws_root)
        r2 = plan_definition(def_path, project_root=ws_root)
        assert r1["total_calls"] == r2["total_calls"]

    def test_retrieval_example_dimensions(self) -> None:
        ws_root = Path(__file__).resolve().parent.parent
        def_path = ws_root / "eval_definitions" / "retrieval-example.yaml"
        if not def_path.exists():
            import pytest

            pytest.skip("Example definition not found")
        result = plan_definition(def_path, project_root=ws_root)
        # 2 questions * 3 KBs * 2 modes * 1 candidate * 1 result = 12
        assert result["total_calls"] == 12
        assert result["dimensions"]["questions"] == 2
        assert result["dimensions"]["knowledge_bases"] == 3
