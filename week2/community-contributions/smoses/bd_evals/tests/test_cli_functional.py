"""Functional offline CLI coverage for remediated commands."""

from pathlib import Path

import pytest
from typer.testing import CliRunner

from rag_evals.artifacts.store import (
    Lifecycle,
    create_dataset,
    transition_lifecycle,
    write_artifact,
)
from rag_evals.cli import app
from rag_evals.planning.identity import canonical_hash

runner = CliRunner()


def _source_dataset(root: Path, dataset_type: str, dataset_id: str, character: str) -> None:
    artifact_id = f"sha256:{character * 64}"
    paths = create_dataset(
        root,
        dataset_type,
        dataset_id,
        {"source": True},
        [{"artifact_id": artifact_id}],
    )
    write_artifact(
        paths,
        artifact_id,
        {
            "question_id": "q1",
            "question_text": "Question?",
            "rubric": None,
            "prepared_context": "context",
            "answer_text": "answer",
            "content_hash": canonical_hash({"source": character}),
        },
    )
    transition_lifecycle(paths, Lifecycle.COMPLETE)


def _definition(root: Path, stage: str, source_type: str, source_id: str) -> Path:
    model_key = "answer_models" if stage == "answer" else "judge_models"
    model_selection = "answer_models" if stage == "answer" else "judge_models"
    output = {"retrieval_judgment": "rj", "answer": "ans", "answer_judgment": "aj"}[stage]
    path = root / f"{stage}.yaml"
    path.write_text(
        f"""schema_version: 1
test: {{id: test, type: {stage}}}
source_dataset: {{type: {source_type}, id: {source_id}}}
{model_key}: {{source: models.yaml}}
selection: {{{model_selection}: {{include: [model]}}}}
prompt_variants:
  - id: prompt
    template:
      system_instructions: "System {{{{ question_id }}}}"
      user_message: "{{{{ question }}}}"
inference_variants:
  - id: inference
    inference_config: {{temperature: 0.0}}
api: {{base_url_env: API_URL, api_key_env: API_KEY}}
output: {{dataset_id: {output}}}
"""
    )
    return path


@pytest.mark.parametrize(
    ("command", "stage", "source_type", "source_id"),
    [
        ("judge-retrieval", "retrieval_judgment", "retrieval", "ret"),
        ("generate-answers", "answer", "retrieval", "ret"),
        ("judge-answers", "answer_judgment", "answers", "answers-source"),
    ],
)
def test_f005_ask_commands_plan_functionally_offline(
    tmp_path: Path, command: str, stage: str, source_type: str, source_id: str
) -> None:
    (tmp_path / "pyproject.toml").write_text("")
    (tmp_path / "models.yaml").write_text(
        "schema_version: 1\nmodels:\n  - id: model\n    model_id: actual-model\n"
    )
    _source_dataset(tmp_path, "retrieval", "ret", "a")
    _source_dataset(tmp_path, "answers", "answers-source", "b")
    definition = _definition(tmp_path, stage, source_type, source_id)
    result = runner.invoke(app, [command, str(definition), "--dry-run"])
    assert result.exit_code == 0, result.output
    assert "Planned calls: 1" in result.output
    assert "No API calls made" in result.output


def test_f006_report_cli_honors_markdown_format(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text("")
    _source_dataset(tmp_path, "retrieval", "ret", "a")
    definition = tmp_path / "report.yaml"
    definition.write_text("datasets:\n  retrieval: ret\noutput_dir: reports/result\n")
    result = runner.invoke(app, ["report", str(definition), "--format", "markdown"])
    assert result.exit_code == 0, result.output
    assert (tmp_path / "reports/result/report.md").exists()
    assert not (tmp_path / "reports/result/report.csv").exists()
