"""Lineage join model and CSV/Markdown report renderers (P10).

Joins question -> retrieval -> retrieval_judgment -> answer -> answer_judgment
by IDs and verified hashes. Produces deterministic CSV and Markdown reports.
"""

from __future__ import annotations

import csv
import io
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from rag_evals.artifacts.io import atomic_write_text, read_json
from rag_evals.artifacts.store import DatasetPath
from rag_evals.errors import ArtifactError
from rag_evals.logging import get_logger
from rag_evals.planning.identity import canonical_hash

logger = get_logger("rag_evals.reporting")


@dataclass
class JoinedRow:
    """A single joined row in the lineage report."""

    question_id: str
    question_text: str = ""
    # Retrieval
    retrieval_artifact_id: str = ""
    kb_catalog_id: str = ""
    kb_actual_id: str = ""
    retrieval_mode: str = ""
    candidate_count: int | None = None
    result_count: int | None = None
    estimated_context_tokens: int | None = None
    retrieval_cost: dict[str, Any] = field(default_factory=dict)
    retrieval_fallback: bool = False
    # Retrieval judgment
    retrieval_judgment_text: str = ""
    retrieval_judge_model: str = ""
    retrieval_judgment_artifact_id: str = ""
    retrieval_judge_cost: dict[str, Any] = field(default_factory=dict)
    # Answer
    answer_artifact_id: str = ""
    answer_text: str = ""
    answer_model: str = ""
    answer_input_tokens: int | None = None
    answer_output_tokens: int | None = None
    answer_total_tokens: int | None = None
    answer_latency_ms: int | None = None
    answer_cost: dict[str, Any] = field(default_factory=dict)
    answer_prompt_variant: str = ""
    answer_inference_variant: str = ""
    # Answer judgment
    answer_judgment_text: str = ""
    answer_judge_model: str = ""
    answer_judgment_artifact_id: str = ""
    answer_judge_cost: dict[str, Any] = field(default_factory=dict)
    # Status
    status: str = "complete"


def load_dataset_artifacts(paths: DatasetPath) -> list[dict[str, Any]]:
    """Load all artifacts from a dataset."""
    artifacts = []
    for f in sorted(paths.artifacts_dir.glob("*.json")):
        try:
            artifacts.append(read_json(f))
        except Exception as exc:
            logger.warning("Failed to read %s: %s", f, exc)
    return artifacts


def join_lineage(
    project_root: Path,
    retrieval_dataset_id: str | None = None,
    retrieval_judgment_dataset_id: str | None = None,
    answer_dataset_id: str | None = None,
    answer_judgment_dataset_id: str | None = None,
) -> list[JoinedRow]:
    """Join artifacts across datasets by question_id and verified hashes."""
    rows: list[JoinedRow] = []
    retrieval_sources: dict[str, dict[str, Any]] = {}
    answer_sources: dict[str, dict[str, Any]] = {}

    # Load retrieval artifacts
    if retrieval_dataset_id:
        paths = DatasetPath(project_root, "retrieval", retrieval_dataset_id)
        if paths.base.exists():
            for art in load_dataset_artifacts(paths):
                artifact_id = art.get("artifact_id", "")
                if not artifact_id:
                    continue
                retrieval_sources[artifact_id] = art
                rows.append(
                    JoinedRow(
                        question_id=art.get("question_id", ""),
                        question_text=art.get("question_text", ""),
                        retrieval_artifact_id=artifact_id,
                        kb_catalog_id=art.get("kb_catalog_id", ""),
                        kb_actual_id=art.get("kb_actual_id", ""),
                        retrieval_mode=art.get("mode", ""),
                        candidate_count=art.get("candidate_count"),
                        result_count=art.get("result_count"),
                        estimated_context_tokens=art.get("estimated_context_tokens"),
                        retrieval_cost=art.get("cost", {}),
                        retrieval_fallback=art.get("objective_metrics", {}).get(
                            "has_fallback", False
                        ),
                        status="retrieval_only",
                    )
                )

    # Load retrieval judgments
    if retrieval_judgment_dataset_id:
        paths = DatasetPath(project_root, "retrieval_judgments", retrieval_judgment_dataset_id)
        if paths.base.exists():
            judgments = _by_source(load_dataset_artifacts(paths), retrieval_sources)
            expanded: list[JoinedRow] = []
            for row in rows:
                matches = judgments.get(row.retrieval_artifact_id, [])
                expanded.extend(
                    [
                        replace(
                            row,
                            retrieval_judgment_artifact_id=art.get("artifact_id", ""),
                            retrieval_judgment_text=art.get("raw_judge_text", ""),
                            retrieval_judge_model=art.get("model_actual_id", ""),
                            retrieval_judge_cost=art.get("cost", {}),
                        )
                        for art in matches
                    ]
                    or [row]
                )
            rows = expanded

    # Load answers
    if answer_dataset_id:
        paths = DatasetPath(project_root, "answers", answer_dataset_id)
        if paths.base.exists():
            answer_artifacts = load_dataset_artifacts(paths)
            answers = _by_source(answer_artifacts, retrieval_sources)
            answer_sources = {
                artifact["artifact_id"]: artifact
                for artifact in answer_artifacts
                if artifact.get("artifact_id")
            }
            expanded = []
            for row in rows:
                matches = answers.get(row.retrieval_artifact_id, [])
                for art in matches:
                    usage = art.get("usage", {})
                    metrics = art.get("metrics", {})
                    expanded.append(
                        replace(
                            row,
                            answer_artifact_id=art.get("artifact_id", ""),
                            answer_text=art.get("answer_text", ""),
                            answer_model=art.get("model_actual_id", ""),
                            answer_prompt_variant=art.get("prompt_variant_id", ""),
                            answer_inference_variant=art.get("inference_variant_id", ""),
                            answer_input_tokens=usage.get("inputTokens"),
                            answer_output_tokens=usage.get("outputTokens"),
                            answer_total_tokens=usage.get("totalTokens"),
                            answer_latency_ms=metrics.get("latencyMs"),
                            answer_cost=art.get("cost", {}),
                            status="answer_missing_judgment",
                        )
                    )
                if not matches:
                    expanded.append(row)
            rows = expanded

    # Load answer judgments
    if answer_judgment_dataset_id:
        paths = DatasetPath(project_root, "answer_judgments", answer_judgment_dataset_id)
        if paths.base.exists():
            judgments = _by_source(load_dataset_artifacts(paths), answer_sources)
            expanded = []
            for row in rows:
                matches = judgments.get(row.answer_artifact_id, [])
                expanded.extend(
                    [
                        replace(
                            row,
                            answer_judgment_artifact_id=art.get("artifact_id", ""),
                            answer_judgment_text=art.get("raw_judge_text", ""),
                            answer_judge_model=art.get("model_actual_id", ""),
                            answer_judge_cost=art.get("cost", {}),
                            status="complete",
                        )
                        for art in matches
                    ]
                    or [row]
                )
            rows = expanded

    return sorted(
        rows,
        key=lambda row: (
            row.retrieval_artifact_id,
            row.retrieval_judgment_artifact_id,
            row.answer_artifact_id,
            row.answer_judgment_artifact_id,
        ),
    )


def _by_source(
    artifacts: list[dict[str, Any]], sources: dict[str, dict[str, Any]]
) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {}
    for artifact in artifacts:
        source_id = artifact.get("source_artifact_id")
        if source_id:
            source = sources.get(source_id)
            if source is None:
                raise ArtifactError(f"Report lineage source is missing: {source_id}")
            expected_hash = artifact.get("source_content_hash") or artifact.get("lineage", {}).get(
                "expected_content_hash"
            )
            if expected_hash != canonical_hash(source):
                raise ArtifactError(f"Report lineage hash mismatch for source: {source_id}")
            result.setdefault(source_id, []).append(artifact)
    return result


def render_csv(rows: list[JoinedRow]) -> str:
    """Render joined rows as deterministic CSV."""
    output = io.StringIO()
    writer = csv.writer(output)

    headers = [
        "question_id",
        "question_text",
        "kb_catalog_id",
        "retrieval_mode",
        "candidate_count",
        "result_count",
        "estimated_context_tokens",
        "retrieval_fallback",
        "retrieval_judge_model",
        "retrieval_judgment_text",
        "answer_model",
        "answer_text",
        "answer_input_tokens",
        "answer_output_tokens",
        "answer_total_tokens",
        "answer_latency_ms",
        "answer_judge_model",
        "answer_judgment_text",
        "status",
    ]
    writer.writerow(headers)

    for row in rows:
        writer.writerow(
            [
                row.question_id,
                row.question_text[:100],
                row.kb_catalog_id,
                row.retrieval_mode,
                row.candidate_count,
                row.result_count,
                row.estimated_context_tokens,
                row.retrieval_fallback,
                row.retrieval_judge_model,
                row.retrieval_judgment_text[:200],
                row.answer_model,
                row.answer_text[:200],
                row.answer_input_tokens,
                row.answer_output_tokens,
                row.answer_total_tokens,
                row.answer_latency_ms,
                row.answer_judge_model,
                row.answer_judgment_text[:200],
                row.status,
            ]
        )

    return output.getvalue()


def render_markdown(rows: list[JoinedRow]) -> str:
    """Render joined rows as deterministic Markdown."""
    lines = [
        "# Evaluation Report",
        "",
        f"Total questions: {len(rows)}",
        "",
        "| Question | KB | Mode | Est. Tokens | Answer Model | "
        "Input Tokens | Output Tokens | Retrieval Judge | Answer Judge |",
        "|---|---|---|---|---|---|---|---|---|",
    ]

    for row in rows:
        lines.append(
            f"| {_markdown_cell(row.question_id)} "
            f"| {_markdown_cell(row.kb_catalog_id)} "
            f"| {_markdown_cell(row.retrieval_mode)} "
            f"| {row.estimated_context_tokens or '-'} "
            f"| {_markdown_cell(row.answer_model or '-')} "
            f"| {row.answer_input_tokens or '-'} "
            f"| {row.answer_output_tokens or '-'} "
            f"| {_markdown_cell(row.retrieval_judge_model or '-')} "
            f"| {_markdown_cell(row.answer_judge_model or '-')} |"
        )

    lines.append("")
    lines.append("## Cost Summary")
    lines.append("")

    total_retrieval = sum(cost_dollars(r.retrieval_cost) for r in rows)
    total_answer = sum(cost_dollars(r.answer_cost) for r in rows)
    retrieval_judges = sum(cost_dollars(r.retrieval_judge_cost) for r in rows)
    answer_judges = sum(cost_dollars(r.answer_judge_cost) for r in rows)

    lines.append(f"- Retrieval cost (tested system): {total_retrieval:.4f}")
    lines.append(f"- Answer inference cost (tested system): {total_answer:.4f}")
    lines.append(f"- Total tested-system cost: {total_retrieval + total_answer:.4f}")
    lines.append(f"- Retrieval-judge cost (evaluation): {retrieval_judges:.4f}")
    lines.append(f"- Answer-judge cost (evaluation): {answer_judges:.4f}")
    lines.append("")
    lines.append("*Judge costs are evaluation infrastructure, not tested-system cost.*")

    return "\n".join(lines)


def cost_dollars(value: Any) -> float:
    """Normalize nested API cost sections without combining their categories."""
    if isinstance(value, (int, float)):
        return float(value)
    if not isinstance(value, dict):
        return 0.0
    if isinstance(value.get("dollars"), (int, float)):
        return float(value["dollars"])
    if isinstance(value.get("microdollars"), (int, float)):
        return float(value["microdollars"]) / 1_000_000
    if isinstance(value.get("total"), (int, float)):
        return float(value["total"])
    return sum(cost_dollars(item) for item in value.values())


def _markdown_cell(value: str) -> str:
    return value.replace("\\", "\\\\").replace("|", "\\|").replace("\n", " ")


def generate_report(
    project_root: Path,
    output_dir: Path,
    retrieval_dataset_id: str | None = None,
    retrieval_judgment_dataset_id: str | None = None,
    answer_dataset_id: str | None = None,
    answer_judgment_dataset_id: str | None = None,
    output_format: str = "both",
) -> dict[str, str]:
    """Generate CSV and Markdown reports. Returns dict of file paths."""
    rows = join_lineage(
        project_root,
        retrieval_dataset_id,
        retrieval_judgment_dataset_id,
        answer_dataset_id,
        answer_judgment_dataset_id,
    )

    csv_path = output_dir / "report.csv"
    md_path = output_dir / "report.md"

    result = {"rows": str(len(rows))}
    if output_format in {"both", "csv", "text", "json"}:
        atomic_write_text(csv_path, render_csv(rows))
        result["csv"] = str(csv_path)
    if output_format in {"both", "markdown", "text", "json"}:
        atomic_write_text(md_path, render_markdown(rows))
        result["markdown"] = str(md_path)
    return result
