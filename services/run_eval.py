"""Generate evaluation dataset and run model evaluation locally.

This module loads the prompt dataset, generates prompts for the tutoring
model, records the model output, and writes both the raw dataset used for the
model call and a flattened evaluation report.  The implementation relies on
Pydantic models so the model provider can be swapped without touching the
business logic.
"""

from __future__ import annotations

import argparse
import ast
import csv
import sys
from pathlib import Path
from typing import Iterable, Literal, Optional, Sequence

try:  # pragma: no cover - convenience for running as a script
    if __package__ is None:  # type: ignore[attr-defined]
        current_dir = Path(__file__).resolve().parent
        sys.path.append(str(current_dir.parent))
except NameError:  # pragma: no cover - safety guard when __file__ is missing
    pass

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from data.prompt_templates import build_rag_prompt
from eval.evaluation_logger import append_example
from services.model_switcher import OutputFormat, evaluate


class PromptExample(BaseModel):
    """Single example from the prompt dataset."""

    user_response: str = Field(alias="user_response")
    key_points_text: str = Field(alias="key_points_text")
    retrieved_text: str = Field(alias="retrieved text")

    model_config = ConfigDict(populate_by_name=True)

    def key_points(self) -> list[str]:
        """Return the list of key points for the prompt."""

        raw_value = self.key_points_text.strip()
        if not raw_value:
            return []

        try:
            parsed = ast.literal_eval(raw_value)
        except (ValueError, SyntaxError):
            parsed = None

        if isinstance(parsed, (list, tuple)):
            return [str(item).strip() for item in parsed if str(item).strip()]

        cleaned = raw_value.strip("[] ")
        if not cleaned:
            return []
        return [segment.strip(" '\"") for segment in cleaned.split(",") if segment.strip()]

    def prompt(self) -> str:
        """Build the RAG prompt for the tutoring model."""

        return build_rag_prompt(
            self.user_response,
            self.retrieved_text,
            self.key_points(),
        )


class EvalSettings(BaseModel):
    """Configuration for generating and evaluating the dataset."""

    prompt_dataset_path: Path = Path("data/prompt_dataset.csv")
    eval_dataset_path: Path = Path("data/eval_dataset.csv")
    results_path: Path = Path("data/eval_results.csv")
    model_choice: Literal["Gemini", "OpenAI"] = "Gemini"
    overwrite: bool = True


class EvaluationRecord(BaseModel):
    """Container pairing a prompt with the parsed model response."""

    prompt: str
    response: OutputFormat

    def flattened(self) -> dict[str, object]:
        """Return a serialisable representation for CSV output."""

        return {
            "prompt": self.prompt,
            "score": self.response.score,
            "correct_points": " | ".join(self.response.correct_points),
            "incorrect_points": " | ".join(self.response.incorrect_points),
            "missed_points": " | ".join(self.response.missed_points),
            "generated_feedback": self.response.generated_feedback,
            "rewritten_answer": self.response.rewritten_answer,
        }


class EvaluationSummary(BaseModel):
    """Light-weight summary information about an evaluation run."""

    total_examples: int
    average_score: Optional[float] = None

    @classmethod
    def from_records(cls, records: Sequence[EvaluationRecord]) -> "EvaluationSummary":
        if not records:
            return cls(total_examples=0, average_score=None)
        score_sum = sum(record.response.score for record in records)
        return cls(total_examples=len(records), average_score=score_sum / len(records))


def load_prompt_dataset(path: Path) -> list[PromptExample]:
    """Load prompt examples from ``path``."""

    if not path.is_file():
        raise FileNotFoundError(f"Prompt dataset not found: {path}")

    examples: list[PromptExample] = []
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if not any(row.values()):
                continue
            try:
                examples.append(PromptExample.model_validate(row))
            except ValidationError as exc:  # pragma: no cover - validation guard
                raise ValueError(f"Invalid row in prompt dataset: {row}") from exc
    return examples


def generate_records(examples: Iterable[PromptExample], settings: EvalSettings) -> list[EvaluationRecord]:
    """Generate evaluation records for the provided prompt examples."""

    records: list[EvaluationRecord] = []
    for example in examples:
        prompt_text = example.prompt()
        raw_response = evaluate(prompt_text, settings.model_choice)
        try:
            response = OutputFormat.model_validate(raw_response)
        except ValidationError as exc:
            raise ValueError(
                "Model response did not match the expected schema."
            ) from exc
        append_example(prompt_text, response.model_dump(), str(settings.eval_dataset_path))
        records.append(EvaluationRecord(prompt=prompt_text, response=response))
    return records


def write_results(
    records: Sequence[EvaluationRecord], destination: Path, append: bool
) -> None:
    """Persist flattened evaluation data to ``destination``."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    append_mode = append and destination.exists()
    mode = "a" if append_mode else "w"
    write_header = not append_mode
    if append_mode and destination.stat().st_size == 0:
        write_header = True

    with destination.open(mode, encoding="utf-8", newline="") as handle:
        fieldnames = [
            "prompt",
            "score",
            "correct_points",
            "incorrect_points",
            "missed_points",
            "generated_feedback",
            "rewritten_answer",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        for record in records:
            writer.writerow(record.flattened())


def run(settings: EvalSettings) -> EvaluationSummary:
    """Generate the evaluation dataset and run the offline evaluation."""

    if settings.overwrite:
        settings.eval_dataset_path.unlink(missing_ok=True)
        settings.results_path.unlink(missing_ok=True)

    settings.eval_dataset_path.parent.mkdir(parents=True, exist_ok=True)

    examples = load_prompt_dataset(settings.prompt_dataset_path)
    records = generate_records(examples, settings)
    write_results(records, settings.results_path, append=not settings.overwrite)

    summary = EvaluationSummary.from_records(records)
    print(f"Generated {summary.total_examples} evaluation examples at {settings.eval_dataset_path}.")
    if summary.average_score is not None:
        print(f"Average score across the dataset: {summary.average_score:.2f}")
    return summary


def parse_args(argv: Optional[Sequence[str]] = None) -> EvalSettings:
    """Parse command line arguments into :class:`EvalSettings`."""

    parser = argparse.ArgumentParser(description="Generate the evaluation dataset and run the evaluation.")
    parser.add_argument("--prompt-dataset", dest="prompt_dataset_path", help="Path to the prompt dataset CSV.")
    parser.add_argument("--eval-dataset", dest="eval_dataset_path", help="Where to store the generated evaluation dataset.")
    parser.add_argument("--results", dest="results_path", help="Where to store the flattened evaluation results.")
    parser.add_argument(
        "--model",
        dest="model_choice",
        choices=["Gemini", "OpenAI"],
        help="Model to use for evaluation via the model switcher.",
    )
    parser.add_argument(
        "--no-overwrite",
        dest="overwrite",
        action="store_false",
        help="Append to existing files instead of replacing them.",
    )

    args = parser.parse_args(argv)
    data = {key: value for key, value in vars(args).items() if value is not None}
    return EvalSettings(**data)


def main(argv: Optional[Sequence[str]] = None) -> EvaluationSummary:
    settings = parse_args(argv)
    return run(settings)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
