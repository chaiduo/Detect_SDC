"""Streaming Prometheus labeling with strict score semantics."""

from __future__ import annotations

import json
import re
from collections import Counter
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

from .config import load_yaml
from .pipeline.jobs import load_pipeline_job


ABS_SYSTEM_PROMPT = (
    "You are a fair judge assistant tasked with providing clear, objective "
    "feedback based on specific criteria, ensuring each assessment reflects "
    "the absolute standards set for performance."
)

ABSOLUTE_PROMPT = """###Task Description:
An instruction (might include an Input inside it), a response to evaluate, a reference answer produced by the fault-free baseline, and a score rubric representing an evaluation criteria are given.
1. Write a detailed feedback that assesses the quality of the response strictly based on the given score rubric, not evaluating in general.
2. After writing a feedback, write a score that is an integer between 0 and 2. You should refer to the score rubric.
3. The output format should look as follows: "Feedback: (write a feedback for criteria) [RESULT] (an integer number between 0 and 2)"
4. Please do not generate any other opening, closing, and explanations.

###The instruction to evaluate:
{instruction}

###Response to evaluate:
{response}

###Reference Answer (fault-free baseline):
{reference_answer}

###Score Rubrics:
{rubric}

###Feedback: """

RUBRIC = """
Score 2: The response is semantically equivalent to the reference answer. It preserves the same core answer and would lead to the same decision or interpretation, even if the wording is different.
Score 1: The response has minor semantic deviations from the reference answer, but the core answer is still mostly preserved and the difference is unlikely to cause a wrong decision.
Score 0: The response has a major semantic deviation from the reference answer, changes or loses the core answer, contradicts the reference answer, or would likely cause a wrong decision.
"""


class LabelStatus(str, Enum):
    VALID = "valid"
    IDENTICAL_ANSWER = "identical_answer"
    PARSE_ERROR = "parse_error"


class Judge(Protocol):
    def evaluate_batch(
        self,
        items: Sequence[Mapping[str, Any]],
    ) -> tuple[list[int], list[str]]: ...


class PrometheusJudge:
    def __init__(
        self,
        model_path: str | Path,
        *,
        device: str,
        max_new_tokens: int = 200,
    ):
        self.model_path = str(model_path)
        self.device = device
        self.max_new_tokens = max_new_tokens
        self._model: Any = None
        self._tokenizer: Any = None

    def load(self) -> None:
        if self._model is not None:
            return
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._model = AutoModelForCausalLM.from_pretrained(self.model_path)
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self._tokenizer.padding_side = "left"
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
            self._tokenizer.pad_token_id = self._tokenizer.eos_token_id
        self._model.to(self.device)
        self._model.eval()

    def evaluate_batch(
        self,
        items: Sequence[Mapping[str, Any]],
    ) -> tuple[list[int], list[str]]:
        import torch

        self.load()
        prompts = [
            build_prompt(
                str(item.get("question", "")),
                str(item.get("pred_answer", "")),
                str(item.get("clean_answer", "")),
            )
            for item in items
        ]
        messages = [
            [
                {"role": "system", "content": ABS_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ]
            for prompt in prompts
        ]
        encoded = self._tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=4096,
            return_dict=True,
        )
        model_inputs = {
            key: value.to(self.device) for key, value in encoded.items()
        }
        with torch.inference_mode():
            generated = self._model.generate(
                **model_inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self._tokenizer.eos_token_id,
            )
        input_length = model_inputs["input_ids"].shape[1]
        decoded = self._tokenizer.batch_decode(
            generated[:, input_length:],
            skip_special_tokens=True,
        )
        return [extract_score(text) for text in decoded], decoded

    def close(self) -> None:
        self._model = None
        self._tokenizer = None
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass


def extract_score(text: str) -> int:
    matches = re.findall(
        r"\[RESULT\]\s*([0-2])\s*$",
        text.strip(),
        flags=re.IGNORECASE,
    )
    return int(matches[0]) if len(matches) == 1 else -1


def quality_score_to_significance(score: int) -> int | None:
    return 2 - score if score in (0, 1, 2) else None


def build_prompt(
    instruction: str,
    response: str,
    reference_answer: str,
) -> str:
    return ABSOLUTE_PROMPT.format(
        instruction=instruction,
        response=response,
        reference_answer=reference_answer,
        rubric=RUBRIC,
    )


def label_records(
    records: Sequence[Mapping[str, Any]],
    judge: Judge,
    *,
    batch_size: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    results = [dict(record) for record in records]
    pending_indices = []
    pending_items = []
    for index, item in enumerate(results):
        if str(item.get("pred_answer", "")) == str(
            item.get("clean_answer", "")
        ):
            item["quality_score"] = 2
            item["significance"] = 0
            item["label_status"] = LabelStatus.IDENTICAL_ANSWER.value
        else:
            pending_indices.append(index)
            pending_items.append(item)

    failures = []
    for start in range(0, len(pending_items), batch_size):
        batch = pending_items[start : start + batch_size]
        scores, feedback = judge.evaluate_batch(batch)
        if len(scores) != len(batch) or len(feedback) != len(batch):
            raise RuntimeError("Judge returned a result count that does not match batch")
        batch_indices = pending_indices[start : start + batch_size]
        for index, score, raw_feedback in zip(batch_indices, scores, feedback):
            item = results[index]
            significance = quality_score_to_significance(score)
            if significance is None:
                item["quality_score"] = None
                item["significance"] = None
                item["label_status"] = LabelStatus.PARSE_ERROR.value
                failures.append(
                    {
                        "question": item.get("question"),
                        "clean_answer": item.get("clean_answer"),
                        "pred_answer": item.get("pred_answer"),
                        "label_status": LabelStatus.PARSE_ERROR.value,
                        "prometheus_feedback": raw_feedback,
                    }
                )
            else:
                item["quality_score"] = score
                item["significance"] = significance
                item["label_status"] = LabelStatus.VALID.value
    return results, failures


def label_jsonl(
    input_path: str | Path,
    output_path: str | Path,
    judge: Judge,
    *,
    batch_size: int = 64,
    chunk_size: int | None = None,
    debug_path: str | Path | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    source = Path(input_path).resolve()
    destination = Path(output_path).resolve()
    debug = (
        Path(debug_path).resolve()
        if debug_path
        else destination.with_name(
            f"{destination.stem}_prometheus_parse_failed{destination.suffix}"
        )
    )
    if not overwrite and (destination.exists() or debug.exists()):
        raise FileExistsError(
            f"Label outputs already exist; pass overwrite=True: {destination}, {debug}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    debug.parent.mkdir(parents=True, exist_ok=True)
    output_temporary = destination.with_suffix(f"{destination.suffix}.tmp")
    debug_temporary = debug.with_suffix(f"{debug.suffix}.tmp")
    effective_chunk_size = chunk_size or batch_size * 8
    if effective_chunk_size < batch_size:
        raise ValueError("chunk_size must be at least batch_size")

    counts: Counter[str] = Counter()
    processed = 0
    try:
        with (
            source.open("r", encoding="utf-8") as input_stream,
            output_temporary.open("w", encoding="utf-8") as output_stream,
            debug_temporary.open("w", encoding="utf-8") as debug_stream,
        ):
            chunk = []
            for line_number, line in enumerate(input_stream, 1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"Invalid JSONL record at {source}:{line_number}"
                    ) from error
                if not isinstance(record, Mapping):
                    raise ValueError(f"Expected JSON object at {source}:{line_number}")
                chunk.append(record)
                if len(chunk) >= effective_chunk_size:
                    processed += _write_labeled_chunk(
                        chunk,
                        judge,
                        batch_size,
                        output_stream,
                        debug_stream,
                        counts,
                    )
                    chunk.clear()
            if chunk:
                processed += _write_labeled_chunk(
                    chunk,
                    judge,
                    batch_size,
                    output_stream,
                    debug_stream,
                    counts,
                )
        output_temporary.replace(destination)
        debug_temporary.replace(debug)
    except Exception:
        output_temporary.unlink(missing_ok=True)
        debug_temporary.unlink(missing_ok=True)
        raise

    return {
        "input": str(source),
        "output": str(destination),
        "debug_output": str(debug),
        "rows": processed,
        "status_counts": dict(sorted(counts.items())),
    }


def run_label_job(
    config_path: str | Path,
    job_name: str,
    *,
    repository_root: str | Path,
    device: str,
    batch_size: int = 64,
    input_path: str | Path | None = None,
    output_path: str | Path | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    job = load_pipeline_job(
        config_path,
        job_name,
        repository_root=repository_root,
    )
    experiment = load_yaml(config_path)
    labeling = experiment.get("labeling")
    if not isinstance(labeling, Mapping):
        raise ValueError("Experiment configuration requires labeling")
    judge = PrometheusJudge(
        str(labeling["model_path"]),
        device=device,
        max_new_tokens=int(labeling.get("max_new_tokens", 200)),
    )
    try:
        summary = label_jsonl(
            input_path or job.paths.injected_output,
            output_path or job.paths.labeled_output,
            judge,
            batch_size=batch_size,
            overwrite=overwrite,
        )
    finally:
        judge.close()
    summary["job"] = job.name
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def _write_labeled_chunk(
    chunk: Sequence[Mapping[str, Any]],
    judge: Judge,
    batch_size: int,
    output_stream: Any,
    debug_stream: Any,
    counts: Counter[str],
) -> int:
    labeled, failures = label_records(chunk, judge, batch_size=batch_size)
    for record in labeled:
        output_stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        counts[str(record["label_status"])] += 1
    for failure in failures:
        debug_stream.write(json.dumps(failure, ensure_ascii=False) + "\n")
    return len(labeled)
