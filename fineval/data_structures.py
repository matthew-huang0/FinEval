"""
FinEval data structures.

These are the Pydantic v2 models that define the contract between datasets,
runners, scorers, and reports. Use them throughout your codebase — they're
the schema you serialize to JSON when saving intermediate results.
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Task taxonomy
# ---------------------------------------------------------------------------

class TaskType(str, Enum):
    """High-level eval task shape. Drives which scorer is used."""
    NUMERICAL_QA = "numerical_qa"      # FinQA, TAT-QA: expects a numeric answer
    MULTITURN_QA = "multiturn_qa"       # ConvFinQA: multi-turn dialogue
    CLASSIFICATION = "classification"   # FinPB: pick a label from a fixed set
    EXTRACTION = "extraction"           # SEC: pull structured fields from text
    GENERATION = "generation"           # Open-ended; scored with similarity or LLM-as-judge


# ---------------------------------------------------------------------------
# Datasets and items
# ---------------------------------------------------------------------------

class EvalItem(BaseModel):
    """One question / one input / one expected answer."""
    id: str
    task_type: TaskType
    input: dict[str, Any] = Field(
        description="Task-specific input. For numerical_qa: {'context': str, 'question': str}. "
                    "For classification: {'text': str}. For extraction: {'document': str}. "
                    "For multiturn_qa: {'context': str, 'turns': [{'q': str}, ...]}"
    )
    expected_output: Any = Field(
        description="Ground truth answer. Type depends on task_type."
    )
    metadata: dict[str, Any] = Field(default_factory=dict)


class EvalDataset(BaseModel):
    """A complete dataset — name, task, items."""
    name: str
    task_type: TaskType
    description: str
    items: list[EvalItem]
    license: str
    source_url: Optional[str] = None
    citation: Optional[str] = None


# ---------------------------------------------------------------------------
# LLM configuration and responses
# ---------------------------------------------------------------------------

class LLMConfig(BaseModel):
    """How to reach a specific model via an OpenAI-compatible endpoint."""
    provider: str  # "nvidia", "openai", "anthropic", "ollama"
    model_id: str  # e.g. "meta/llama-3.3-70b-instruct"
    base_url: str  # e.g. "https://integrate.api.nvidia.com/v1"
    api_key_env: str = Field(description="Name of env var holding the API key. e.g. 'NVIDIA_API_KEY'")
    max_tokens: int = 1024
    temperature: float = 0.0  # default deterministic
    # Approximate cost per 1M tokens (input/output) for cost-per-correct-answer math
    cost_per_million_input_tokens_cents: float = 0.0
    cost_per_million_output_tokens_cents: float = 0.0


class LLMResponse(BaseModel):
    """One model response to one item."""
    raw_text: str
    parsed_value: Any = None  # what the scorer extracted (a number, a label, etc.)
    latency_ms: int
    tokens_in: int
    tokens_out: int
    cost_cents: float
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Eval results
# ---------------------------------------------------------------------------

class EvalResult(BaseModel):
    """Per-item scoring outcome."""
    item_id: str
    dataset_name: str
    model_id: str
    score: float  # 0.0 to 1.0
    is_correct: bool  # boolean view for accuracy aggregation
    response: LLMResponse
    expected: Any
    timestamp: datetime


class LeaderboardEntry(BaseModel):
    """Aggregate row per (model, dataset)."""
    model_id: str
    dataset_name: str
    task_type: TaskType
    n_items: int
    accuracy: float  # mean is_correct
    mean_score: float  # mean of score field (may differ from accuracy if score isn't binary)
    p50_latency_ms: int
    p95_latency_ms: int
    total_cost_cents: float
    cost_per_correct_cents: float  # total_cost / n_correct
    timestamp: datetime


class Leaderboard(BaseModel):
    """Final leaderboard rolling up across (model × dataset) combinations."""
    generated_at: datetime
    entries: list[LeaderboardEntry]

    def to_markdown(self) -> str:
        """Render as a Markdown table. You'll implement this."""
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Baselines (classical ML)
# ---------------------------------------------------------------------------

class BaselineConfig(BaseModel):
    """Configuration for a classical-ML baseline."""
    name: str  # e.g. "tfidf-logistic", "xgboost-numerical"
    task_type: TaskType
    description: str
    hyperparameters: dict[str, Any] = Field(default_factory=dict)


class BaselineResult(BaseModel):
    """Per-item result from a classical baseline, parallel to EvalResult."""
    item_id: str
    dataset_name: str
    baseline_name: str
    score: float
    is_correct: bool
    predicted: Any
    expected: Any
    timestamp: datetime


# ---------------------------------------------------------------------------
# Scorer protocol (for documentation; implement in scorers/)
# ---------------------------------------------------------------------------

class Scorer:
    """
    Abstract scorer. Concrete scorers live in `fineval.scorers`.

    Implementations:
      - NumericalScorer (handles $1.5M == 1500000 == 1.5e6 equivalence)
      - ClassificationScorer (exact match on label, case-insensitive)
      - ExtractionScorer (field-by-field exact match with normalization)
      - MultiTurnScorer (per-turn scoring + final aggregate)
      - LLMJudgeScorer (uses a strong LLM to grade open-ended responses)
    """

    def score(self, predicted: Any, expected: Any) -> tuple[float, bool]:
        """Return (score in [0,1], is_correct boolean)."""
        raise NotImplementedError
