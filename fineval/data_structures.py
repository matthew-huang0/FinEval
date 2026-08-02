"""
Core data structures for FinEval.

These Pydantic v2 models are shared by datasets, runners, scorers, and reports.
They also define the format used when saving intermediate results as JSON.
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


# Task types

class TaskType(str, Enum):
    """The main types of evaluation tasks supported by FinEval."""
    NUMERICAL_QA = "numerical_qa"      # FinQA and TAT-QA: expects a number
    MULTITURN_QA = "multiturn_qa"      # ConvFinQA: uses a multi-turn conversation
    CLASSIFICATION = "classification"   # FinPB: selects from a fixed set of labels
    EXTRACTION = "extraction"           # SEC: extracts structured information from text
    GENERATION = "generation"           # Open-ended response scored by similarity or an LLM judge


# Dataset models

class EvalItem(BaseModel):
    """A single evaluation example with an input and expected answer."""
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
    """A dataset and the evaluation items it contains."""
    name: str
    task_type: TaskType
    description: str
    items: list[EvalItem]
    license: str
    source_url: Optional[str] = None
    citation: Optional[str] = None


# LLM configuration + responses

class LLMConfig(BaseModel):
    """Settings for connecting to a model through an OpenAI-compatible API."""
    provider: str  # Examples: "nvidia", "openai", "anthropic", or "ollama"
    model_id: str  # Example: "meta/llama-3.3-70b-instruct"
    base_url: str  # Example: "https://integrate.api.nvidia.com/v1"
    api_key_env: str = Field(description="Name of env var holding the API key. e.g. 'NVIDIA_API_KEY'")
    max_tokens: int = 1024
    temperature: float = 0.0  # Use deterministic output by default
    # Estimated input and output costs per one million tokens
    cost_per_million_input_tokens_cents: float = 0.0
    cost_per_million_output_tokens_cents: float = 0.0


class LLMResponse(BaseModel):
    """A model's response to one evaluation item."""
    raw_text: str
    parsed_value: Any = None  # Value extracted by the scorer, such as a number or label
    latency_ms: int
    tokens_in: int
    tokens_out: int
    cost_cents: float
    error: Optional[str] = None


# Evaluation results

class EvalResult(BaseModel):
    """The scoring result for one evaluation item."""
    item_id: str
    dataset_name: str
    model_id: str
    score: float  # Value from 0.0 to 1.0
    is_correct: bool  # Used when calculating accuracy
    response: LLMResponse
    expected: Any
    timestamp: datetime


class LeaderboardEntry(BaseModel):
    """Summary results for one model and dataset pair."""
    model_id: str
    dataset_name: str
    task_type: TaskType
    n_items: int
    accuracy: float  # Average of the is_correct values
    mean_score: float  # May differ from accuracy when scores are not binary
    p50_latency_ms: int
    p95_latency_ms: int
    total_cost_cents: float
    cost_per_correct_cents: float  # Total cost divided by the number of correct answers
    timestamp: datetime


class Leaderboard(BaseModel):
    """The final leaderboard across all model and dataset combinations."""
    generated_at: datetime
    entries: list[LeaderboardEntry]

    def to_markdown(self) -> str:
        """Convert the leaderboard into a Markdown table."""
        raise NotImplementedError


# Classical ML baselines

class BaselineConfig(BaseModel):
    """Settings for a classical machine-learning baseline."""
    name: str  # Examples: "tfidf-logistic" or "xgboost-numerical"
    task_type: TaskType
    description: str
    hyperparameters: dict[str, Any] = Field(default_factory=dict)


class BaselineResult(BaseModel):
    """The result of running a classical baseline on one item."""
    item_id: str
    dataset_name: str
    baseline_name: str
    score: float
    is_correct: bool
    predicted: Any
    expected: Any
    timestamp: datetime


# Shared interface for scorers

class Scorer:
    """
    Base interface for scorers in `fineval.scorers`.

    Available scorer types:
      - NumericalScorer: treats values such as $1.5M, 1500000, and 1.5e6 as equivalent
      - ClassificationScorer: compares labels without considering capitalization
      - ExtractionScorer: compares normalized values one field at a time
      - MultiTurnScorer: scores each turn and calculates an overall result
      - LLMJudgeScorer: uses a stronger model to grade open-ended answers
    """

    def score(self, predicted: Any, expected: Any) -> tuple[float, bool]:
        """Return the score and whether the answer should count as correct."""
        raise NotImplementedError