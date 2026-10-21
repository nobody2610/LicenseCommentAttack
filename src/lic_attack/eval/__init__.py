"""Evaluation: baselines, metrics, collateral damage."""
from .conflict import (
    ConflictVerdict,
    check_conflict,
    licenses_named_in_comment,
)
from .metrics import (
    NAIVE_COMMENT_TEXT,
    BaselineOutcome,
    best_comment_by_margin,
    clause_collateral,
    evaluate_comment,
    naive_comment_ids,
    random_comment_ids,
    summarize,
)

__all__ = [
    "ConflictVerdict",
    "check_conflict",
    "licenses_named_in_comment",
    "NAIVE_COMMENT_TEXT",
    "BaselineOutcome",
    "best_comment_by_margin",
    "clause_collateral",
    "evaluate_comment",
    "naive_comment_ids",
    "random_comment_ids",
    "summarize",
]
