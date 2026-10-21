"""Attack: scoring objectives, GCG token-search, deletion minimization, clause attack."""
from .clause_attack import (
    ClauseAttackResult,
    ClauseCandidate,
    attack_clause,
)
from .gcg import AttackConfig, AttackResult, AttackTrace, run_gcg
from .minimize import MinimizeConfig, MinimizeResult, minimize_comment
from .scoring import (
    LabelScores,
    objective_from_onehot,
    score_labels_batch,
    null_calibration,
    score_labels_from_embeds,
)
from .rewrite_search import RewriteSearchConfig, RewriteSearchResult, rewrite_search
from .m3_rewrite import BagIndex, M3RewriteConfig, M3RewriteResult, m3_rewrite_search

__all__ = [
    "BagIndex",
    "M3RewriteConfig",
    "M3RewriteResult",
    "m3_rewrite_search",
    "AttackConfig",
    "AttackResult",
    "AttackTrace",
    "run_gcg",
    "MinimizeConfig",
    "MinimizeResult",
    "minimize_comment",
    "LabelScores",
    "objective_from_onehot",
    "null_calibration",
    "score_labels_from_embeds",
    "score_labels_batch",
    "attack_clause",
    "ClauseAttackResult",
    "ClauseCandidate",
    "RewriteSearchConfig",
    "RewriteSearchResult",
    "rewrite_search",
]
