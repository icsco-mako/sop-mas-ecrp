"""Data adapter between llm_mas_ecrp problems and the mako baseline interface.

llm_mas_ecrp owns the instance set and ground truth (``data/NLP4LP.csv``). The
four mako baselines receive a problem dict whose ``sample`` field is the raw
input-parameter dict, which mako's ``auto_preprocess`` accepts as a
``named_dict``. Ground-truth comparison uses the same ``isclose(rel_tol=1e-6)``
judgment as Table 1.
"""
from __future__ import annotations

import math
from typing import Any, Dict

from llm_mas_ecrp.utils.utils import dataset_loader


def load_problem(prob_idx: int, dataset: str = "NLP4LP") -> Dict[str, Any]:
    """Load a problem via llm_mas_ecrp's dataset_loader and attach ground truth.

    ``dataset_loader`` reads ``./data/NLP4LP.csv`` relative to the repo root, so
    this must be invoked with the repo root as CWD.
    """
    raw = dataset_loader(dataset, prob_idx)
    sample = raw["sample"]
    if not sample:
        raise ValueError(f"{dataset}[{prob_idx}] has no sample (ground truth missing)")
    return {
        "description": raw["description"],
        "sample": sample,                       # list[dict] with input/output/solution
        "dataset": dataset,
        "prob_name": str(prob_idx),
        "ground_truth": sample[0]["output"][0],
    }


def to_mako_problem(problem: Dict[str, Any]) -> Dict[str, Any]:
    """Convert an llm_mas_ecrp problem into mako's expected problem dict.

    mako's ``auto_preprocess`` accepts a dict (``named_dict``), list of records,
    or list; we feed the raw input-parameter dict.
    """
    return {
        "description": problem["description"],
        "sample": problem["sample"][0].get("input", {}),
        "dataset": problem["dataset"],
        "prob_name": problem["prob_name"],
    }


def is_obj_correct(obj_value: Any, ground_truth: Any, rel_tol: float = 1e-6) -> bool:
    """Table-1 judgment: ``math.isclose`` with ``rel_tol=1e-6``."""
    if obj_value is None or ground_truth is None:
        return False
    try:
        return math.isclose(float(obj_value), float(ground_truth), rel_tol=rel_tol)
    except (TypeError, ValueError):
        return False
