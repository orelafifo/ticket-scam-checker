"""tests/test_v27.py - v2.7 Upgrade 1 (meaning score). Run: python -m pytest tests"""
import sys
from pathlib import Path

import joblib
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import semantic_flags as sem  # noqa: E402
from data_utils import REAL_FILES_V27, load_real  # noqa: E402

needs_minilm = pytest.mark.skipif(not sem.available(), reason="MiniLM not installed")


def test_training_files_exclude_blind5():
    assert not any("blind5" in f for f in REAL_FILES_V27)


def test_v27_training_rows_load_with_unique_groups_for_genuine():
    real = load_real(REAL_FILES_V27)
    assert len(real) == 294
    genuine = real[real.label == 0]
    assert genuine.group.is_unique


@needs_minilm
def test_oof_scores_are_probabilities():
    from meaning_model import oof_scores
    texts = ["pay by bank transfer only, dm me", "selling at face value via ticketmaster transfer"] * 10
    y = np.array([1, 0] * 10)
    s = oof_scores(texts, y, np.arange(20), C=1.0)
    assert s.shape == (20,) and ((s >= 0) & (s <= 1)).all()


@needs_minilm
def test_checker_returns_meaning_score_with_v27_model():
    bundle = joblib.load("model/scam_model.joblib")
    if "meaning" not in bundle:
        pytest.skip("run python step2_train_v27.py first")
    from checker import check_listing
    scam = check_listing("Can't go anymore, pay by bank transfer and I'll send them after, DM me",
                         platform="Instagram", run_event_check=False, bundle=bundle)
    fair = check_listing("2 tickets, block 112 row F, face value, transfer through the Ticketmaster app",
                         platform="Twickets", run_event_check=False, bundle=bundle)
    assert 0 <= scam["meaning_score"] <= 1
    assert scam["meaning_score"] > fair["meaning_score"]


def test_frozen_v26_bundle_still_works_without_meaning():
    from checker import check_listing
    v26 = joblib.load("model/scam_model_v26.joblib")
    r = check_listing("Selling 2 tickets, DM me", platform="Instagram", run_event_check=False,
                      bundle=v26)
    assert r["meaning_score"] is None
