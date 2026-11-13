"""
data_utils.py - shared helpers: turn the data files into flag tables.

flag_fn is features.all_flags (v2.6) or rules_v25_frozen.all_flags (to rebuild v2.5).
Real-world rows: unknown seller details = neutral, event check off, "how you met the
seller" unknown (the published cases don't record it).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from channels import channel_info

REAL_FILES = ["data/real_test_set.csv", "data/heldout_test_set.csv"]   # training (published cases)
# v2.7: blind tests #2-#4 have been run once, so they become training data, plus the
# 51 new UK music rows (DATA_COLLECTION_v27.md). Blind test #5 is NOT in this list.
REAL_FILES_V27 = REAL_FILES + ["data/blind2_test_set.csv", "data/blind3_test_set.csv",
                               "data/blind4_test_set.csv", "data/v27_new_real_rows.csv",
                               "data/v27_orela_rows.csv"]   # 3 suspected scams, open WhatsApp groups


def num(v, d):
    return d if pd.isna(v) or v == "" else float(v)


def tri(v):
    return None if v == -1 else bool(v)


def synth_flags(df, flag_fn, **kw):
    return pd.DataFrame([flag_fn(
        r.text, r.account_age_days, r.followers, r.price_ratio, r.has_seat_details, r.sudden_seller,
        dict(found=tri(r.event_found), details_match=tri(r.event_details_match),
             high_demand=bool(r.event_high_demand), before_on_sale=bool(r.event_before_on_sale),
             app_only=bool(r.event_app_only)),
        dict(category=None if r.ch_category == "none" else r.ch_category, lookalike=bool(r.ch_lookalike),
             new_domain=bool(r.ch_new_domain), insecure_or_short=bool(r.ch_insecure_or_short)), **kw)
        for r in df.itertuples()])


def real_flags(df, flag_fn, **kw):
    rows = []
    for r in df.itertuples():
        p, f = num(r.price, 0), num(r.face_value, 0)
        ch = channel_info(r.platform, r.link if isinstance(r.link, str) else "", check_age=False)
        rows.append(flag_fn(r.text, num(r.account_age_days, 365), num(r.followers, 200),
                            p / f if p and f else 1.0, int(num(r.has_seat_details, 0)),
                            int(num(r.sudden_seller, 0)), None, ch, **kw))
    return pd.DataFrame(rows)


def load_real(files=None):
    files = REAL_FILES if files is None else files
    real = pd.concat([pd.read_csv(f).assign(file=Path(f).stem) for f in files if Path(f).exists()],
                     ignore_index=True)
    # source-grouped CV: scams from the same article stay together; genuine rows on their own
    # (file name added so ids such as "G01" in two files never get merged)
    real["group"] = real["source"].fillna("constructed").where(real.label == 1, real["file"] + ":" + real["id"].astype(str))
    return real


def metrics(y, pred):
    y, pred = np.asarray(y), np.asarray(pred)
    tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    return dict(caught=f"{tp}/{int(y.sum())}", false_alarms=f"{fp}/{int((1 - y).sum())}",
                recall=tp / max(y.sum(), 1), precision=tp / max(tp + fp, 1),
                F1=2 * tp / max(2 * tp + fp + fn, 1), false_alarm_rate=fp / max((1 - y).sum(), 1))
