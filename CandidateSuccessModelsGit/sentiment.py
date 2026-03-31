import pandas as pd
import numpy as np

GROUP_COLS = ["hubspot_id", "election_date"]


def add_message_level_text_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add row-level and group-level text/script features to message-level data.
    Returns the original message-level dataframe with candidate-election-level
    text aggregates merged back onto each row.
    """
    df = df.copy().reset_index(drop=True)

    # standard cleanup
    df["hubspot_id"] = df["hubspot_id"].fillna("Unknown").astype(str).str.strip()
    df["election_date"] = pd.to_datetime(df["election_date"], errors="coerce")
    df["outreach_date"] = pd.to_datetime(df["outreach_date"], errors="coerce")
    df["script"] = df["script"].fillna("").astype(str)

    # -----------------------------------
    # 1) Theme features
    # -----------------------------------
    def _get_theme_label(text: str) -> str:
        themes = {
            "trust": {"together", "protect", "safe", "support", "reliable", "community", "family", "ensure", "honest", "proven", "trust"},
            "hope": {"tomorrow", "future", "ready", "build", "opportunity", "win", "next", "change", "imagine", "better", "hope"},
            "fear": {"threat", "dangerous", "lose", "risk", "warning", "stop", "attack", "cut", "crisis", "radical", "extreme", "fear"},
            "anger": {"wrong", "betray", "unfair", "fight", "unacceptable", "demand", "shame", "failed", "corrupt", "anger"},
        }

        words = [w.strip('.,!?;:"').lower() for w in text.split()]
        if not words:
            return "neutral"

        counts = {k: sum(1 for w in words if w in vocab) for k, vocab in themes.items()}
        max_val = max(counts.values())
        if max_val == 0:
            return "neutral"
        return max(counts, key=counts.get)

    df["_theme"] = df["script"].apply(_get_theme_label)
    for t in ["trust", "hope", "fear", "anger"]:
        df[f"_is_{t}"] = (df["_theme"] == t).astype(int)

    # -----------------------------------
    # 2) Authenticity feature
    # -----------------------------------
    def _get_auth_value(text: str) -> int:
        t = text.lower()

        if any(p in t for p in ["it's ", "i'm ", "this is "]):
            if "with the " not in t and "campaign" not in t[:50]:
                return 2
            else:
                return -2

        if any(p in t for p in ["vote for", "support", "elect"]):
            return -1

        return 0

    df["_auth_score"] = df["script"].apply(_get_auth_value)

    # -----------------------------------
    # 3) Perspective features
    # -----------------------------------
    script_clean = df["script"].str.lower()
    i_pattern = r"\b(i|me|my|mine|myself)\b"
    you_pattern = r"\b(you|your|yours|yourself|yourselves)\b"

    df["_raw_i_cnt"] = script_clean.str.count(i_pattern).fillna(0)
    df["_raw_you_cnt"] = script_clean.str.count(you_pattern).fillna(0)

    # -----------------------------------
    # Aggregate text features to candidate-election grain
    # -----------------------------------
    text_agg = (
        df.groupby(GROUP_COLS, dropna=False)
          .agg(
              score_theme_trust_pct=("_is_trust", "mean"),
              score_theme_hope_pct=("_is_hope", "mean"),
              score_theme_fear_pct=("_is_fear", "mean"),
              score_theme_anger_pct=("_is_anger", "mean"),
              score_candidate_authenticity=("_auth_score", "mean"),
              score_perspective_candidate_avg=("_raw_i_cnt", "mean"),
              score_perspective_voter_avg=("_raw_you_cnt", "mean"),
          )
          .reset_index()
    )

    df = df.merge(text_agg, on=GROUP_COLS, how="left")

    drop_cols = [
        "_theme", "_is_trust", "_is_hope", "_is_fear", "_is_anger",
        "_auth_score", "_raw_i_cnt", "_raw_you_cnt"
    ]
    return df.drop(columns=drop_cols, errors="ignore")

