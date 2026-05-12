from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import binomtest, fisher_exact, ttest_1samp

try:
    import statsmodels.formula.api as smf
except ImportError:
    smf = None


# =========================================================
# CONFIG
# =========================================================
RESULTS_ROOT = Path("results")
OUTPUT_DIR = RESULTS_ROOT / "analysis_output"

# First 10 participants have GSR; the remaining 10 do not.
GSR_PARTICIPANTS = 10

# GSR artifact rejection
IQR_MULTIPLIER = 1.5
MAX_DELTA = 20.0

# Angles treated as spatially ambiguous for the mixed-trial analysis.
AMBIGUOUS_ANGLES = {0, 45, 135, 180, 225, 315}

# Sound label mapping
WHITE_NOISE_KEYS = {"white_noise", "whitenoise", "noise", "white"}


# =========================================================
# HELPERS
# =========================================================
def normalize_choice(x) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip().upper()


def participant_has_gsr(participant_idx: int) -> bool:
    return participant_idx < GSR_PARTICIPANTS


def search_roots(participant_dir: Path) -> list[Path]:
    roots = [participant_dir]
    data_dir = participant_dir / "data"
    if data_dir.exists():
        roots.append(data_dir)
    return roots


def find_responses_csv(participant_dir: Path) -> Optional[Path]:
    direct = participant_dir / "responses.csv"
    if direct.exists():
        return direct

    matches = list(participant_dir.rglob("responses.csv"))
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        return sorted(matches, key=lambda p: len(str(p)))[0]
    return None


def find_trial_csv(participant_dir: Path, trial_idx: int, side: str) -> Optional[Path]:
    """
    Flexible lookup for trial GSR files.
    Accepts names like:
      trial1_A.csv
      trial_1_A.csv
      trial01_A.csv
      1_A.csv
      anything containing the trial number and side letter
    """
    side = side.upper()
    trial_re = re.compile(rf"(^|[^0-9])0*{trial_idx}([^0-9]|$)")

    exact_names = {
        f"trial{trial_idx}_{side}.csv",
        f"trial_{trial_idx}_{side}.csv",
        f"trial{trial_idx}{side}.csv",
        f"trial_{trial_idx}{side}.csv",
        f"{trial_idx}_{side}.csv",
        f"{trial_idx}{side}.csv",
    }
    exact_names = {name.lower() for name in exact_names}

    candidates: list[Path] = []

    for root in search_roots(participant_dir):
        if not root.exists():
            continue

        for p in root.rglob("*.csv"):
            name = p.name.lower()
            stem = p.stem.lower()

            if name in exact_names:
                return p

            if trial_re.search(stem) and side.lower() in stem:
                candidates.append(p)

    if not candidates:
        return None

    return sorted(candidates, key=lambda p: (len(str(p)), len(p.name)))[0]


def parse_angle(value) -> Optional[int]:
    """
    Extract trailing angle from values like:
      audio/bear_315.wav -> 315
      bear_90.wav        -> 90
    """
    if pd.isna(value):
        return None

    text = os.path.basename(str(value))
    m = re.search(r"_(\d+)(?:\.\w+)?$", text)
    if not m:
        return None
    return int(m.group(1))


def angle_kind(angle: Optional[int]) -> str:
    if angle is None:
        return "unknown"
    return "ambiguous" if angle in AMBIGUOUS_ANGLES else "discrete"


def inferred_pair_class(a_kind: str, b_kind: str) -> str:
    if a_kind == "unknown" or b_kind == "unknown":
        return "unknown"
    if a_kind == "ambiguous" and b_kind == "ambiguous":
        return "AMBIG_AMBIG"
    if a_kind == "discrete" and b_kind == "discrete":
        return "DIST_DIST"
    if {a_kind, b_kind} == {"ambiguous", "discrete"}:
        return "AMBIG_DIST"
    return "unknown"


def sound_group(raw_sound) -> str:
    """
    Collapse the raw 'sound' column to broad categories.
    """
    s = str(raw_sound).strip().lower()
    if any(key in s for key in WHITE_NOISE_KEYS):
        return "white_noise"
    if s in {"priming", "mixed", ""}:
        return "other"
    return "scary"


def extract_gsr_delta(csv_path: Path) -> float:
    """
    Peak(stimulus/post) - baseline mean.
    Returns NaN if the file is unusable.
    """
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return np.nan

    required = {"phase", "gsr_raw"}
    if not required.issubset(df.columns):
        return np.nan

    df = df.copy()
    df["gsr_raw"] = pd.to_numeric(df["gsr_raw"], errors="coerce")
    df = df.dropna(subset=["phase", "gsr_raw"])

    if df.empty:
        return np.nan

    baseline = df.loc[df["phase"] == "baseline", "gsr_raw"].mean()
    stim_post = df.loc[df["phase"].isin(["stimulus", "post"]), "gsr_raw"]

    if pd.isna(baseline) or stim_post.empty:
        return np.nan

    peak = stim_post.max()
    return float(peak - baseline)


def add_within_participant_gsr_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds within-participant normalized GSR metrics.

    New columns:
      A_z, B_z
      gsr_diff_z            = A_z - B_z
      ambig_minus_discrete_z = positive means ambiguous stimulus had higher GSR
      gsr_choice            = A or B, whichever had larger z-scored GSR
      gsr_match_z           = whether behavioral choice matches gsr_choice
    """
    out = df.copy()

    for col in ["A_z", "B_z", "gsr_diff_z", "ambig_minus_discrete_z"]:
        out[col] = np.nan

    out["gsr_choice"] = ""
    out["gsr_match_z"] = np.nan
    out["participant_gsr_mean"] = np.nan
    out["participant_gsr_std"] = np.nan

    mask = out["has_gsr"] & out["A_delta"].notna() & out["B_delta"].notna()

    for participant, idx in out.loc[mask].groupby("participant").groups.items():
        pooled = pd.concat(
            [out.loc[idx, "A_delta"], out.loc[idx, "B_delta"]],
            ignore_index=True,
        ).astype(float)
        mu = pooled.mean()
        sd = pooled.std(ddof=0)

        if not np.isfinite(sd) or sd == 0:
            continue

        out.loc[idx, "participant_gsr_mean"] = mu
        out.loc[idx, "participant_gsr_std"] = sd

        out.loc[idx, "A_z"] = (out.loc[idx, "A_delta"] - mu) / sd
        out.loc[idx, "B_z"] = (out.loc[idx, "B_delta"] - mu) / sd
        out.loc[idx, "gsr_diff_z"] = out.loc[idx, "A_z"] - out.loc[idx, "B_z"]
        out.loc[idx, "gsr_choice"] = np.where(out.loc[idx, "A_z"] > out.loc[idx, "B_z"], "A", "B")
        out.loc[idx, "gsr_match_z"] = (out.loc[idx, "choice"] == out.loc[idx, "gsr_choice"]).astype(int)

        out.loc[idx, "ambig_minus_discrete_z"] = np.where(
            out.loc[idx, "A_kind"] == "ambiguous",
            out.loc[idx, "A_z"] - out.loc[idx, "B_z"],
            np.where(
                out.loc[idx, "B_kind"] == "ambiguous",
                out.loc[idx, "B_z"] - out.loc[idx, "A_z"],
                np.nan,
            ),
        )

    return out


def robust_iqr_bounds(series: pd.Series, k: float = 1.5) -> tuple[float, float]:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        return np.nan, np.nan
    q1 = clean.quantile(0.25)
    q3 = clean.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - k * iqr
    upper = q3 + k * iqr
    return float(lower), float(upper)


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return np.nan, np.nan

    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    margin = z * np.sqrt((p * (1 - p) + z**2 / (4 * n)) / n) / denom
    return float(center - margin), float(center + margin)


def safe_mean(s: pd.Series) -> float:
    s = pd.to_numeric(s, errors="coerce")
    return float(s.mean()) if s.notna().any() else np.nan


def get_mixed_trials(df: pd.DataFrame) -> pd.DataFrame:
    """Trials where one stimulus is ambiguous and the other is discrete."""
    mix = df[df["pair_class"] == "AMBIG_DIST"].copy()
    mix = mix[mix["chosen_is_ambiguous"].notna()].copy()
    return mix


# =========================================================
# LOAD DATA
# =========================================================
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
all_results: list[dict] = []

participant_dirs = sorted([p for p in RESULTS_ROOT.iterdir() if p.is_dir()], key=lambda p: p.name)

for p_idx, participant_dir in enumerate(participant_dirs):
    responses_path = find_responses_csv(participant_dir)
    if responses_path is None:
        continue

    responses = pd.read_csv(responses_path)
    required_cols = {"trial_type", "sound", "file_A", "file_B", "choice"}
    if not required_cols.issubset(responses.columns):
        raise RuntimeError(
            f"{responses_path} is missing required columns. Found: {list(responses.columns)}"
        )

    has_gsr = participant_has_gsr(p_idx)
    print(f"\nProcessing participant: {participant_dir.name} | GSR={has_gsr}")

    responses = responses.copy()
    responses["choice"] = responses["choice"].map(normalize_choice)

    for row_idx, row in responses.iterrows():
        trial_num = row_idx + 1
        trial_type = str(row["trial_type"]).strip()
        raw_sound = row["sound"]
        file_a = str(row["file_A"])
        file_b = str(row["file_B"])
        choice = normalize_choice(row["choice"])

        a_angle = parse_angle(file_a)
        b_angle = parse_angle(file_b)
        a_kind = angle_kind(a_angle)
        b_kind = angle_kind(b_angle)
        pair_class = inferred_pair_class(a_kind, b_kind)

        chosen_side = choice if choice in {"A", "B"} else ""
        chosen_is_ambiguous = np.nan
        chosen_angle = np.nan
        if chosen_side == "A":
            chosen_is_ambiguous = int(a_kind == "ambiguous") if a_kind != "unknown" else np.nan
            chosen_angle = a_angle if a_angle is not None else np.nan
        elif chosen_side == "B":
            chosen_is_ambiguous = int(b_kind == "ambiguous") if b_kind != "unknown" else np.nan
            chosen_angle = b_angle if b_angle is not None else np.nan

        record = {
            "participant": participant_dir.name,
            "participant_index": p_idx,
            "has_gsr": has_gsr,
            "trial": trial_num,
            "type": trial_type,
            "sound": raw_sound,
            "sound_group": sound_group(raw_sound),
            "file_A": file_a,
            "file_B": file_b,
            "choice": choice,
            "A_angle": a_angle,
            "B_angle": b_angle,
            "A_kind": a_kind,
            "B_kind": b_kind,
            "pair_class": pair_class,
            "trial_type_matches_inferred": (np.nan if trial_type == "PRIMING" else trial_type == pair_class),
            "chosen_side": chosen_side,
            "chosen_angle": chosen_angle,
            "chosen_is_ambiguous": chosen_is_ambiguous,
            "A_delta": np.nan,
            "B_delta": np.nan,
            "phys_choice": "",
            "match": np.nan,
        }

        if has_gsr:
            a_csv = find_trial_csv(participant_dir, trial_num, "A")
            b_csv = find_trial_csv(participant_dir, trial_num, "B")

            if a_csv is not None:
                record["A_delta"] = extract_gsr_delta(a_csv)
            if b_csv is not None:
                record["B_delta"] = extract_gsr_delta(b_csv)

            if pd.notna(record["A_delta"]) and pd.notna(record["B_delta"]):
                record["phys_choice"] = "A" if record["A_delta"] > record["B_delta"] else "B"
                record["match"] = int(choice == record["phys_choice"]) if choice in {"A", "B"} else np.nan

        all_results.append(record)


df = pd.DataFrame(all_results)
if df.empty:
    raise RuntimeError("No usable data found.")

# Save the merged trial table before cleaning.
df.to_csv(OUTPUT_DIR / "all_trials_raw.csv", index=False)


# =========================================================
# GSR CLEANING
# =========================================================
df_gsr = df[df["has_gsr"] & df["A_delta"].notna() & df["B_delta"].notna()].copy()
excluded_rows: list[dict] = []

if not df_gsr.empty:
    all_deltas = pd.concat([df_gsr["A_delta"], df_gsr["B_delta"]], ignore_index=True)
    lower, upper = robust_iqr_bounds(all_deltas, k=IQR_MULTIPLIER)

    print(f"\nIQR thresholds: {lower:.2f} to {upper:.2f}")
    print(f"Hard cap threshold: {MAX_DELTA:.2f}")

    gsr_valid = (
        df["A_delta"].between(lower, upper, inclusive="both")
        & df["B_delta"].between(lower, upper, inclusive="both")
        & (df["A_delta"] <= MAX_DELTA)
        & (df["B_delta"] <= MAX_DELTA)
    )

    for _, row in df.loc[df["has_gsr"]].iterrows():
        if pd.isna(row["A_delta"]) or pd.isna(row["B_delta"]):
            continue

        reasons = []
        if not (lower <= row["A_delta"] <= upper):
            reasons.append(f"A_delta={row['A_delta']:.2f}")
        if not (lower <= row["B_delta"] <= upper):
            reasons.append(f"B_delta={row['B_delta']:.2f}")
        if row["A_delta"] > MAX_DELTA:
            reasons.append(f"A>{MAX_DELTA}")
        if row["B_delta"] > MAX_DELTA:
            reasons.append(f"B>{MAX_DELTA}")

        if reasons:
            excluded_rows.append(
                {
                    "participant": row["participant"],
                    "trial": row["trial"],
                    "type": row["type"],
                    "reason": ", ".join(reasons),
                }
            )

    print("\n=== EXCLUDED GSR TRIALS ===")
    if excluded_rows:
        for r in excluded_rows:
            print(f"[{r['participant']}] Trial {r['trial']} ({r['type']}): {r['reason']}")
    else:
        print("None")

    # Keep all non-GSR rows, filter only GSR rows by the artifact mask.
    df_clean = df.loc[~df["has_gsr"] | gsr_valid].copy()
else:
    print("\nNo GSR trials found; skipping physiological outlier filtering.")
    df_clean = df.copy()
    lower, upper = np.nan, np.nan

# Add within-participant normalized GSR metrics to both raw and cleaned tables.
df = add_within_participant_gsr_metrics(df)
df_clean = add_within_participant_gsr_metrics(df_clean)

df_clean.to_csv(OUTPUT_DIR / "clean_trials.csv", index=False)
df_clean.to_csv(OUTPUT_DIR / "clean_trials_with_gsr_z.csv", index=False)

if excluded_rows:
    pd.DataFrame(excluded_rows).to_csv(OUTPUT_DIR / "excluded_trials.csv", index=False)


# =========================================================
# BASIC SUMMARY
# =========================================================
print(f"\nBefore: {len(df)} trials")
print(f"After:  {len(df_clean)} trials")
print(f"Removed: {len(df) - len(df_clean)} trials")

print("\n=== ORIGINAL PHYSIOLOGY RESULTS (GSR PARTICIPANTS ONLY) ===")
gsr_all = df[df["has_gsr"]].copy()
if gsr_all["match"].notna().any():
    print("Overall accuracy:", safe_mean(gsr_all["match"]))
    print("\nBy condition:")
    print(gsr_all.groupby("type")["match"].mean())
    print("\nBy participant:")
    print(gsr_all.groupby("participant")["match"].mean())
else:
    print("No valid match values found.")

print("\n=== CLEANED PHYSIOLOGY RESULTS (GSR PARTICIPANTS ONLY) ===")
gsr_clean = df_clean[df_clean["has_gsr"]].copy()
if gsr_clean["match"].notna().any():
    print("Overall accuracy:", safe_mean(gsr_clean["match"]))
    print("\nBy condition:")
    print(gsr_clean.groupby("type")["match"].mean())
    print("\nBy participant:")
    print(gsr_clean.groupby("participant")["match"].mean())
else:
    print("No valid cleaned match values found.")

print("\n=== CLEANED WITHIN-PARTICIPANT GSR NORMALIZATION ===")
gsr_norm = df_clean[df_clean["has_gsr"] & df_clean["ambig_minus_discrete_z"].notna()].copy()
if not gsr_norm.empty:
    print("Overall mean A/B z-difference:", safe_mean(gsr_norm["gsr_diff_z"]))
    print("Overall mean ambiguous-minus-discrete z-difference:", safe_mean(gsr_norm["ambig_minus_discrete_z"]))
    print("\nBy condition:")
    print(gsr_norm.groupby("type")[["gsr_diff_z", "ambig_minus_discrete_z"]].mean())
    print("\nBy participant:")
    print(gsr_norm.groupby("participant")[["gsr_diff_z", "ambig_minus_discrete_z"]].mean())
else:
    print("No valid normalized GSR rows found.")
    
gsr_mixed = df_clean[
    (df_clean["has_gsr"]) &
    (df_clean["pair_class"] == "AMBIG_DIST") &
    (df_clean["ambig_minus_discrete_z"].notna())
].copy()

t_stat, p_val = ttest_1samp(
    gsr_mixed["ambig_minus_discrete_z"],
    0.0,
    alternative="greater"
)

print("\n=== GSR SIGNIFICANCE TEST ===")
print(f"n = {len(gsr_mixed)}")
print(f"mean = {gsr_mixed['ambig_minus_discrete_z'].mean():.3f}")
print(f"t = {t_stat:.3f}, p = {p_val:.6f}")


# =========================================================
# MAIN BEHAVIORAL ANALYSIS: AMBIGUOUS vs DISCRETE SCARINESS
# =========================================================
mixed_all = get_mixed_trials(df)
mixed_clean = get_mixed_trials(df_clean)

print("\n=== AMBIGUOUS vs DISCRETE ANALYSIS ===")
print(f"Mixed trials (raw):   {len(mixed_all)}")
print(f"Mixed trials (clean): {len(mixed_clean)}")

if len(mixed_clean) == 0:
    print("No mixed ambiguous/discrete trials available for analysis.")
else:
    n = int(len(mixed_clean))
    k = int(mixed_clean["chosen_is_ambiguous"].sum())
    p_hat = k / n
    ci_low, ci_high = wilson_ci(k, n)
    bt = binomtest(k, n, p=0.5, alternative="greater")

    print("\nPrimary hypothesis test:")
    print(f"P(ambiguous chosen) = {p_hat:.3f}")
    print(f"95% Wilson CI       = [{ci_low:.3f}, {ci_high:.3f}]")
    print(f"Binomial test       = k={k}, n={n}, p={bt.pvalue:.6f}")
    print(f"Effect vs chance    = {p_hat - 0.5:+.3f}")

    # Per-sound-category comparison
    print("\nBy broad sound category:")
    by_group = (
        mixed_clean.groupby("sound_group")["chosen_is_ambiguous"]
        .agg(["mean", "count", "sum"])
        .rename(columns={"mean": "p_ambiguous", "count": "n", "sum": "k"})
    )
    print(by_group)

    # Binomial test for each sound group
    group_tests = []
    for grp, sub in mixed_clean.groupby("sound_group"):
        n_g = int(sub["chosen_is_ambiguous"].notna().sum())
        if n_g == 0:
            continue
        k_g = int(sub["chosen_is_ambiguous"].sum())
        p_g = k_g / n_g
        ci_g = wilson_ci(k_g, n_g)
        test_g = binomtest(k_g, n_g, p=0.5, alternative="greater")
        group_tests.append(
            {
                "sound_group": grp,
                "n": n_g,
                "k_ambiguous": k_g,
                "p_ambiguous": p_g,
                "ci_low": ci_g[0],
                "ci_high": ci_g[1],
                "binom_p_greater": test_g.pvalue,
            }
        )

    group_tests_df = pd.DataFrame(group_tests).sort_values("sound_group") if group_tests else pd.DataFrame()
    if not group_tests_df.empty:
        print("\nBinomial tests by sound group:")
        print(group_tests_df.to_string(index=False))

    # White-noise vs scary comparison
    wn = mixed_clean[mixed_clean["sound_group"] == "white_noise"].copy()
    scary = mixed_clean[mixed_clean["sound_group"] == "scary"].copy()

    if len(wn) > 0 and len(scary) > 0:
        wn_amb = int(wn["chosen_is_ambiguous"].sum())
        wn_dis = int((wn["chosen_is_ambiguous"] == 0).sum())
        sc_amb = int(scary["chosen_is_ambiguous"].sum())
        sc_dis = int((scary["chosen_is_ambiguous"] == 0).sum())

        table = np.array([[wn_amb, wn_dis], [sc_amb, sc_dis]])
        odds_ratio, fisher_p = fisher_exact(table, alternative="two-sided")

        print("\nWhite-noise vs scary comparison:")
        print("Contingency table [[white_noise_amb, white_noise_dis], [scary_amb, scary_dis]]:")
        print(table)
        print(f"Fisher exact p = {fisher_p:.6f}")
        print(f"Odds ratio     = {odds_ratio:.3f}")

    # Participant-level analysis
    participant_rates = mixed_clean.groupby("participant")["chosen_is_ambiguous"].mean().dropna()
    if len(participant_rates) > 1:
        t_stat, t_p = ttest_1samp(participant_rates, 0.5, alternative="greater")
        print("\nParticipant-level one-sample t-test against chance:")
        print(f"Mean participant rate = {participant_rates.mean():.3f}")
        print(f"t = {t_stat:.3f}, p = {t_p:.6f}")

    # Logistic regression
    if smf is not None and len(mixed_clean["participant"].unique()) > 1:
        print("\nLogistic regression on mixed trials:")
        try:
            model = smf.logit(
                "chosen_is_ambiguous ~ C(sound_group)",
                data=mixed_clean,
            ).fit(disp=False)

            print(model.summary())

            params = model.params
            conf = model.conf_int()
            odds = np.exp(params)
            odds_ci = np.exp(conf)
            lr_table = pd.DataFrame(
                {
                    "coef": params,
                    "odds_ratio": odds,
                    "ci_low_odds": odds_ci[0],
                    "ci_high_odds": odds_ci[1],
                    "p_value": model.pvalues,
                }
            )
            print("\nOdds-ratio table:")
            print(lr_table.to_string())
            lr_table.to_csv(OUTPUT_DIR / "logistic_regression_odds_ratios.csv")
        except Exception as e:
            print(f"Logistic regression failed: {e}")
    else:
        print("\nLogistic regression skipped (statsmodels unavailable or not enough participants).")

    # Save behavioral summaries
    summary_rows = [
        {
            "analysis": "overall_mixed_clean",
            "n": n,
            "k_ambiguous": k,
            "p_ambiguous": p_hat,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "binom_p_greater": bt.pvalue,
            "effect_vs_chance": p_hat - 0.5,
        }
    ]

    if not group_tests_df.empty:
        for _, r in group_tests_df.iterrows():
            summary_rows.append(
                {
                    "analysis": f"sound_group_{r['sound_group']}",
                    "n": int(r["n"]),
                    "k_ambiguous": int(r["k_ambiguous"]),
                    "p_ambiguous": float(r["p_ambiguous"]),
                    "ci_low": float(r["ci_low"]),
                    "ci_high": float(r["ci_high"]),
                    "binom_p_greater": float(r["binom_p_greater"]),
                    "effect_vs_chance": float(r["p_ambiguous"] - 0.5),
                }
            )

    pd.DataFrame(summary_rows).to_csv(OUTPUT_DIR / "ambiguous_vs_discrete_summary.csv", index=False)


# =========================================================
# VALIDATION / SANITY CHECKS
# =========================================================
print("\n=== SANITY CHECKS ===")
spatial_rows = df[df["type"] != "PRIMING"].copy()
mismatch_count = int((spatial_rows["trial_type_matches_inferred"] == False).sum())  # noqa: E712
print(f"Trial-type mismatches vs inferred angle class: {mismatch_count}")
if mismatch_count > 0:
    mismatch_df = spatial_rows[spatial_rows["trial_type_matches_inferred"] == False].copy()  # noqa: E712
    print(
        mismatch_df[[
            "participant",
            "trial",
            "type",
            "file_A",
            "file_B",
            "A_angle",
            "B_angle",
            "pair_class",
        ]].head(20)
    )
    mismatch_df.to_csv(OUTPUT_DIR / "trial_type_mismatches.csv", index=False)

print(f"\nSaved analysis outputs to: {OUTPUT_DIR}")
