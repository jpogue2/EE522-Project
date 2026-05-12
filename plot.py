from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ANALYSIS_DIR = Path("results") / "analysis_output"
FIG_DIR = ANALYSIS_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


# =========================================================
# LOAD
# =========================================================
def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame | None]:
    raw_path = ANALYSIS_DIR / "all_trials_raw.csv"
    clean_path = ANALYSIS_DIR / "clean_trials.csv"
    summary_path = ANALYSIS_DIR / "ambiguous_vs_discrete_summary.csv"
    gsrz_path = ANALYSIS_DIR / "clean_trials_with_gsr_z.csv"

    if not raw_path.exists():
        raise FileNotFoundError(f"Missing {raw_path}")
    if not clean_path.exists():
        raise FileNotFoundError(f"Missing {clean_path}")
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing {summary_path}")

    raw_df = pd.read_csv(raw_path)
    clean_df = pd.read_csv(clean_path)
    summary_df = pd.read_csv(summary_path)
    gsrz_df = pd.read_csv(gsrz_path) if gsrz_path.exists() else None
    return raw_df, clean_df, summary_df, gsrz_df


def save_fig(fig: plt.Figure, filename: str) -> None:
    path = FIG_DIR / filename
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return np.nan, np.nan
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    margin = z * ((p * (1 - p) + z**2 / (4 * n)) / n) ** 0.5 / denom
    return float(center - margin), float(center + margin)


def get_mixed_trials(df: pd.DataFrame) -> pd.DataFrame:
    """Return only trials where one stimulus was ambiguous and the other was discrete."""
    mix_df = df[df["pair_class"] == "AMBIG_DIST"].copy()
    mix_df = mix_df[mix_df["chosen_is_ambiguous"].notna()].copy()
    return mix_df


def map_cone(angle: int | float | None) -> str:
    if angle is None or pd.isna(angle):
        return "other"
    angle = int(angle)
    cone_groups = {
        "0/180": {0, 180},
        "45/315": {45, 315},
        "90/270": {90, 270},
        "135/225": {135, 225},
    }
    for name, vals in cone_groups.items():
        if angle in vals:
            return name
    return "other"


# =========================================================
# PLOTS: PRIMARY RESULTS
# =========================================================
def plot_main_ambiguity_bar(summary_df: pd.DataFrame) -> None:
    row = summary_df.loc[summary_df["analysis"] == "overall_mixed_clean"]
    if row.empty:
        raise RuntimeError("Could not find overall_mixed_clean row in ambiguous_vs_discrete_summary.csv")

    p = float(row.iloc[0]["p_ambiguous"])
    ci_low = float(row.iloc[0]["ci_low"])
    ci_high = float(row.iloc[0]["ci_high"])
    n = int(row.iloc[0]["n"])

    fig, ax = plt.subplots(figsize=(5.6, 4.3))
    bars = ax.bar([0], [p], width=0.6)
    ax.errorbar(
        [0],
        [p],
        yerr=[[p - ci_low], [ci_high - p]],
        fmt="none",
        ecolor="black",
        elinewidth=1.6,
        capsize=7,
        capthick=1.6,
        zorder=3,
    )
    ax.axhline(0.5, linestyle="--", linewidth=1)
    ax.set_xlim(-0.8, 0.8)
    ax.set_ylim(0, 1)
    ax.set_xticks([0])
    ax.set_xticklabels(["Ambiguous chosen"])
    ax.set_ylabel("Proportion")
    ax.set_title("Ambiguous vs. discrete scariness")
    ax.text(0, min(p + 0.04, 0.97), f"n={n}", ha="center", va="bottom", fontsize=9)
    bars[0].set_zorder(2)
    save_fig(fig, "ambiguity_main_bar.png")


# =========================================================
# PLOTS: SOUND-LEVEL ANALYSIS
# =========================================================
def plot_sound_individual(clean_df: pd.DataFrame) -> None:
    """Individual sound plot using mixed trials only, including white noise."""
    mix_df = get_mixed_trials(clean_df)
    if mix_df.empty:
        print("No mixed trials available; skipping individual sound plot.")
        return

    stats = (
        mix_df.groupby("sound")["chosen_is_ambiguous"]
        .agg(["mean", "count", "sum"])
        .rename(columns={"mean": "p", "count": "n", "sum": "k"})
        .sort_values(["p", "n"], ascending=[False, False])
    )

    ci = stats.apply(lambda r: wilson_ci(int(r["k"]), int(r["n"])), axis=1)
    stats["ci_low"] = [c[0] for c in ci]
    stats["ci_high"] = [c[1] for c in ci]

    fig, ax = plt.subplots(figsize=(10, 5.2))
    x = np.arange(len(stats))
    y = stats["p"].values
    yerr = [
        [p - lo for p, lo in zip(stats["p"], stats["ci_low"])],
        [hi - p for p, hi in zip(stats["p"], stats["ci_high"])],
    ]

    bars = ax.bar(x, y, width=0.72)
    ax.errorbar(
        x,
        y,
        yerr=yerr,
        fmt="none",
        ecolor="black",
        elinewidth=1.5,
        capsize=6,
        capthick=1.5,
        zorder=3,
    )
    ax.axhline(0.5, linestyle="--", linewidth=1)
    ax.set_ylim(0, 1)
    ax.set_xticks(x)
    ax.set_xticklabels(stats.index.tolist(), rotation=45, ha="right")
    ax.set_ylabel("P(ambiguous chosen)")
    ax.set_title("Ambiguity effect by individual sound")

    for bar, n in zip(bars, stats["n"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            min(bar.get_height() + 0.03, 0.98),
            f"n={int(n)}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    save_fig(fig, "ambiguity_by_sound_individual.png")


# =========================================================
# PLOTS: ANGLE / CONE FREQUENCY
# =========================================================
def plot_angle_frequency(clean_df: pd.DataFrame) -> None:
    mix_df = get_mixed_trials(clean_df)
    if mix_df.empty:
        print("No mixed trials available; skipping angle frequency plot.")
        return

    played_angles = pd.concat([mix_df["A_angle"], mix_df["B_angle"]], ignore_index=True).dropna().astype(int)
    chosen_angles = mix_df["chosen_angle"].dropna().astype(int)

    angle_order = sorted(set(played_angles.unique()).union(set(chosen_angles.unique())))
    played_counts = played_angles.value_counts().reindex(angle_order, fill_value=0)
    chosen_counts = chosen_angles.value_counts().reindex(angle_order, fill_value=0)

    played_freq = played_counts / played_counts.sum() if played_counts.sum() > 0 else played_counts.astype(float)
    chosen_freq = chosen_counts / chosen_counts.sum() if chosen_counts.sum() > 0 else chosen_counts.astype(float)

    x = np.arange(len(angle_order))
    width = 0.38
    height_max = max(float(played_freq.max()), float(chosen_freq.max())) if len(angle_order) else 1.0

    fig, ax = plt.subplots(figsize=(10, 5.2))
    ax.bar(x - width / 2, played_freq.values, width=width, label="Played")
    ax.bar(x + width / 2, chosen_freq.values, width=width, label="Chosen")
    ax.set_ylim(0, height_max * 1.25 if height_max > 0 else 1)
    ax.set_xlabel("Angle (degrees)")
    ax.set_ylabel("Relative frequency")
    ax.set_title("Angle frequency: played vs chosen (mixed ambiguous/discrete trials)")
    ax.set_xticks(x)
    ax.set_xticklabels([str(a) for a in angle_order])
    ax.legend()

    save_fig(fig, "angle_frequency.png")


def plot_cone_group_frequency(clean_df: pd.DataFrame) -> None:
    mix_df = get_mixed_trials(clean_df)
    if mix_df.empty:
        print("No mixed trials available; skipping cone frequency plot.")
        return

    played = pd.concat([mix_df["A_angle"], mix_df["B_angle"]], ignore_index=True).dropna().astype(int)
    chosen = mix_df["chosen_angle"].dropna().astype(int)

    played_groups = played.apply(map_cone)
    chosen_groups = chosen.apply(map_cone)

    group_order = ["0/180", "45/315", "90/270", "135/225"]
    played_counts = played_groups.value_counts().reindex(group_order, fill_value=0)
    chosen_counts = chosen_groups.value_counts().reindex(group_order, fill_value=0)

    played_freq = played_counts / played_counts.sum() if played_counts.sum() > 0 else played_counts.astype(float)
    chosen_freq = chosen_counts / chosen_counts.sum() if chosen_counts.sum() > 0 else chosen_counts.astype(float)

    x = np.arange(len(group_order))
    width = 0.38
    height_max = max(float(played_freq.max()), float(chosen_freq.max())) if len(group_order) else 1.0

    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    ax.bar(x - width / 2, played_freq.values, width=width, label="Played")
    ax.bar(x + width / 2, chosen_freq.values, width=width, label="Chosen")
    ax.set_ylim(0, height_max * 1.25 if height_max > 0 else 1)
    ax.set_xlabel("Cone group")
    ax.set_ylabel("Relative frequency")
    ax.set_title("Cone-group frequency: played vs chosen (mixed ambiguous/discrete trials)")
    ax.set_xticks(x)
    ax.set_xticklabels(group_order)
    ax.legend()

    save_fig(fig, "cone_frequency.png")


# =========================================================
# PLOTS: PARTICIPANT + GSR
# =========================================================
def plot_participant_rates(clean_df: pd.DataFrame) -> None:
    mix_df = get_mixed_trials(clean_df)
    if mix_df.empty:
        print("No mixed trials available; skipping participant plot.")
        return

    rates = mix_df.groupby("participant")["chosen_is_ambiguous"].mean().sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(10.2, 5.0))
    ax.bar(np.arange(len(rates)), rates.values)
    ax.axhline(0.5, linestyle="--", linewidth=1)
    ax.set_ylim(0, 1)
    ax.set_xticks(np.arange(len(rates)))
    ax.set_xticklabels(rates.index.tolist(), rotation=45, ha="right")
    ax.set_ylabel("P(ambiguous chosen)")
    ax.set_title("Participant-level ambiguity choice rate")

    save_fig(fig, "ambiguity_by_participant.png")


def plot_gsr_accuracy_by_condition(clean_df: pd.DataFrame) -> None:
    gsr_df = clean_df[clean_df["has_gsr"]].copy()
    gsr_df = gsr_df[gsr_df["match"].notna()].copy()
    if gsr_df.empty:
        print("No GSR rows available; skipping GSR accuracy plot.")
        return

    by_type = gsr_df.groupby("type")["match"].mean().sort_index()

    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    ax.bar(np.arange(len(by_type)), by_type.values)
    ax.axhline(0.5, linestyle="--", linewidth=1)
    ax.set_ylim(0, 1)
    ax.set_xticks(np.arange(len(by_type)))
    ax.set_xticklabels(by_type.index.tolist(), rotation=25, ha="right")
    ax.set_ylabel("Accuracy")
    ax.set_title("GSR agreement by condition")

    save_fig(fig, "gsr_accuracy_by_condition.png")


def plot_gsr_normalized_within_participant(gsrz_df: pd.DataFrame | None) -> None:
    if gsrz_df is None or gsrz_df.empty:
        print("No normalized GSR data available.")
        return

    gsr_mixed = gsrz_df[
        (gsrz_df["has_gsr"]) &
        (gsrz_df["pair_class"] == "AMBIG_DIST") &
        (gsrz_df["ambig_minus_discrete_z"].notna())
    ].copy()

    if gsr_mixed.empty:
        print("No mixed GSR data available.")
        return

    # --- per participant stats ---
    grouped = gsr_mixed.groupby("participant")["ambig_minus_discrete_z"]

    means = grouped.mean()
    sems = grouped.sem()  # standard error

    overall_mean = means.mean()
    overall_sem = means.sem()

    x = np.arange(len(means))

    fig, ax = plt.subplots(figsize=(9, 5))

    # bars
    bars = ax.bar(x, means.values)

    # error bars (participant SEM)
    ax.errorbar(
        x,
        means.values,
        yerr=sems.values,
        fmt="none",
        ecolor="black",
        elinewidth=1.5,
        capsize=5,
        zorder=3
    )

    # reference lines
    ax.axhline(0, linestyle="--", linewidth=1, label="No difference")
    ax.axhline(overall_mean, linestyle=":", linewidth=1.5, label="Group mean")

    ax.set_xticks(x)

    labels = [name.split("_")[-1] for name in means.index.tolist()]
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Ambiguous − discrete (z-score)")
    ax.set_xlabel("Participant ID")
    ax.set_title("Within-participant GSR difference")

    ax.legend()

    save_fig(fig, "gsr_normalized_ambig_minus_discrete.png")
    
def plot_gsr_group_summary(gsrz_df: pd.DataFrame | None) -> None:
    if gsrz_df is None:
        return

    gsr_mixed = gsrz_df[
        (gsrz_df["has_gsr"]) &
        (gsrz_df["pair_class"] == "AMBIG_DIST") &
        (gsrz_df["ambig_minus_discrete_z"].notna())
    ].copy()

    if gsr_mixed.empty:
        return

    participant_means = gsr_mixed.groupby("participant")["ambig_minus_discrete_z"].mean()

    mean = participant_means.mean()
    sem = participant_means.sem()

    fig, ax = plt.subplots(figsize=(4, 5))

    ax.bar([0], [mean])
    ax.errorbar(
        [0],
        [mean],
        yerr=[sem],
        fmt="none",
        ecolor="black",
        capsize=6
    )

    ax.axhline(0, linestyle="--")
    ax.set_ylim(min(-0.1, mean - sem * 2), max(0.4, mean + sem * 2))
    ax.set_xticks([0])
    ax.set_xticklabels(["GSR difference"])
    ax.set_ylabel("Ambiguous − discrete (z)")
    ax.set_title("GSR effect (group level)")

    save_fig(fig, "gsr_group_summary.png")


# =========================================================
# MAIN
# =========================================================
def main() -> None:
    raw_df, clean_df, summary_df, gsrz_df = load_data()

    plot_main_ambiguity_bar(summary_df)
    plot_sound_individual(clean_df)
    plot_angle_frequency(clean_df)
    plot_cone_group_frequency(clean_df)
    plot_participant_rates(clean_df)
    plot_gsr_accuracy_by_condition(clean_df)
    plot_gsr_normalized_within_participant(gsrz_df)
    plot_gsr_group_summary(gsrz_df)

    print("Done.")


if __name__ == "__main__":
    main()
