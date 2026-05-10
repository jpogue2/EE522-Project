import os
import pandas as pd
import numpy as np

RESULTS_ROOT = "results"

# --- PARAMETERS ---
MAX_DELTA = 20  # hard cap for obvious artifacts


def extract_features(csv_path):
    df = pd.read_csv(csv_path)

    df["gsr_raw"] = pd.to_numeric(df["gsr_raw"], errors="coerce")
    df = df.dropna()

    baseline = df[df["phase"] == "baseline"]["gsr_raw"].mean()
    stim_post = df[df["phase"].isin(["stimulus", "post"])]

    peak = stim_post["gsr_raw"].max()
    delta = peak - baseline

    return delta


def find_csv(results_dir, prefix):
    matches = [
        f for f in os.listdir(results_dir)
        if f.startswith(prefix) and f.endswith(".csv")
    ]

    if len(matches) != 1:
        raise RuntimeError(f"Expected 1 match for {prefix}, found {matches}")

    return matches[0]


all_results = []

# =========================================================
# 🔹 Load all participants
# =========================================================
for participant in os.listdir(RESULTS_ROOT):
    participant_dir = os.path.join(RESULTS_ROOT, participant)

    if not os.path.isdir(participant_dir):
        continue

    responses_path = os.path.join(participant_dir, "responses.csv")
    if not os.path.exists(responses_path):
        continue

    responses = pd.read_csv(responses_path)

    print(f"\nProcessing participant: {participant}")

    for i, row in responses.iterrows():
        trial_idx = i + 1

        trial_type = row["trial_type"]
        choice = row["choice"]

        A_csv = find_csv(participant_dir, f"trial{trial_idx}_A")
        B_csv = find_csv(participant_dir, f"trial{trial_idx}_B")

        A_path = os.path.join(participant_dir, A_csv)
        B_path = os.path.join(participant_dir, B_csv)

        A_delta = extract_features(A_path)
        B_delta = extract_features(B_path)

        phys_choice = "A" if A_delta > B_delta else "B"

        all_results.append({
            "participant": participant,
            "trial": trial_idx,
            "type": trial_type,
            "choice": choice,
            "phys_choice": phys_choice,
            "A_delta": A_delta,
            "B_delta": B_delta,
            "match": choice == phys_choice
        })

# =========================================================
# 🔹 Build DataFrame
# =========================================================
df = pd.DataFrame(all_results)

# =========================================================
# 🔹 Compute IQR thresholds
# =========================================================
all_deltas = pd.concat([df["A_delta"], df["B_delta"]])

Q1 = all_deltas.quantile(0.25)
Q3 = all_deltas.quantile(0.75)
IQR = Q3 - Q1

lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

print(f"\nOutlier thresholds (IQR): {lower:.2f} to {upper:.2f}")
print(f"Hard cap threshold: {MAX_DELTA}")

# =========================================================
# 🔹 Identify outliers (with logging)
# =========================================================
excluded_rows = []

for _, row in df.iterrows():
    A = row["A_delta"]
    B = row["B_delta"]

    reasons = []

    if not (lower <= A <= upper):
        reasons.append(f"A_delta={A:.2f}")
    if not (lower <= B <= upper):
        reasons.append(f"B_delta={B:.2f}")
    if A > MAX_DELTA:
        reasons.append(f"A>{MAX_DELTA}")
    if B > MAX_DELTA:
        reasons.append(f"B>{MAX_DELTA}")

    if len(reasons) > 0:
        excluded_rows.append({
            "participant": row["participant"],
            "trial": row["trial"],
            "type": row["type"],
            "reason": ", ".join(reasons)
        })

# --- Print exclusions ---
print("\n=== EXCLUDED TRIALS ===")
for r in excluded_rows:
    print(f"[{r['participant']}] Trial {r['trial']} ({r['type']}): {r['reason']}")

# =========================================================
# 🔹 Apply filter
# =========================================================
mask = (
    (df["A_delta"] >= lower) & (df["A_delta"] <= upper) &
    (df["B_delta"] >= lower) & (df["B_delta"] <= upper) &
    (df["A_delta"] <= MAX_DELTA) &
    (df["B_delta"] <= MAX_DELTA)
)

df_clean = df[mask]

print(f"\nBefore: {len(df)} trials")
print(f"After:  {len(df_clean)} trials")
print(f"Removed: {len(df) - len(df_clean)} trials")

# =========================================================
# 🔹 RESULTS
# =========================================================
print("\n=== ORIGINAL RESULTS ===")
print("Accuracy:", df["match"].mean())
print("\nBy condition:")
print(df.groupby("type")["match"].mean())

print("\n=== CLEANED RESULTS ===")
print("Accuracy:", df_clean["match"].mean())
print("\nBy condition:")
print(df_clean.groupby("type")["match"].mean())

print("\nBy participant:")
print(df_clean.groupby("participant")["match"].mean())