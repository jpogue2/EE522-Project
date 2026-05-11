import os
import random
import time
import csv
import subprocess
from datetime import datetime

# -------- CONFIG --------
AUDIO_DIR = "audio"
OUTPUT_DIR = "results"

REST_SEC = 0

AMBIG = [0, 45, 135, 180, 225, 315]
DIST = [90, 270]

PRIMING_N = 2

N_AMBIG_DIST = 6
N_AMBIG_AMBIG = 3
N_DIST_DIST = 3
# ------------------------

os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Participant ---
participant_id = input("Enter participant ID: ").strip()
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

out_dir = os.path.join(OUTPUT_DIR, f"{timestamp}_{participant_id}")
os.makedirs(out_dir, exist_ok=True)

responses_path = os.path.join(out_dir, "responses.csv")

# --- Helpers ---
def get_angle(path):
    return int(os.path.basename(path).split("_")[-1].replace(".wav", ""))

def start_audio(path):
    return subprocess.Popen(["afplay", path])

def play_audio(path):
    pass
    # proc = start_audio(path)
    # try:
    #     proc.wait()
    # finally:
    #     try:
    #         proc.wait(timeout=2)
    #     except subprocess.TimeoutExpired:
    #         proc.terminate()
    #         try:
    #             proc.wait(timeout=2)
    #         except subprocess.TimeoutExpired:
    #             proc.kill()

# --- Group files ---
groups = {}
for f in os.listdir(AUDIO_DIR):
    if f.endswith(".wav"):
        name, angle = f.replace(".wav", "").rsplit("_", 1)
        groups.setdefault(name, []).append(os.path.join(AUDIO_DIR, f))

all_files = [f for files in groups.values() for f in files]

if not all_files:
    raise ValueError(f"No .wav files found in {AUDIO_DIR}")

# --- Priming (guaranteed count) ---
priming_trials = []
angles_available = list(set(get_angle(f) for f in all_files))

while len(priming_trials) < PRIMING_N:
    angle = random.choice(angles_available)
    candidates = [f for f in all_files if get_angle(f) == angle]
    if len(candidates) >= 2:
        f1, f2 = random.sample(candidates, 2)
        priming_trials.append(("PRIMING", "mixed", f1, f2))

# --- Main trials ---
all_trials = []
for sound, files in groups.items():
    amb = [f for f in files if get_angle(f) in AMBIG]
    dist = [f for f in files if get_angle(f) in DIST]

    for a in amb:
        for d in dist:
            all_trials.append(("AMBIG_DIST", sound, a, d))

    for i in range(len(amb)):
        for j in range(i + 1, len(amb)):
            all_trials.append(("AMBIG_AMBIG", sound, amb[i], amb[j]))

    for i in range(len(dist)):
        for j in range(i + 1, len(dist)):
            all_trials.append(("DIST_DIST", sound, dist[i], dist[j]))

amb_dist = [t for t in all_trials if t[0] == "AMBIG_DIST"]
amb_amb = [t for t in all_trials if t[0] == "AMBIG_AMBIG"]
dist_dist = [t for t in all_trials if t[0] == "DIST_DIST"]

# --- Enforce exact counts ---
if len(amb_dist) < N_AMBIG_DIST or len(amb_amb) < N_AMBIG_AMBIG or len(dist_dist) < N_DIST_DIST:
    raise ValueError("Not enough stimuli to satisfy trial counts.")

main_trials = []
main_trials += random.sample(amb_dist, N_AMBIG_DIST)
main_trials += random.sample(amb_amb, N_AMBIG_AMBIG)
main_trials += random.sample(dist_dist, N_DIST_DIST)

random.shuffle(main_trials)

trials = priming_trials + main_trials

# --- Save header ---
with open(responses_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["trial_type", "sound", "file_A", "file_B", "choice"])

print(f"\nPriming: {len(priming_trials)} | Main: {len(main_trials)} | Total: {len(trials)}")

# --- Run ---
for idx, (trial_type, sound, f1, f2) in enumerate(trials, 1):
    print(f"\n=== Trial {idx} ===")

    pair = [f1, f2]
    random.shuffle(pair)
    A_file, B_file = pair

    # input("Press Enter for A...")
    play_audio(A_file)

    if REST_SEC > 0:
        print(f"Resting ({REST_SEC}s)...")
        time.sleep(REST_SEC)

    # input("Press Enter for B...")
    play_audio(B_file)

    while True:
        choice = "A"
        if choice in ["A", "B"]:
            break

    with open(responses_path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([trial_type, sound, A_file, B_file, choice])

    if REST_SEC > 0:
        time.sleep(REST_SEC)

print("\nExperiment complete.")