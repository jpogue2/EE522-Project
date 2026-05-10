import os
import random
import time
import csv
import subprocess
from datetime import datetime

import soundfile as sf
import serial
import pandas as pd
import matplotlib.pyplot as plt

# -------- CONFIG --------
AUDIO_DIR = "audio"
OUTPUT_DIR = "results"

SERIAL_PORT = "/dev/cu.usbmodem196816201"
BAUD_RATE = 115200

BASELINE_SEC = 2
POST_SEC = 2
REST_SEC = 0
SMOOTH_WINDOW = 25

AMBIG = [0, 45, 135, 180, 225, 315]
DIST = [90, 270]

PRIMING_N = 2

N_AMBIG_DIST = 6
N_AMBIG_AMBIG = 3
N_DIST_DIST = 3
# ------------------------

os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Serial ---
ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
time.sleep(3)
ser.reset_input_buffer()

def read_sample():
    try:
        line = ser.readline().decode(errors="ignore").strip()
        if line:
            return float(line)
    except Exception:
        pass
    return None

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

# --- Plotting ---
def make_plot(csv_path):
    df = pd.read_csv(csv_path)
    df["gsr_raw"] = pd.to_numeric(df["gsr_raw"], errors="coerce")
    df = df.dropna()

    df["time_cont"] = 0.0
    offset = 0.0
    phase_bounds = {}

    for phase in ["baseline", "stimulus", "post"]:
        mask = df["phase"] == phase
        if mask.any():
            t = df.loc[mask, "elapsed_sec"].values
            t = t - t[0]
            df.loc[mask, "time_cont"] = t + offset
            phase_bounds[phase] = (offset, offset + t[-1])
            offset += t[-1]

    df["smooth"] = df["gsr_raw"].rolling(SMOOTH_WINDOW, center=True).mean()
    baseline_mean = df[df["phase"] == "baseline"]["gsr_raw"].mean()
    df["norm"] = df["smooth"] - baseline_mean

    plt.figure(figsize=(10, 4))
    plt.plot(df["time_cont"], df["norm"], label="GSR")

    if "stimulus" in phase_bounds:
        stim_start, stim_end = phase_bounds["stimulus"]
        plt.axvline(stim_start, linestyle="--", linewidth=2, label="Start")
        plt.axvline(stim_end, linestyle="--", linewidth=2, label="End")
        plt.axvspan(stim_start, stim_end, alpha=0.15)

    plt.xlabel("Time (s)")
    plt.ylabel("ΔGSR")
    plt.title(os.path.basename(csv_path))
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(csv_path.replace(".csv", ".png"))
    plt.close()

# --- Recording ---
def record_stimulus(audio_file, trial_idx, label):
    name = os.path.splitext(os.path.basename(audio_file))[0]
    csv_path = os.path.join(out_dir, f"trial{trial_idx}_{label}_{name}.csv")

    ser.reset_input_buffer()

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "elapsed_sec", "gsr_raw", "phase"])

        # Baseline period
        t0 = time.time()
        while time.time() - t0 < BASELINE_SEC:
            val = read_sample()
            if val is not None:
                now = time.time()
                writer.writerow([now, now - t0, val, "baseline"])

        # Play audio with macOS native player
        duration = sf.info(audio_file).duration
        stim_start = time.time()
        proc = start_audio(audio_file)

        try:
            while time.time() - stim_start < duration:
                val = read_sample()
                if val is not None:
                    now = time.time()
                    writer.writerow([now, now - stim_start, val, "stimulus"])
        finally:
            try:
                proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                proc.terminate()
                try:
                    proc.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    proc.kill()

        # Post period
        post_start = time.time()
        while time.time() - post_start < POST_SEC:
            val = read_sample()
            if val is not None:
                now = time.time()
                writer.writerow([now, now - post_start, val, "post"])

    make_plot(csv_path)
    print(f"Saved: {csv_path}")

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
    print(f"\n=== Trial {idx} ({trial_type}) ===")

    pair = [f1, f2]
    random.shuffle(pair)
    A_file, B_file = pair

    input("Press Enter for A...")
    record_stimulus(A_file, idx, "A")

    print(f"Resting ({REST_SEC}s)...")
    time.sleep(REST_SEC)

    input("Press Enter for B...")
    record_stimulus(B_file, idx, "B")

    while True:
        choice = input("Which was scarier? (A/B): ").strip().upper()
        if choice in ["A", "B"]:
            break

    with open(responses_path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([trial_type, sound, A_file, B_file, choice])

    time.sleep(REST_SEC)

print("\nExperiment complete.")
ser.close()