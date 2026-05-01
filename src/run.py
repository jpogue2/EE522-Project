import serial
import time
import csv
import os
from datetime import datetime

import sounddevice as sd
import soundfile as sf
import pandas as pd
import matplotlib.pyplot as plt

# -------- CONFIG --------
SERIAL_PORT = '/dev/cu.usbmodem196816201'
BAUD_RATE = 115200

AUDIO_FILES = [
    "audio/bear.wav",
    "audio/death_whistle.wav",
    "audio/scream.wav",
    "audio/tiger.wav",
    "audio/white_noise.wav",
    "audio/zombie.wav"
]

OUTPUT_DIR = "gsr_data"

BASELINE_SEC = 3
POST_SEC = 3
REST_SEC = 5
SMOOTH_WINDOW = 25
# ------------------------

os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Serial setup ---
ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
time.sleep(3)
ser.reset_input_buffer()

# --- Participant setup ---
participant_id = input("Enter participant ID: ").strip()
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

participant_dir = os.path.join(OUTPUT_DIR, f"{timestamp}_{participant_id}")
os.makedirs(participant_dir, exist_ok=True)

notes_path = os.path.join(participant_dir, "notes.txt")

with open(notes_path, "w") as nf:
    nf.write(f"Participant: {participant_id}\n")
    nf.write(f"Session start: {timestamp}\n")
    nf.write("Audio order:\n")
    for a in AUDIO_FILES:
        nf.write(f"{a}\n")
    nf.write("\nNotes:\n")

print(f"\nSaving all data to: {participant_dir}")


# --- Serial read helper ---
def read_sample():
    try:
        line = ser.readline().decode(errors='ignore').strip()
        if line:
            return float(line)
    except:
        pass
    return None


# --- Plotting ---
def make_plot(csv_path):
    df = pd.read_csv(csv_path)

    df["gsr_raw"] = pd.to_numeric(df["gsr_raw"], errors="coerce")
    df = df.dropna()

    # --- Build continuous timeline (NO GAPS) ---
    df["time_cont"] = 0.0
    offset = 0.0

    for phase in ["baseline", "stimulus", "post"]:
        mask = df["phase"] == phase
        if mask.any():
            t = df.loc[mask, "elapsed_sec"].values
            t = t - t[0]  # normalize phase to start at 0

            df.loc[mask, "time_cont"] = t + offset
            offset += t[-1]

    # --- Smooth signal ---
    df["gsr_smooth"] = df["gsr_raw"].rolling(SMOOTH_WINDOW, center=True).mean()

    # --- Normalize to baseline ---
    baseline_mean = df[df["phase"] == "baseline"]["gsr_raw"].mean()
    df["gsr_norm"] = df["gsr_smooth"] - baseline_mean

    # --- Plot ---
    plt.figure(figsize=(12, 5))

    for phase, color in zip(
        ["baseline", "stimulus", "post"],
        ["blue", "red", "green"]
    ):
        subset = df[df["phase"] == phase]
        plt.plot(subset["time_cont"], subset["gsr_norm"], label=phase)

    # Add vertical separators
    transitions = []
    for phase in ["baseline", "stimulus"]:
        t = df[df["phase"] == phase]["time_cont"]
        if len(t) > 0:
            transitions.append(t.max())

    for t in transitions:
        plt.axvline(t, linestyle="--", linewidth=1)

    plt.xlabel("Time (s)")
    plt.ylabel("GSR (Δ from baseline)")
    plt.title(os.path.basename(csv_path))
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    png_path = csv_path.replace(".csv", ".png")
    plt.savefig(png_path)
    plt.close()

    print(f"Saved plot: {png_path}")


# --- Trial recording ---
def record_trial(audio_file):
    base_name = os.path.splitext(os.path.basename(audio_file))[0]
    csv_path = os.path.join(participant_dir, f"{base_name}.csv")

    print(f"\n=== Trial: {base_name} ===")

    ser.reset_input_buffer()

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "elapsed_sec", "gsr_raw", "phase"])

        # --- BASELINE ---
        print(f"Baseline ({BASELINE_SEC}s)...")
        t0 = time.time()

        while time.time() - t0 < BASELINE_SEC:
            val = read_sample()
            if val is not None:
                now = time.time()
                writer.writerow([now, now - t0, val, "baseline"])

        # --- STIMULUS ---
        print(f"Playing: {audio_file}")
        data, samplerate = sf.read(audio_file)

        stim_start = time.time()
        sd.play(data, samplerate)

        duration = len(data) / samplerate

        while time.time() - stim_start < duration:
            val = read_sample()
            if val is not None:
                now = time.time()
                writer.writerow([now, now - stim_start, val, "stimulus"])

        sd.stop()

        # --- POST ---
        print(f"Post-stimulus ({POST_SEC}s)...")
        post_start = time.time()

        while time.time() - post_start < POST_SEC:
            val = read_sample()
            if val is not None:
                now = time.time()
                writer.writerow([now, now - post_start, val, "post"])

    print(f"Saved: {csv_path}")

    # --- Generate plot ---
    make_plot(csv_path)

    # --- Notes ---
    note = input("Enter notes for this trial (or press Enter to skip): ")
    if note.strip():
        with open(notes_path, "a") as nf:
            nf.write(f"{base_name}: {note}\n")


# --- Run experiment ---
for audio in AUDIO_FILES:
    input(f"\nPress Enter to start trial for {audio}...")

    record_trial(audio)

    print(f"Resting ({REST_SEC}s)...")
    time.sleep(REST_SEC)

print("\nExperiment complete.")
ser.close()