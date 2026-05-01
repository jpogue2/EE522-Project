import serial
import time
import csv
import os
import simpleaudio as sa

# -------- CONFIG --------
SERIAL_PORT = '/dev/cu.usbmodem196816201'      # change this
BAUD_RATE = 115200
AUDIO_FILES = [
    "calm.wav",
    "scary.wav",
    "neutral.wav"
]
OUTPUT_DIR = "gsr_data"
# ------------------------

os.makedirs(OUTPUT_DIR, exist_ok=True)

ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
time.sleep(2)  # allow Teensy to reset

BASELINE_SEC = 3

def record_during_audio(audio_file):
    print(f"\nBaseline ({BASELINE_SEC}s)...")
    start_time = time.time()

    base_name = os.path.splitext(os.path.basename(audio_file))[0]
    out_path = os.path.join(OUTPUT_DIR, f"{base_name}.csv")

    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "elapsed_sec", "gsr_raw", "phase"])

        # baseline
        while time.time() - start_time < BASELINE_SEC:
            if ser.in_waiting:
                line = ser.readline().decode(errors='ignore').strip()
                try:
                    value = float(line)
                    now = time.time()
                    writer.writerow([now, now - start_time, value, "baseline"])
                except:
                    pass

        print(f"Playing: {audio_file}")
        wave_obj = sa.WaveObject.from_wave_file(audio_file)
        play_obj = wave_obj.play()

        stim_start = time.time()

        while play_obj.is_playing():
            if ser.in_waiting:
                line = ser.readline().decode(errors='ignore').strip()
                try:
                    value = float(line)
                    now = time.time()
                    writer.writerow([now, now - stim_start, value, "stimulus"])
                except:
                    pass

    print(f"Saved: {out_path}")

# Run experiment
for audio in AUDIO_FILES:
    record_during_audio(audio)

    print("Resting...")
    time.sleep(5)  # recovery period