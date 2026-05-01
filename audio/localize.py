import os
import slab
import numpy as np

# -------- CONFIG --------
INPUT_DIR = "input_audio"
OUTPUT_DIR = "audio"
ANGLES = [0, 45, 90, 135, 180, 225, 270, 315]
# ------------------------

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load KEMAR HRTF
hrtf = slab.HRTF.kemar()

# Extract source positions
sources = np.array(hrtf.sources.vertical_polar)  # [azimuth, elevation, distance]

def get_index(angle):
    """
    Convert desired angle (0 front, 90 left)
    into slab/KEMAR coordinate system and find closest index.
    """

    # 🔧 Fix coordinate system:
    # slab: +90 = right
    # we want: +90 = left
    angle = (-angle) % 360

    az = sources[:, 0]
    el = sources[:, 1]

    # Restrict to ear-level (important)
    mask = np.abs(el) < 5
    az = az[mask]
    idxs = np.where(mask)[0]

    # Wraparound-safe angular distance
    def ang_dist(a, b):
        return np.abs((a - b + 180) % 360 - 180)

    best = np.argmin(ang_dist(az, angle))
    return idxs[best]


# --- Process all input sounds ---
for fname in os.listdir(INPUT_DIR):
    if not fname.endswith(".wav"):
        continue

    path = os.path.join(INPUT_DIR, fname)
    sound = slab.Sound(path)

    name = os.path.splitext(fname)[0]

    print(f"\nProcessing: {name}")

    for angle in ANGLES:
        idx = get_index(angle)

        # Apply HRTF
        spatial = hrtf.apply(idx, sound)

        # Normalize loudness (important for experiments)
        spatial = spatial.level_normalize()

        out_name = f"{name}_{angle}.wav"
        out_path = os.path.join(OUTPUT_DIR, out_name)

        spatial.write(out_path)

        print(f"Saved: {out_name}")

print("\nDone.")