import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav

DEVICE      = 4        # QCar 2 stereo mic
SAMPLERATE  = 48000
CHANNELS    = 2
DURATION    = 5        # seconds
OUTPUT_FILE = "stereo_output.wav"

print(f"Recording {DURATION} seconds... make some noise!")

audio = sd.rec(
    frames     = int(DURATION * SAMPLERATE),
    samplerate = SAMPLERATE,
    channels   = CHANNELS,
    dtype      = "float32",
    device     = DEVICE,
    blocking   = True
)

print(f"Done!")
print(f"Left  peak: {np.max(np.abs(audio[:,0])):.4f}")
print(f"Right peak: {np.max(np.abs(audio[:,1])):.4f}")

wav.write(OUTPUT_FILE, SAMPLERATE, audio)
print(f"Saved to {OUTPUT_FILE}")
