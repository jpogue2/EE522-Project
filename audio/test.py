import sounddevice as sd
print(sd.check_output_settings(device=2, samplerate=44100, channels=1))