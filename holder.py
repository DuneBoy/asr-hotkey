import pyaudio
import audio_recorder

print("Holder")


pya = pyaudio.PyAudio()

ad = audio_recorder.detect_audio_device(pya)
print(ad)

print("Done")
