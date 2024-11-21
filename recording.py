import sounddevice as sd
import soundfile as sf

samplerate = 44100  # Hertz
duration = 10  # seconds
filename = 'output.wav'
print("start recording")
mydata = sd.rec(int(samplerate * duration), samplerate=samplerate,
                channels=2, blocking=True)
sf.write(filename, mydata, samplerate)
print("recorded successfully")