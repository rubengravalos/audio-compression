import pickle
import librosa
import soundfile as sf


with open('/home/alumnos/alumno3/work/TFM/experimento2capas/speech.pkl','rb') as f:
    x = pickle.load(f)

#sf.write('stereo_file.wav', x[:1], 8000, subtype='PCM_24')
#librosa.output.write_wav('audios/nuevo_audio.wav', x[:1], 8000)

