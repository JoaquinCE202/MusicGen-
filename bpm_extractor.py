import os
import librosa
from pydub import AudioSegment
from frechet_audio_distance import FrechetAudioDistance

# Inicializar FrechetAudioDistance
frechet = FrechetAudioDistance(
    model_name="vggish",
    use_pca=False,
    use_activation=False,
    verbose=False
)

def convert_to_wav(audio_path):
    audio = AudioSegment.from_file(audio_path)
    wav_path = audio_path.replace(audio_path.split('.')[-1], 'wav')
    audio.export(wav_path, format='wav')
    return wav_path

def calculate_bpm(audio_path):
    wav_path = convert_to_wav(audio_path)
    y, sr = librosa.load(wav_path, sr=None)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    return tempo

if __name__ == "__main__":
    # Directorios donde se encuentran los audios
    output_dir = "/home/joaquin/MusicGen/Generations"
    feedback_dir = "/home/joaquin/MusicGen/Feedbacks_exp1"
    
    # Calcular BPM y FAD para los audios en output_dir
    print("Calculating BPM for audios in output_dir:")
    for i, filename in enumerate(sorted(os.listdir(output_dir))):
        if filename.endswith(".wav"):
            audio_path = os.path.join(output_dir, filename)
            bpm = calculate_bpm(audio_path)
            print(f"Audio {i}:")
            print(f"BPM: {bpm}")
            print("-----------------------------")

    # Calcular BPM y FAD para los audios en feedback_dir
    print("Calculating BPM for audios in feedback_dir:")
    for i, filename in enumerate(sorted(os.listdir(feedback_dir))):
        if filename.endswith(".wav"):
            audio_path = os.path.join(feedback_dir, filename)
            bpm = calculate_bpm(audio_path)
            print(f"Audio {i}:")
            print(f"BPM: {bpm}")
            print("-----------------------------")
            
    fad_score = frechet.score('/home/joaquin/MusicGen/dataset_dir/audios', output_dir, dtype="float32")
    print("FAD initial generation: ",fad_score)
    
    fad_score = frechet.score('/home/joaquin/MusicGen/dataset_dir/audios', feedback_dir, dtype="float32")
    print("FAD feedback generation: ",fad_score)

