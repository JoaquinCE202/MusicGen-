import os
import json
import torchaudio
import tempfile
import torch
import numpy as np
from audiocraft.models import MusicGen
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

def generate_audio(prompt):
    output = model.generate(descriptions=[prompt], progress=True, return_tokens=True)
    return output[0]

def save_audio(output, filepath):
    # Convertir el output a formato que pueda ser guardado con torchaudio
    audio_tensor = output[0]  # Obtenemos la primera salida
    if audio_tensor.ndim == 3:
        audio_tensor = audio_tensor.squeeze(0)  # Eliminar la primera dimensión si es 3D

    sample_rate = 44100  # Puedes ajustar la tasa de muestreo según tus necesidades
    
    # Guardar el audio en un archivo
    torchaudio.save(filepath, audio_tensor, sample_rate)

def get_user_feedback(audio_path):
    bpm = calculate_bpm(audio_path)
    user_feedback = f"I want it slower, I need it much lower than {bpm} BPM"
    print(user_feedback)
    return user_feedback

def process_feedback(user_feedback, previous_generation):
    processed_feedback = f"{previous_generation}. {user_feedback}"
    return processed_feedback

def generate_audio_with_feedback(prompt, feedback, previous_generation, save_path):
    combined_input = f"{prompt}. {feedback}"
    print("Combined output:", combined_input)
    output = model.generate(descriptions=[combined_input], progress=True, return_tokens=True)
    audio_data = output[0].squeeze().numpy()  # Convertir tensor a array numpy
    # Si audio_data es 1D, agregar una dimensión para hacerlo 2D
    if len(audio_data.shape) == 1:
        audio_data = audio_data.reshape(1, -1)
    # Convertir audio data a tensor
    audio_tensor = torch.tensor(audio_data)
    # Guardar audio en archivo temporal
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
        torchaudio.save(tmpfile.name, audio_tensor, 32000)  # Guardar audio usando torchaudio
        tmpfile.seek(0)
        # Leer datos de audio del archivo temporal y guardarlos en la ubicación deseada
        with open(save_path, 'wb') as f:
            f.write(tmpfile.read())
    return audio_tensor

# Directorio de los archivos JSON y el directorio donde se guardarán los audios generados
metadata_dir = "/home/joaquin/MusicGen/dataset_dir/metadata"
output_dir = "/home/joaquin/MusicGen/Generations"
feedback_dir = "/home/joaquin/MusicGen/Feedbacks_exp1"

# Crear los directorios de salida si no existen
os.makedirs(output_dir, exist_ok=True)
os.makedirs(feedback_dir, exist_ok=True)

# Cargar el modelo preentrenado
model = MusicGen.get_pretrained('facebook/musicgen-small')

# Obtener y ordenar los archivos JSON en el directorio
json_files = sorted(
    [f for f in os.listdir(metadata_dir) if f.endswith(".json")],
    key=lambda x: int(''.join(filter(str.isdigit, x)))
)

# Iterar sobre los archivos JSON ordenados
for i, filename in enumerate(json_files):
    print("-------------------------------------------------------------------------")
    # Leer el archivo JSON
    with open(os.path.join(metadata_dir, filename), 'r') as f:
        data = json.load(f)
    
    # Obtener el prompt del archivo JSON
    prompt = data['prompt']
    print(prompt)
    
    # Generar el audio
    audio_output = generate_audio(prompt)
    
    # Guardar el audio generado en un archivo
    initial_audio_path = os.path.join(output_dir, f"initial_generation_{i}.wav")
    save_audio(audio_output, initial_audio_path)
    print(f"Audio {i} generado: {initial_audio_path}")

    # Obtener feedback del usuario
    user_feedback = get_user_feedback(initial_audio_path)

    # Procesar el feedback
    processed_feedback = process_feedback(user_feedback, prompt)

    # Generar audio con feedback
    feedback_audio_path = os.path.join(feedback_dir, f"feedback_generation_{i}.wav")
    final_generation = generate_audio_with_feedback(prompt, processed_feedback, audio_output, feedback_audio_path)
    
    # Mostrar información
    print(f"Initial Generation Prompt {i}: {prompt}")
    print(f"User Feedback {i}: {user_feedback}")
    print(f"Processed Feedback {i}: {processed_feedback}")
    print(f"Audio con feedback guardado en: {feedback_audio_path}")
    print("-------------------------------------------------------------------------")

