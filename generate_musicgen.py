import os
import json
import torchaudio
from audiocraft.models import MusicGen
import torch

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

# Cargar el modelo preentrenado
model = MusicGen.get_pretrained('facebook/musicgen-small')

# Directorio de los archivos JSON y el directorio donde se guardarán los audios generados
metadata_dir = "/home/joaquin/MusicGen/dataset_dir/metadata"
output_dir = "/home/joaquin/MusicGen/Generations"

# Crear el directorio de salida si no existe
os.makedirs(output_dir, exist_ok=True)

# Obtener y ordenar los archivos JSON en el directorio
json_files = sorted(
    [f for f in os.listdir(metadata_dir) if f.endswith(".json")],
    key=lambda x: int(''.join(filter(str.isdigit, x)))
)

# Iterar sobre los archivos JSON ordenados
for filename in json_files:
    # Leer el archivo JSON
    with open(os.path.join(metadata_dir, filename), 'r') as f:
        data = json.load(f)
    
    # Obtener el prompt del archivo JSON
    prompt = data['prompt']
    print(prompt)
    
    # Generar el audio
    audio_output = generate_audio(prompt)
    
    # Guardar el audio generado en un archivo
    audio_filename = f"{os.path.splitext(filename)[0]}.wav"
    save_audio(audio_output, os.path.join(output_dir, audio_filename))
    
    # Imprimir mensaje de confirmación
    print(f"Audio {os.path.splitext(filename)[0]} generado")

