import os
import librosa
import soundfile as sf
import numpy as np

# Chemins
input_folder = "./MIR-1K/Wavfile"  
output_folder = "./Clean_Vocals_Only"

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

files = [f for f in os.listdir(input_folder) if f.endswith('.wav')]
print(f"Lancement du traitement sur {len(files)} fichiers...")

# --- CORRECTION ICI : On boucle sur 'files' directement ---
for i, audio_file in enumerate(files): 
    path = os.path.join(input_folder, audio_file)
    
    try:
        # Chargement
        y, sr = librosa.load(path, sr=None, mono=False)
        
        # Extraction Canal Droit (Voix)
        if y.ndim == 2 and y.shape[0] == 2:
            voice_only = y[1, :] 
            
            # Sauvegarde
            output_path = os.path.join(output_folder, f"vocal_{audio_file}")
            sf.write(output_path, voice_only, sr)
        else:
            print(f"Ignoré (pas stéréo) : {audio_file}")

    except Exception as e:
        print(f"Erreur sur {audio_file}: {e}")

print("Terminé ! Tout le dataset est extrait.")