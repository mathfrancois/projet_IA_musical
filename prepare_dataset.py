import os
import random
import numpy as np
import librosa
import soundfile as sf
from tqdm import tqdm
import scipy.interpolate

CONFIG = {
    'sr': 16000,          # Sample rate MIR-1K standard
    'n_fft': 1024,        # Taille fenêtre FFT
    'hop_length': 256,    # Saut entre les frames
    'n_mels': 80,         # Nombre de bandes de fréquences
    'fmin': 0,
    'fmax': 8000,
    'segment_len': 128,   # Nombre de frames temporelles par segment (env 2 sec)
                          
}

PATHS = {
    'input_wavs': "./Clean_Vocals_Only",  # Vos fichiers wav extraits
    'output_dir': "./Dataset_Ready_For_AI"
}

def get_mel(y, p):
    """Calcule le Log-Mel-Spectrogramme normalisé"""
    # 1. Mel-Spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=y, sr=p['sr'], n_fft=p['n_fft'], hop_length=p['hop_length'],
        n_mels=p['n_mels'], fmin=p['fmin'], fmax=p['fmax']
    )
    # 2. Logarithme (dB)
    log_mel = librosa.power_to_db(mel_spec, ref=np.max)
    
    # 3. Normalisation simple entre [-1, 1] environ
    # On assume que le min est vers -80dB (silence) et max à 0dB
    log_mel = (log_mel / 80.0) + 1.0 
    return log_mel.astype(np.float32)

def detune(y, sr):
    """
    Applique un désaccordage qui varie dans le temps (Wobble/Jitter).
    Simule une voix instable qui oscille autour de la note.
    Garde la synchronisation temporelle parfaite avec l'original.
    """
    # 1. Création d'une courbe de variation lente (bruit basse fréquence)
    n_samples = len(y)
    
    # On crée quelques points de contrôle aléatoires (ex: un changement toutes les 0.5s)
    num_points = int(n_samples / (sr * 0.5)) + 2 
    
    # Valeurs de pitch aléatoires entre -0.7 et +0.7 demi-tons
    random_points = np.random.uniform(-0.7, 0.7, num_points)
    
    # Interpolation pour rendre la courbe lisse (pas d'escalier)
    x_points = np.linspace(0, n_samples, num_points)
    x_full = np.arange(n_samples)
    
    # Création de la courbe de shift frame par frame
    f = scipy.interpolate.interp1d(x_points, random_points, kind='quadratic')
    pitch_curve = f(x_full) # Courbe de pitch pour chaque échantillon

    # 2. Application du pitch shift dynamique
    # Pour faire ça vite et bien en Python pur, on utilise une astuce de ré-échantillonnage
    # (Resampling) variable.
    
    # On calcule les nouveaux indices d'échantillons (Time warping léger)
    # L'astuce : index_new = index_old + integrale(pitch_curve)
    # C'est une approximation mathématique du changement de vitesse
    
    # Facteur de vitesse : 2^(semitones/12)
    speed_factors = 2 ** (pitch_curve / 12.0)
    
    # On intègre la vitesse pour trouver la position temporelle
    # C'est comme avancer sur une bande magnétique à vitesse variable
    new_indices = np.cumsum(speed_factors)
    
    # Normalisation pour garder EXACTEMENT la même durée totale (Sync X/Y)
    new_indices = new_indices / new_indices[-1] * (n_samples - 1)
    
    # 3. Interpolation finale (Reconstruction de l'audio)
    y_wobbly = np.interp(np.arange(n_samples), new_indices, y)
    
    return y_wobbly.astype(np.float32)

def process():
    os.makedirs(os.path.join(PATHS['output_dir'], 'X'), exist_ok=True)
    os.makedirs(os.path.join(PATHS['output_dir'], 'Y'), exist_ok=True)
    
    files = [f for f in os.listdir(PATHS['input_wavs']) if f.endswith('.wav')]
    print(f"Traitement de {len(files)} fichiers...")
    
    idx = 0
    for f in tqdm(files):
        path = os.path.join(PATHS['input_wavs'], f)
        
        try:
            # 1. Charger Audio Juste (Ground Truth)
            y_clean, _ = librosa.load(path, sr=CONFIG['sr'])
            
            # 2. Créer Audio Faux (Input)
            y_noisy = detune(y_clean, CONFIG['sr'])
            
            # Alignement des longueurs
            min_len = min(len(y_clean), len(y_noisy))
            y_clean = y_clean[:min_len]
            y_noisy = y_noisy[:min_len]
            
            # 3. Conversion en Mel-Spectrogrammes
            mel_clean = get_mel(y_clean, CONFIG)  # Cible (Y)
            mel_noisy = get_mel(y_noisy, CONFIG)  # Entrée (X)
            
            # 4. Découpage en segments fixes (Optionnel mais recommandé pour le batching)
            # Si segment_len est défini, on découpe l'audio long en petits morceaux
            seg_len = CONFIG['segment_len']
            if seg_len:
                num_frames = mel_clean.shape[1]
                for start in range(0, num_frames - seg_len, seg_len):
                    # Extraction du segment
                    sub_clean = mel_clean[:, start:start+seg_len]
                    sub_noisy = mel_noisy[:, start:start+seg_len]
                    
                    # Sauvegarde numpy (.npy)
                    np.save(os.path.join(PATHS['output_dir'], 'Y', f"{idx}.npy"), sub_clean)
                    np.save(os.path.join(PATHS['output_dir'], 'X', f"{idx}.npy"), sub_noisy)
                    idx += 1
            else:
                # Sauvegarde du fichier entier
                np.save(os.path.join(PATHS['output_dir'], 'Y', f"{idx}.npy"), mel_clean)
                np.save(os.path.join(PATHS['output_dir'], 'X', f"{idx}.npy"), mel_noisy)
                idx += 1
                
        except Exception as e:
            print(f"Erreur sur {f}: {e}")

if __name__ == "__main__":
    process()
    print("Terminé ! Dataset prêt.")