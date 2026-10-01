import os
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm

def extract_features(file_path):
    try:
        y, sr = librosa.load(file_path, duration=30)  # load first 30 seconds, one file gets skipped due to size
    except Exception as e:
        print(f"Skipping file {file_path} due to load error: {e}")
        return None

    features = {}

    # MFCCs
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    for i in range(1, 14):
        features[f'mfcc_{i}'] = np.mean(mfccs[i-1])

    # Delta MFCCs
    delta_mfccs = librosa.feature.delta(mfccs)
    for i in range(1, 14):
        features[f'delta_mfcc_{i}'] = np.mean(delta_mfccs[i-1])

    # Chroma
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    features['chroma_mean'] = np.mean(chroma)

    # Spectral Contrast
    spec_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    features['spectral_contrast_mean'] = np.mean(spec_contrast)

    # Spectral Rolloff
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    features['spectral_rolloff_mean'] = np.mean(rolloff)

    # Spectral Bandwidth
    spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    features['spectral_bandwidth_mean'] = np.mean(spec_bw)

    # Zero-Crossing Rate (ZCR)
    features['zcr'] = np.mean(librosa.feature.zero_crossing_rate(y))

    # Root-Mean-Square Energy
    features['rms'] = np.mean(librosa.feature.rms(y=y))

    # Tempo
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    features['tempo'] = tempo

    return features

def process_gtzan(base_path=r"C:/Users/noaha/OneDrive\Desktop/archive\Data/genres_original", output_csv='GTZAN_features.csv'):
    all_features = []
    skipped_files = []

    genres = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]

    for genre in genres:
        genre_path = os.path.join(base_path, genre)
        files = [f for f in os.listdir(genre_path) if f.endswith('.wav')]
        print(f"Processing genre: {genre}, {len(files)} files")

        for file in tqdm(files):
            file_path = os.path.join(genre_path, file)
            feats = extract_features(file_path)
            if feats is None:
                skipped_files.append(file_path)
                continue
            feats['genre'] = genre
            feats['file_name'] = file
            all_features.append(feats)

    # Convert to DataFrame
    df = pd.DataFrame(all_features)

    # Save to CSV
    df.to_csv(output_csv, index=False)
    print(f"Feature extraction complete. CSV saved as {output_csv}")

    if skipped_files:
        print(f"\nSkipped {len(skipped_files)} files due to errors:")
        for f in skipped_files:
            print(f)

if __name__ == "__main__":
    process_gtzan()
