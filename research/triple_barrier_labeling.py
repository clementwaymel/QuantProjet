import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Permet d'importer depuis le dossier parent
sys.path.append(os.getcwd()) 

from core.database_manager import QuantDatabase
from quant_maths.kalman import KalmanRegression
from ml_models.features import FeatureEngine

print("--- GÉNÉRATION DU DATASET (META-LABELING TRIPLE BARRIÈRE) ---")

# --- 1. CONFIGURATION ---
TICKER_Y = 'KO'
TICKER_X = 'PEP'
Z_ENTRY_LAX = 0.75  # Seuil permissif pour générer beaucoup de signaux
PT_SL_RATIO = [1.0, 1.0] # [Take-Profit, Stop-Loss] exprimés en multiples de volatilité
MAX_HOLDING_PERIOD = 20 # Barrière verticale (20 jours max)

db = QuantDatabase()
df_y = db.get_ticker_data(TICKER_Y)
df_x = db.get_ticker_data(TICKER_X)

df = pd.DataFrame({'Y': df_y, 'X': df_x}).dropna()

# --- 2. RECONSTRUCTION DU MOTEUR DE BASE (Kalman + Z-Score) ---
print("1. Calcul du Spread et des signaux primaires...")
kalman = KalmanRegression(delta=1e-6, R=1e-3)
spreads, z_scores, volatilities = [], [], []

for i in range(len(df)):
    y_val, x_val = df['Y'].iloc[i], df['X'].iloc[i]
    kalman.update(x_val, y_val)
    
    current_spread = y_val - (kalman.get_beta() * x_val + kalman.get_alpha())
    spreads.append(current_spread)
    
    if i >= 30:
        window = spreads[-30:]
        std = np.std(window)
        z = (current_spread - np.mean(window)) / std if std > 0 else 0
    else:
        std, z = 0, 0
        
    z_scores.append(z)
    volatilities.append(std)

df['Spread'] = spreads
df['Z_Score'] = z_scores
df['Spread_Vol'] = volatilities

# Identification des signaux
df['Signal_Side'] = 0
df.loc[df['Z_Score'] < -Z_ENTRY_LAX, 'Signal_Side'] = 1  # Long Spread
df.loc[df['Z_Score'] > Z_ENTRY_LAX, 'Signal_Side'] = -1  # Short Spread

# --- 3. APPLICATION DE LA TRIPLE BARRIÈRE ---
print("2. Application de la méthode de la Triple Barrière...")
labels = pd.Series(index=df.index, dtype=float)
labels[:] = np.nan
first_touch_dates = pd.Series(index=df.index, dtype='datetime64[ns]')

signal_indices = df[df['Signal_Side'] != 0].index

for idx in signal_indices:
    side = df.loc[idx, 'Signal_Side']
    start_idx_num = df.index.get_loc(idx)
    
    # On définit la fin de la fenêtre (Barrière Verticale)
    end_idx_num = min(start_idx_num + MAX_HOLDING_PERIOD, len(df) - 1)
    
    # Le chemin du spread dans le futur
    path = df['Spread'].iloc[start_idx_num : end_idx_num + 1]
    
    # On centre le chemin sur 0 pour mesurer la variation
    path_returns = (path - path.iloc[0]) * side
    
    # Définition des barrières dynamiques
    vol = df.loc[idx, 'Spread_Vol']
    if vol == 0: continue
        
    upper_barrier = vol * PT_SL_RATIO[0]
    lower_barrier = -vol * PT_SL_RATIO[1]
    
    # Recherche du First Passage Time
    touch_idx = None
    label = np.nan
    
    for t_step, ret in path_returns.items():
        if ret >= upper_barrier:
            touch_idx = t_step
            label = 1.0 # Succès
            break
        elif ret <= lower_barrier:
            touch_idx = t_step
            label = 0.0 # Échec
            break
            
    # Si aucune barrière n'est touchée, c'est la barrière verticale qui agit
    if touch_idx is None:
        touch_idx = path.index[-1]
        label = 1.0 if path_returns.iloc[-1] > 0 else 0.0
        
    labels[idx] = label
    first_touch_dates[idx] = touch_idx

df['Label'] = labels

# --- 4. CALCUL DES FEATURES (LE CONTEXTE) ---
print("3. Calcul des Features par le FeatureEngine...")
# On crée un faux DataFrame de Prix basé sur le Spread pour utiliser ton FeatureEngine
df_pseudo = pd.DataFrame({'Close': df['Spread'] + 100}) # +100 pour éviter les prix négatifs
engine = FeatureEngine()
features = engine.compute_all_features(df_pseudo)

# --- 5. FUSION ET SAUVEGARDE DU DATASET FINAL ---
print("4. Création du Dataset Machine Learning...")
dataset = features.copy()
dataset['Label'] = df['Label']

# On ne garde QUE les lignes où un signal a été généré
dataset = dataset.dropna(subset=['Label'])

# Suppression des colonnes devenues inutiles/bruitées
cols_to_drop = ['log_ret', 'Target_Return', 'Target_Direction', 'Close']
dataset = dataset.drop(columns=[c for c in cols_to_drop if c in dataset.columns])

print(f"\nDataset généré : {len(dataset)} échantillons labellisés.")
print("Répartition des classes :")
print(dataset['Label'].value_counts(normalize=True) * 100)

if not os.path.exists('data'):
    os.makedirs('data')
dataset.to_csv('data/ml_dataset_ko_pep.csv')
print("\n>>> Fichier sauvegardé sous 'data/ml_dataset_ko_pep.csv'")