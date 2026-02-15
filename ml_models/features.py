import pandas as pd
import numpy as np

class FeatureEngine:
    """
    Usine à caractéristiques (Feature Factory).
    Transforme des prix bruts (Non-Stationnaires) en indicateurs (Stationnaires)
    digestes pour des algorithmes de Machine Learning.
    """
    
    def compute_all_features(self, df, window_short=10, window_long=30):
        """
        Prend un DataFrame avec une colonne 'Close' et ajoute les Features.
        """
        data = df.copy()
        
        # --- 1. TRANSFORMATION DE BASE (STATIONNARITÉ) ---
        # On ne travaille jamais sur le Prix, mais sur le Log-Rendement
        data['log_ret'] = np.log(data['Close'] / data['Close'].shift(1))
        
        # --- 2. VOLATILITÉ (RISQUE) ---
        # Écart-type sur la fenêtre longue (ex: 30 jours)
        data['volatility'] = data['log_ret'].rolling(window=window_long).std()
        
        # --- 3. MOMENTUM (RSI - Relative Strength Index) ---
        # Formule vectorisée du RSI (0 à 100)
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        data['rsi'] = 100 - (100 / (1 + rs))
        
        # Normalisation du RSI pour l'IA (centré autour de 0.5 entre 0 et 1)
        data['rsi_norm'] = data['rsi'] / 100.0
        
        # --- 4. TENDANCE (MACD) ---
        # Moving Average Convergence Divergence
        ema_12 = data['Close'].ewm(span=12, adjust=False).mean()
        ema_26 = data['Close'].ewm(span=26, adjust=False).mean()
        data['macd'] = ema_12 - ema_26
        
        # --- 5. RETOUR À LA MOYENNE (Distance) ---
        # Z-Score du prix par rapport à sa moyenne 50 jours
        # (Prix - Moyenne) / Ecart-Type
        ma_50 = data['Close'].rolling(window=50).mean()
        std_50 = data['Close'].rolling(window=50).std()
        data['z_score_price'] = (data['Close'] - ma_50) / std_50
        
        # --- 6. LA CIBLE (TARGET - CE QU'ON VEUT PRÉDIRE) ---
        # Objectif : Le rendement de DEMAIN sera-t-il positif ?
        # .shift(-1) permet de "ramener" le futur sur la ligne d'aujourd'hui.
        # Si log_ret(t+1) > 0 -> 1 (Hausse)
        # Sinon -> 0 (Baisse)
        data['Target_Return'] = data['log_ret'].shift(-1)
        data['Target_Direction'] = np.where(data['Target_Return'] > 0, 1, 0)
        
        # Nettoyage des NaN générés par les fenêtres glissantes
        data.dropna(inplace=True)
        
        return data