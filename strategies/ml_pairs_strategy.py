import numpy as np
import pandas as pd
import joblib
import os
from collections import deque
from core.event import SignalEvent
from core.strategy import Strategy
from quant_maths.kalman import KalmanRegression
from ml_models.features import FeatureEngine

class MLPairsTradingStrategy(Strategy):
    """
    Stratégie Hybride : Kalman (Génération de signal laxiste) + Random Forest (Filtre asymétrique > 65%).
    """
    def __init__(self, bars, events_queue, model_path="models/meta_model_ko_pep.joblib", z_entry=0.75, z_exit=0.0):
        self.bars = bars
        self.events_queue = events_queue
        
        # On utilise un Z-Score bas (laxiste) car c'est l'IA qui fait le tri
        self.z_entry = z_entry
        self.z_exit = z_exit
        
        self.ticker_y = "KO"
        self.ticker_x = "PEP"
        
        # 1. Chargement du Cerveau IA
        if os.path.exists(model_path):
            self.model = joblib.load(model_path)
            print(f"[Strategy] Modèle IA chargé : {model_path}")
        else:
            raise FileNotFoundError(f"Modèle introuvable : {model_path}")
            
        # 2. Moteur Kalman (Paramétrage lent)
        self.kalman = KalmanRegression(delta=1e-6, R=1e-3)
        
        self.spread_history = deque(maxlen=60)
        self.price_history_y = deque(maxlen=60)
        
        self.state = 'NEUTRAL'
        self.feature_engine = FeatureEngine()

    def calculate_signals(self, event):
        if event.type == 'MARKET':
            price_y_bar = self.bars.get_latest_bar(self.ticker_y)
            price_x_bar = self.bars.get_latest_bar(self.ticker_x)
            
            if price_y_bar is None or price_x_bar is None: return

            y_val = price_y_bar['close']
            x_val = price_x_bar['close']
            
            self.price_history_y.append(y_val)

            self.kalman.update(x_val, y_val)
            current_beta = self.kalman.get_beta()
            current_alpha = self.kalman.get_alpha()
            
            current_spread = y_val - (current_beta * x_val + current_alpha)
            self.spread_history.append(current_spread)
            
            if len(self.spread_history) < 30: return

            recent_spreads = list(self.spread_history)[-30:]
            z_score = (current_spread - np.mean(recent_spreads)) / np.std(recent_spreads)
            
            ai_confirmation = False
            
            # Si le signal brut est déclenché
            if (z_score < -self.z_entry or z_score > self.z_entry) and len(self.price_history_y) > 50:
                
                df_hist = pd.DataFrame({'Close': list(self.price_history_y)})
                
                try:
                    features = self.feature_engine.compute_all_features(df_hist)
                    
                    if not features.empty:
                        # ATTENTION : L'ordre des features doit être EXACTEMENT le même que lors de l'entraînement
                        feature_cols = ['volatility', 'rsi_norm', 'macd', 'z_score_price']
                        current_features = features.iloc[[-1]][feature_cols]
                        
                        # --- LE DISJONCTEUR PROBABILISTE (65%) ---
                        proba_success = self.model.predict_proba(current_features)[0][1]
                        
                        if proba_success >= 0.65:
                            ai_confirmation = True
                            print(f"[IA Check] Signal VALIDÉ (Confiance: {proba_success:.1%} | Z={z_score:.2f})")
                        else:
                            # On décommente ce print pour voir l'IA travailler en temps réel
                            print(f"[IA Check] Signal REJETÉ (Confiance: {proba_success:.1%} | Z={z_score:.2f})")
                            
                except Exception as e:
                    pass

            # --- LOGIQUE DE TRADING ---
            if self.state == 'NEUTRAL':
                if z_score < -self.z_entry and ai_confirmation:
                    self._send_signal(self.ticker_y, 'LONG')
                    self._send_signal(self.ticker_x, 'SHORT')
                    self.state = 'LONG_SPREAD'
                    
                elif z_score > self.z_entry and ai_confirmation:
                    self._send_signal(self.ticker_y, 'SHORT')
                    self._send_signal(self.ticker_x, 'LONG')
                    self.state = 'SHORT_SPREAD'

            elif self.state == 'LONG_SPREAD' and z_score > -self.z_exit:
                self._send_signal(self.ticker_y, 'EXIT')
                self._send_signal(self.ticker_x, 'EXIT')
                self.state = 'NEUTRAL'

            elif self.state == 'SHORT_SPREAD' and z_score < self.z_exit:
                self._send_signal(self.ticker_y, 'EXIT')
                self._send_signal(self.ticker_x, 'EXIT')
                self.state = 'NEUTRAL'

    def _send_signal(self, ticker, direction):
        sig = SignalEvent(ticker, None, direction, strength=1.0)
        self.events_queue.put(sig)