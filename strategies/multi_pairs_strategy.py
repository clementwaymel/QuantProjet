import numpy as np
from collections import deque
from core.event import SignalEvent
from core.strategy import Strategy
from quant_maths.kalman import KalmanRegression
from quant_maths.fractals import calculate_hurst_exponent
from quant_maths.statistics import calculate_half_life

class MultiPairsTradingStrategy(Strategy):
    """
    Version Définitive : Multi-Pairs + Kalman Bivarié + Hurst + Half-Life + Risk Parity
    """
    def __init__(self, bars, events_queue, pairs_list, z_entry=2.5, z_exit=0.0):
        self.bars = bars
        self.events_queue = events_queue
        self.pairs_list = pairs_list
        
        self.z_entry = z_entry
        self.z_exit = z_exit
        self.z_max_entry = 3.5 # Disjoncteur anti-Fat Tails
        self.z_stop_loss = 4.5 # Stop loss d'urgence
        
        # Dictionnaires pour suivre l'état de chaque paire de manière indépendante
        self.kalmans = {}
        self.spread_histories = {}
        self.states = {}
        
        for y, x in self.pairs_list:
            key = f"{y}_{x}"
            # Filtre de Kalman "lent" pour éviter l'overfitting
            self.kalmans[key] = KalmanRegression(delta=1e-6, R=1e-3)
            self.spread_histories[key] = deque(maxlen=100) # Fenêtre longue pour Hurst
            self.states[key] = 'NEUTRAL'

    def calculate_signals(self, event):
        if event.type == 'MARKET':
            for ticker_y, ticker_x in self.pairs_list:
                key = f"{ticker_y}_{ticker_x}"
                
                price_y = self.bars.get_latest_bar(ticker_y)
                price_x = self.bars.get_latest_bar(ticker_x)
                
                if price_y is None or price_x is None: continue

                y_val = price_y['close']
                x_val = price_x['close']

                # 1. Mise à jour Kalman
                kalman = self.kalmans[key]
                kalman.update(x_val, y_val)
                current_beta = kalman.get_beta()
                current_alpha = kalman.get_alpha()
                
                # 2. Calcul du Spread
                current_spread = y_val - (current_beta * x_val + current_alpha)
                self.spread_histories[key].append(current_spread)
                
                # Il nous faut au moins 60 jours pour que Hurst soit pertinent
                if len(self.spread_histories[key]) < 60: continue

                # 3. Statistiques et Z-Score (sur les 30 derniers jours)
                spreads_array = list(self.spread_histories[key])
                recent_spreads = spreads_array[-30:]
                
                mean_spread = np.mean(recent_spreads)
                std_spread = np.std(recent_spreads) # Utilisé pour le sizing par le portfolio
                
                if std_spread == 0: continue
                z_score = (current_spread - mean_spread) / std_spread
                
                current_state = self.states[key]

                # 4. GESTION DES RISQUES (STOP LOSS EN COURS DE TRADE)
                if current_state != 'NEUTRAL':
                    if current_state == 'LONG' and z_score < -self.z_stop_loss:
                        print(f"[RISK] Stop Loss sur {key} (Z={z_score:.2f})")
                        self._send_pair_signal(ticker_y, 'EXIT', ticker_x, 'EXIT', std_spread)
                        self.states[key] = 'NEUTRAL'
                        continue
                        
                    elif current_state == 'SHORT' and z_score > self.z_stop_loss:
                        print(f"[RISK] Stop Loss sur {key} (Z={z_score:.2f})")
                        self._send_pair_signal(ticker_y, 'EXIT', ticker_x, 'EXIT', std_spread)
                        self.states[key] = 'NEUTRAL'
                        continue

                # 5. FILTRES AVANCÉS (Pré-Trade)
                is_valid_trade = True
                
                # A. Filtre de Hurst (Détection de tendance)
                hurst = calculate_hurst_exponent(spreads_array)
                if hurst > 0.5:
                    is_valid_trade = False
                    
                # B. Demi-Vie (Vitesse de retour à la moyenne)
                hl = calculate_half_life(recent_spreads)
                if hl is not None and (hl > 25 or hl < 1):
                    is_valid_trade = False

                # 6. LOGIQUE D'ENTRÉE ET DE SORTIE
                if current_state == 'NEUTRAL' and is_valid_trade:
                    # Troncature de la queue gauche (Achat du spread)
                    if -self.z_max_entry < z_score < -self.z_entry:
                        print(f"[{key}] BUY SPREAD (Z={z_score:.2f} | H={hurst:.2f} | HL={hl:.1f}j)")
                        self._send_pair_signal(ticker_y, 'LONG', ticker_x, 'SHORT', std_spread)
                        self.states[key] = 'LONG'
                        
                    # Troncature de la queue droite (Vente du spread)
                    elif self.z_entry < z_score < self.z_max_entry:
                        print(f"[{key}] SELL SPREAD (Z={z_score:.2f} | H={hurst:.2f} | HL={hl:.1f}j)")
                        self._send_pair_signal(ticker_y, 'SHORT', ticker_x, 'LONG', std_spread)
                        self.states[key] = 'SHORT'

                # Sorties classiques (Take Profit)
                elif current_state == 'LONG' and z_score > -self.z_exit:
                    self._send_pair_signal(ticker_y, 'EXIT', ticker_x, 'EXIT', std_spread)
                    self.states[key] = 'NEUTRAL'

                elif current_state == 'SHORT' and z_score < self.z_exit:
                    self._send_pair_signal(ticker_y, 'EXIT', ticker_x, 'EXIT', std_spread)
                    self.states[key] = 'NEUTRAL'

    def _send_pair_signal(self, tick_y, dir_y, tick_x, dir_x, volatility):
        safe_vol = volatility if volatility > 0 else 0.01
        self.events_queue.put(SignalEvent(tick_y, None, dir_y, strength=1.0, est_volatility=safe_vol))
        self.events_queue.put(SignalEvent(tick_x, None, dir_x, strength=1.0, est_volatility=safe_vol))