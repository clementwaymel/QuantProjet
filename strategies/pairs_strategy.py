import numpy as np
from collections import deque
from core.event import SignalEvent
from core.strategy import Strategy
from quant_maths.kalman import KalmanRegression
from quant_maths.fractals import calculate_hurst_exponent
from quant_maths.statistics import calculate_half_life # <--- NOUVEL IMPORT

class PairsTradingStrategy(Strategy):
    """
    Version 5.0 (Blindée) : KALMAN + HURST + HALF-LIFE + STOP LOSS.
    """
    def __init__(self, bars, events_queue, z_entry=2.0, z_exit=0.0):
        self.bars = bars
        self.events_queue = events_queue
        
        self.z_entry = z_entry
        self.z_exit = z_exit # On sort quand Z revient à 0
        
        self.ticker_y = "KO"
        self.ticker_x = "PEP"
        
        # Moteur Kalman
        self.kalman = KalmanRegression(delta=1e-7, R=1e-3)
        
        # Historique pour Z-Score, Hurst et Half-Life
        self.spread_history = deque(maxlen=100) 
        
        # Gestion de position
        self.state = 'NEUTRAL'
        self.entry_price_spread = 0.0 # Pour calculer le P&L latent

    def calculate_signals(self, event):
        if event.type == 'MARKET':
            price_y = self.bars.get_latest_bar(self.ticker_y)
            price_x = self.bars.get_latest_bar(self.ticker_x)
            
            if price_y is None or price_x is None: return

            y_val = price_y['close']
            x_val = price_x['close']

            # 1. Mise à jour Kalman
            self.kalman.update(x_val, y_val)
            current_beta = self.kalman.get_beta()
            
            # 2. Calcul Spread
            current_spread = y_val - (current_beta * x_val)
            self.spread_history.append(current_spread)
            
            if len(self.spread_history) < 30: return

            # 3. Calcul Z-Score
            recent_spreads = list(self.spread_history)[-30:] # Fenêtre courte pour Z
            mean_spread = np.mean(recent_spreads)
            std_spread = np.std(recent_spreads)
            
            if std_spread == 0: return
            z_score = (current_spread - mean_spread) / std_spread
            
            # 4. FILTRES AVANCÉS (Hurst + Half-Life)
            is_valid_trade = True
            
            # A. Filtre Fractal (Hurst) sur fenêtre longue
            if len(self.spread_history) > 60:
                hurst = calculate_hurst_exponent(self.spread_history)
                if hurst > 0.5: # Si > 0.5, c'est du Trending (Divergence)
                    is_valid_trade = False
            
            # B. Filtre Temporel (Half-Life) sur fenêtre récente
            hl = calculate_half_life(recent_spreads)
            # Si le HL est > 20 jours, c'est trop lent, on risque de rester coincé
            if hl is not None and hl > 20: 
                is_valid_trade = False

            # 5. GESTION DES RISQUES (STOP LOSS)
            # Si on est en position, on vérifie si ça tourne mal
            if self.state != 'NEUTRAL':
                # Estimation grossière du P&L en Z-Score
                # Si on est LONG et que Z continue de descendre violemment (< -4) -> STOP
                # Si on est SHORT et que Z continue de monter violemment (> +4) -> STOP
                stop_loss_z = 4.0 
                
                if self.state == 'LONG_SPREAD' and z_score < -stop_loss_z:
                    print(f"[RISK] STOP LOSS ACTIVÉ (Z={z_score:.2f}). EXIT LONG.")
                    self._exit_positions()
                    return
                
                if self.state == 'SHORT_SPREAD' and z_score > stop_loss_z:
                    print(f"[RISK] STOP LOSS ACTIVÉ (Z={z_score:.2f}). EXIT SHORT.")
                    self._exit_positions()
                    return

            # 6. LOGIQUE D'ENTRÉE (Corrigée avec intervalle de confiance)
            z_max_entry = 3.5 # Disjoncteur stochastique (Anti-Fat Tails)
            
            if self.state == 'NEUTRAL' and is_valid_trade:
                # Troncature de la queue de distribution gauche
                if -z_max_entry < z_score < -self.z_entry:
                    print(f"[Signal] Buy Spread (Z={z_score:.2f} | HL={hl:.1f}j).")
                    self._send_signal(self.ticker_y, 'LONG')
                    self._send_signal(self.ticker_x, 'SHORT')
                    self.state = 'LONG_SPREAD'
                    self.entry_price_spread = current_spread
                    
                # Troncature de la queue de distribution droite
                elif self.z_entry < z_score < z_max_entry:
                    print(f"[Signal] Sell Spread (Z={z_score:.2f} | HL={hl:.1f}j).")
                    self._send_signal(self.ticker_y, 'SHORT')
                    self._send_signal(self.ticker_x, 'LONG')
                    self.state = 'SHORT_SPREAD'
                    self.entry_price_spread = current_spread
            # 7. LOGIQUE DE SORTIE (Take Profit)
            elif self.state == 'LONG_SPREAD' and z_score > -self.z_exit:
                self._exit_positions()

            elif self.state == 'SHORT_SPREAD' and z_score < self.z_exit:
                self._exit_positions()

    def _exit_positions(self):
        self._send_signal(self.ticker_y, 'EXIT')
        self._send_signal(self.ticker_x, 'EXIT')
        self.state = 'NEUTRAL'

    def _send_signal(self, ticker, direction):
        sig = SignalEvent(ticker, None, direction, strength=1.0)
        self.events_queue.put(sig)


