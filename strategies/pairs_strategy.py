import numpy as np
from collections import deque
from core.event import SignalEvent
from core.strategy import Strategy
from core.statistics import RollingOLS, calculate_half_life

class PairsTradingStrategy(Strategy):
    """
    Version 2.0 : Utilise une Régression Dynamique (Rolling OLS).
    Le Hedge Ratio s'adapte au marché.
    """
    def __init__(self, bars, events_queue, window_size=30, z_entry=2.0, z_exit=0.5):
        self.bars = bars
        self.events_queue = events_queue
        
        self.window_size = window_size
        self.z_entry = z_entry
        self.z_exit = z_exit
        
        self.ticker_y = "KO" # Variable dépendante
        self.ticker_x = "PEP" # Variable explicative
        
        # --- NOYAU STATISTIQUE ---
        # Objet qui va apprendre la relation KO/PEP jour après jour
        self.ols_model = RollingOLS(window_size=window_size)
        
        # Pour le Z-Score, on garde l'historique des Résidus (Spreads)
        self.residuals_history = deque(maxlen=window_size)
        
        self.state = 'NEUTRAL'

    def calculate_signals(self, event):
        if event.type == 'MARKET':
            # 1. Récupération des prix
            price_y = self.bars.get_latest_bar(self.ticker_y)
            price_x = self.bars.get_latest_bar(self.ticker_x)
            
            if price_y is None or price_x is None:
                return

            y_val = price_y['close']
            x_val = price_x['close']

            # 2. Mise à jour du modèle mathématique
            # On nourrit le modèle avec les nouvelles données
            self.ols_model.update(x_val, y_val)
            
            if not self.ols_model.is_ready:
                return # On attend d'avoir 30 jours de données

            # 3. Calcul du Spread DYNAMIQUE (Résidu)
            # Contrairement à avant où Beta était fixe, ici Beta change tous les jours !
            current_spread = self.ols_model.get_residual(x_val, y_val)
            
            # On stocke ce spread
            self.residuals_history.append(current_spread)
            
            # 4. Calcul du Z-Score sur les résidus
            spreads = np.array(self.residuals_history)
            mean_spread = np.mean(spreads)
            std_spread = np.std(spreads)
            
            if std_spread == 0: return
            
            z_score = (current_spread - mean_spread) / std_spread
            
            # Affichage périodique du Beta (Pour voir s'il bouge)
            # beta, alpha = self.ols_model.get_params()
            # print(f"Beta actuel: {beta:.4f} | Z-Score: {z_score:.2f}")

            # 5. LOGIQUE DE TRADING (Identique à la V1)
            # La logique ne change pas, c'est la QUALITÉ du signal qui s'améliore
            # --- NOUVEAU : CALCUL DE LA VITESSE DE RETOUR ---
            # On regarde l'historique récent des spreads
            current_half_life = calculate_half_life(self.residuals_history)
            
            # FILTRE DE QUALITÉ :
            # Si le Half-Life est > 15 jours, l'élastique est trop mou. On ne fait rien.
            is_fast_reversion = False
            if current_half_life is not None and current_half_life < 15:
                is_fast_reversion = True
            
            # On affiche pour le debug
            # if current_half_life: print(f"Half-Life: {current_half_life:.1f} jours")

            # 5. LOGIQUE DE TRADING AVEC FILTRE
            if self.state == 'NEUTRAL':
                # On ajoute la condition "and is_fast_reversion"
                if z_score < -self.z_entry and is_fast_reversion:
                    print(f"[Strategy] Z-Score {z_score:.2f} | HL {current_half_life:.1f}j (Rapide). BUY SPREAD")
                    self._send_signal(self.ticker_y, 'LONG')
                    self._send_signal(self.ticker_x, 'SHORT')
                    self.state = 'LONG_SPREAD'
                    
                elif z_score > self.z_entry and is_fast_reversion:
                    print(f"[Strategy] Z-Score {z_score:.2f} | HL {current_half_life:.1f}j (Rapide). SELL SPREAD")
                    self._send_signal(self.ticker_y, 'SHORT')
                    self._send_signal(self.ticker_x, 'LONG')
                    self.state = 'SHORT_SPREAD'

            elif self.state == 'LONG_SPREAD':
                if z_score > -self.z_exit:
                    self._send_signal(self.ticker_y, 'EXIT')
                    self._send_signal(self.ticker_x, 'EXIT')
                    self.state = 'NEUTRAL'

            elif self.state == 'SHORT_SPREAD':
                if z_score < self.z_exit:
                    self._send_signal(self.ticker_y, 'EXIT')
                    self._send_signal(self.ticker_x, 'EXIT')
                    self.state = 'NEUTRAL'
        

    def _send_signal(self, ticker, direction):
        sig = SignalEvent(ticker, None, direction, strength=1.0)
        self.events_queue.put(sig)

    

