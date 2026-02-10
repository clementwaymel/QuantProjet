import numpy as np
import pandas as pd
from collections import deque

# --- On garde la classe RollingOLS si tu l'utilises ailleurs ---
class RollingOLS:
    def __init__(self, window_size=30):
        self.window_size = window_size
        self.x_history = deque(maxlen=window_size)
        self.y_history = deque(maxlen=window_size)
        self.beta = 0.0
        self.alpha = 0.0
        self.is_ready = False

    def update(self, x, y):
        self.x_history.append(x)
        self.y_history.append(y)
        if len(self.x_history) == self.window_size:
            self.is_ready = True
            self._compute_regression()

    def _compute_regression(self):
        x_array = np.array(self.x_history)
        y_array = np.array(self.y_history)
        cov_matrix = np.cov(x_array, y_array, bias=True)
        variance_x = cov_matrix[0, 0]
        covariance_xy = cov_matrix[0, 1]
        if variance_x == 0: self.beta = 0
        else: self.beta = covariance_xy / variance_x
        self.alpha = np.mean(y_array) - (self.beta * np.mean(x_array))

    def get_residual(self, x, y):
        if not self.is_ready: return 0.0
        return y - (self.beta * x + self.alpha)

# --- NOUVELLE FONCTION CRUCIALE ---
def calculate_half_life(spread_series):
    """
    Estime la Demi-Vie (Half-Life) d'un spread via un processus d'Ornstein-Uhlenbeck.
    Modèle : dx_t = -theta * (x_t - mu) * dt + sigma * dW_t
    
    Retourne : Le nombre de jours espérés pour que l'écart se résorbe de moitié.
    """
    # On a besoin d'un minimum de données pour que la régression soit fiable
    if len(spread_series) < 30:
        return None
        
    spread = np.array(spread_series)
    
    # Lagged spread (décalé d'un jour)
    # Spread_lag = Spread[t-1]
    spread_lag = spread[:-1]
    # Spread_ret = Spread[t]
    spread_ret = spread[1:]
    
    # Régression : (Spread_t - Spread_t-1) ~ Theta * (Spread_t-1 - Mean)
    delta_spread = spread_ret - spread_lag
    
    # On centre sur la moyenne locale
    mu = np.mean(spread_lag)
    X = spread_lag - mu
    Y = delta_spread
    
    # Régression Linéaire simple sans intercept (Y = -theta * X) pour trouver Theta
    # Formule OLS (Moindres Carrés) : theta = - sum(x*y) / sum(x^2)
    denom = np.sum(X**2)
    if denom == 0: return None
    
    theta = -np.sum(X * Y) / denom
    
    # Si theta est proche de 0 ou négatif, pas de retour à la moyenne (Marche aléatoire ou Divergence)
    if theta <= 0:
        return 1000.0 # Valeur arbitraire "Infinie"
        
    # Calcul Half-Life : temps pour diviser l'écart par 2
    # HL = ln(2) / theta
    half_life = np.log(2) / theta
    return half_life