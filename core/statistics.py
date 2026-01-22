import numpy as np
from collections import deque

class RollingOLS:
    """
    Régression des Moindres Carrés Ordinaires (OLS) sur fenêtre glissante.
    Permet de calculer le Hedge Ratio (Beta) dynamique entre deux actifs.
    """
    def __init__(self, window_size=30):
        self.window_size = window_size
        self.x_history = deque(maxlen=window_size) # Variable explicative (ex: PEP)
        self.y_history = deque(maxlen=window_size) # Variable dépendante (ex: KO)
        
        self.beta = 0.0
        self.alpha = 0.0
        self.is_ready = False

    def update(self, x, y):
        """
        Ajoute un nouveau point (x_t, y_t) et recalcule la régression.
        """
        self.x_history.append(x)
        self.y_history.append(y)
        
        if len(self.x_history) == self.window_size:
            self.is_ready = True
            self._compute_regression()

    def _compute_regression(self):
        """
        Calcul vectorisé de Beta et Alpha.
        Formule : Beta = Cov(X,Y) / Var(X)
        """
        x_array = np.array(self.x_history)
        y_array = np.array(self.y_history)
        
        # Calcul de la matrice de covariance [[Var(X), Cov(X,Y)], [Cov(Y,X), Var(Y)]]
        # bias=True pour variance population, False pour échantillon (peu d'impact ici)
        cov_matrix = np.cov(x_array, y_array, bias=True)
        
        variance_x = cov_matrix[0, 0]
        covariance_xy = cov_matrix[0, 1]
        
        # Protection division par zéro
        if variance_x == 0:
            self.beta = 0
        else:
            self.beta = covariance_xy / variance_x
            
        # Alpha (Ordonnée à l'origine) = Moyenne(Y) - Beta * Moyenne(X)
        self.alpha = np.mean(y_array) - (self.beta * np.mean(x_array))

    def get_params(self):
        """Retourne les paramètres actuels du modèle."""
        return self.beta, self.alpha

    def get_residual(self, x, y):
        """
        Calcule l'erreur du modèle pour le point actuel (le Spread).
        Epsilon = Y - (Beta * X + Alpha)
        """
        if not self.is_ready:
            return 0.0
        # On utilise le Beta actuel pour voir si le point actuel s'écarte
        return y - (self.beta * x + self.alpha)

def calculate_half_life(spread_series):
    """
    Estime la Demi-Vie (Half-Life) d'un spread via un processus Ornstein-Uhlenbeck.
    Retourne le nombre de jours espérés pour revenir à la moyenne.
    """
    # On a besoin d'au moins quelques points
    if len(spread_series) < 10:
        return None
        
    spread = np.array(spread_series)
    
    # Lagged spread (décalé d'un jour)
    # Spread_lag = Spread[0:-1] (Hier)
    # Spread_ret = Spread[1:] (Aujourd'hui)
    spread_lag = spread[:-1]
    spread_ret = spread[1:]
    
    # Régression : Spread_today - Spread_yesterday = Theta * (Mean - Spread_yesterday)
    # Forme simplifiée : Delta_Spread ~ Theta * Spread_Lag
    delta_spread = spread_ret - spread_lag
    
    # On centre les données (Mean = 0 supposé pour le spread résiduel)
    # Régression Linéaire simple sans intercept (Y = aX) pour trouver Theta
    # Y = Delta_Spread, X = Spread_Lag - Mean
    
    X = spread_lag - np.mean(spread_lag)
    Y = delta_spread
    
    # Formule OLS (Moindres Carrés) : a = sum(xy) / sum(xx)
    denom = np.sum(X**2)
    if denom == 0: return None
    
    theta = -np.sum(X * Y) / denom
    
    # Si theta est négatif ou nul, pas de retour à la moyenne (Divergence)
    if theta <= 0:
        return None
        
    # Calcul Half-Life
    half_life = np.log(2) / theta
    return half_life