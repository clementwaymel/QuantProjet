import numpy as np

class KalmanRegression:
    """
    Filtre de Kalman 1D bivarié pour estimer dynamiquement l'intercepte (Alpha) et la pente (Beta) d'une régression.
    Modèle : y_t = alpha_t + beta_t * x_t + epsilon_t
    """
    def __init__(self, delta=1e-4, R=1e-3):
        # delta : variance du Random Walk pour l'évolution des paramètres.
        self.delta = delta
        # R : variance du bruit d'observation (variance de l'erreur epsilon_t).
        self.R = R

        # --- ÉTAT DU SYSTÈME ---
        # x : Le vecteur d'état [alpha_t, beta_t]^T (taille 2x1)
        self.x = np.zeros((2, 1)) 
        
        # P : La Matrice de Covariance de l'erreur d'estimation (taille 2x2)
        # On initialise avec une grande incertitude
        self.P = np.eye(2) * 10.0

    def update(self, x_meas, y_meas):
        """
        Met à jour l'estimation (Alpha, Beta) avec une nouvelle observation (prix X et Y).
        """
        # H : Matrice d'observation [1, x_t] (taille 1x2)
        H = np.array([[1.0, x_meas]])
        
        # --- 1. PRÉDICTION ---
        # Le modèle d'évolution est une marche aléatoire : x_{t|t-1} = x_{t-1|t-1}
        x_pred = self.x
        
        # Q : Matrice de covariance du bruit de processus (taille 2x2)
        # On utilise self.delta pour la diagonale. Pour des modèles plus avancés,
        # Q pourrait être proportionnelle à P ou adaptative.
        Q = np.eye(2) * self.delta
        P_pred = self.P + Q
        
        # --- 2. MISE À JOUR ---
        # Innovation (erreur de prédiction) : e_t = y_t - y_{t|t-1}
        # y_{t|t-1} = H * x_pred = alpha_pred + beta_pred * x_meas
        error = y_meas - np.dot(H, x_pred)
        
        # S : Variance de l'innovation (scalaire, car l'observation Y est 1D)
        # S_t = H * P_{t|t-1} * H^T + R
        S = np.dot(H, np.dot(P_pred, H.T)) + self.R
        
        # K : Gain de Kalman (taille 2x1)
        # K_t = P_{t|t-1} * H^T * S_t^{-1}
        # Comme S est un scalaire (1x1), l'inversion est une simple division.
        K = np.dot(P_pred, H.T) / S
        
        # Mise à jour de l'état : x_{t|t} = x_{t|t-1} + K_t * e_t
        self.x = x_pred + K * error
        
        # Mise à jour de la covariance : P_{t|t} = (I - K_t * H) * P_{t|t-1}
        I = np.eye(2)
        self.P = np.dot((I - np.dot(K, H)), P_pred)

    def get_alpha(self):
        """Retourne l'estimation actuelle de l'intercepte (Alpha)."""
        return self.x[0, 0]

    def get_beta(self):
        """Retourne l'estimation actuelle de la pente (Beta / Hedge Ratio)."""
        return self.x[1, 0]