import numpy as np

class KalmanRegression:
    """
    Filtre de Kalman 1D pour estimer dynamiquement la pente (Beta) d'une régression.
    Modèle : y_t = beta_t * x_t + epsilon_t
    Contrairement au Rolling OLS, il n'a pas de fenêtre fixe et s'adapte instantanément.
    """
    def __init__(self, delta=1e-4, R=1e-3):
        # --- PARAMÈTRES (TUNING) ---
        # delta (Q) : La "nervosité" du Beta. 
        # Si delta est grand, le modèle réagit vite (mais bruité).
        # Si delta est petit, le modèle est stable (mais lent).
        self.delta = delta
        
        # R : Le bruit de mesure (Variance du marché).
        self.R = R

        # --- ÉTAT DU SYSTÈME ---
        # x : Le vecteur d'état (Ici, c'est juste le Beta, donc taille 1x1)
        # On l'initialise à 0
        self.x = np.zeros((1, 1)) 
        
        # P : La Matrice de Covariance de l'erreur (Notre incertitude sur Beta)
        # On commence avec une incertitude énorme (Identité)
        self.P = np.ones((1, 1)) 

    def update(self, x_meas, y_meas):
        """
        Met à jour l'estimation du Beta avec une nouvelle observation (prix X et Y).
        Applique les équations récursives du filtre de Kalman.
        """
        # Transformation en matrices numpy pour l'algèbre linéaire
        # Equation d'observation : y = H * beta + bruit
        # Donc H (Observation Matrix) est x_meas
        H = np.array([[x_meas]])
        
        # --- 1. PRÉDICTION (Time Update) ---
        # On parie que le Beta n'a pas changé depuis hier (Random Walk)
        # x_pred = x_prev
        x_pred = self.x
        
        # L'incertitude augmente légèrement avec le temps (Process Noise Q)
        # Q = delta / (1 - delta) * I  (Formule classique pour régression adaptative)
        # Ou plus simplement Q = delta * I
        Q = np.ones((1, 1)) * self.delta
        P_pred = self.P + Q
        
        # --- 2. MISE À JOUR (Measurement Update) ---
        # Calcul de l'erreur de prédiction (Innovation)
        # y_hat = y_reel - (Beta_pred * x_reel)
        error = y_meas - np.dot(H, x_pred)
        
        # Variance de l'innovation (S)
        S = np.dot(H, np.dot(P_pred, H.T)) + self.R
        
        # Calcul du Gain de Kalman (K)
        # K = P * H.T * inv(S)
        # C'est le "Poids" qu'on donne à la nouvelle information
        K = np.dot(P_pred, np.dot(H.T, np.linalg.inv(S)))
        
        # Mise à jour de l'état (Beta)
        # Beta_new = Beta_old + K * Erreur
        self.x = x_pred + np.dot(K, error)
        
        # Mise à jour de l'incertitude (P)
        # P_new = (I - K * H) * P_pred
        I = np.eye(self.x.shape[0])
        self.P = np.dot((I - np.dot(K, H)), P_pred)

    def get_beta(self):
        """Retourne l'estimation actuelle du Hedge Ratio"""
        return self.x[0, 0]