import numpy as np
import pandas as pd
from scipy.optimize import minimize

class MarkowitzOptimizer:
    """
    Calculateur d'allocation optimale (Minimum Variance Portfolio).
    Cherche les poids w qui minimisent w^T * Cov * w.
    """
    def __init__(self, lookback_days=60):
        self.lookback_days = lookback_days

    def get_optimal_weights(self, prices_df):
        """
        Calcule les poids optimaux basés sur l'historique récent des prix.
        Input: DataFrame des prix (colonnes = tickers, index = dates)
        Output: Dictionnaire {ticker: poids}
        """
        # 1. Calcul des rendements sur la fenêtre glissante
        # On ne regarde que les N derniers jours
        recent_prices = prices_df.tail(self.lookback_days)
        returns = recent_prices.pct_change().dropna()
        
        if len(returns) < 10:
            # Pas assez de données, on fait une allocation égalitaire (1/N)
            n = len(prices_df.columns)
            return {col: 1.0/n for col in prices_df.columns}

        # 2. Matrice de Covariance (Le cœur du risque)
        cov_matrix = returns.cov() * 252 # Annualisée
        
        # 3. Fonction à minimiser (La Variance du Portefeuille)
        # Var = w.T * Sigma * w
        def portfolio_variance(weights, cov_mat):
            return np.dot(weights.T, np.dot(cov_mat, weights))

        # 4. Contraintes et Bornes
        num_assets = len(prices_df.columns)
        initial_weights = np.array([1.0 / num_assets] * num_assets) # Départ équiréparti
        
        # Contrainte : La somme des poids doit faire 1 (100% du capital)
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        
        # Bornes : Pas de short selling dans l'allocation long terme (0 <= w <= 1)
        # On peut changer (0, 1) en (-1, 1) si on veut autoriser le short
        bounds = tuple((0.0, 1.0) for _ in range(num_assets))

        # 5. Optimisation (Solveur SLSQP)
        result = minimize(
            portfolio_variance, 
            initial_weights, 
            args=(cov_matrix,), 
            method='SLSQP', 
            bounds=bounds, 
            constraints=constraints
        )

        if not result.success:
            print(f"[Optimiseur] Échec de l'optimisation : {result.message}")
            return {col: 1.0/num_assets for col in prices_df.columns}

        # 6. Formatage du résultat
        optimal_weights = result.x
        return {ticker: round(weight, 4) for ticker, weight in zip(prices_df.columns, optimal_weights)}