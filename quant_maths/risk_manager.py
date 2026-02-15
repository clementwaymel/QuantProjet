import numpy as np
import pandas as pd

class RiskManager:
    """
    Module de contrôle du risque global.
    Ajuste l'exposition (Levier) pour maintenir une volatilité constante.
    """
    def __init__(self, target_vol=0.15, max_leverage=3.0):
        # target_vol = 0.15 (15% par an, comme le S&P500)
        # max_leverage = 3.0 (Sécurité : on n'emprunte pas plus de 3x le capital)
        self.target_vol = target_vol
        self.max_leverage = max_leverage
        
        # Historique des rendements journaliers du portefeuille
        # Nécessaire pour calculer la volatilité réalisée
        self.returns_history = []
        self.last_equity = None

    def update(self, current_equity):
        """
        À appeler chaque jour pour mettre à jour l'historique des rendements.
        """
        if self.last_equity is not None:
            # Calcul du rendement journalier du portefeuille
            if self.last_equity > 0:
                ret = (current_equity - self.last_equity) / self.last_equity
                self.returns_history.append(ret)
        
        self.last_equity = current_equity

    def get_leverage_factor(self):
        """
        Calcule le multiplicateur de levier à appliquer.
        Formule : Target_Vol / Realized_Vol
        """
        # On a besoin d'au moins 20 jours pour estimer une volatilité fiable
        if len(self.returns_history) < 20:
            return 1.0 # Pas de levier au début
            
        # Calcul de la volatilité annualisée récente (20 derniers jours)
        recent_returns = self.returns_history[-20:]
        std_dev = np.std(recent_returns)
        
        # Si la vol est nulle (cas rare), on reste à 1
        if std_dev == 0:
            return 1.0
            
        # Annualisation (Racine de 252)
        annualized_vol = std_dev * np.sqrt(252)
        
        # Calcul du Ratio de Levier
        # Si ma vol actuelle est 5% et je vise 15% -> Levier = 3.0
        leverage = self.target_vol / annualized_vol
        
        # Plafond de sécurité (Cap)
        leverage = min(leverage, self.max_leverage)
        
        # Plancher (on ne descend pas en dessous de 0.1)
        leverage = max(leverage, 0.1)
        
        return leverage