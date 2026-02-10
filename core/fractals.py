import numpy as np
import pandas as pd

def calculate_hurst_exponent(time_series, max_lag=20):
    """
    Calcule l'Exposant de Hurst (H) d'une série temporelle via la méthode R/S simplifiée.
    
    H est une mesure de la "mémoire longue" de la série.
    
    Interprétation :
    - H < 0.5 : Mean Reverting (Anti-persistant). La série veut revenir à sa moyenne. (IDÉAL pour nous)
    - H = 0.5 : Marche Aléatoire (Brownien). Imprévisible.
    - H > 0.5 : Trending (Persistant). Si ça monte, ça continue de monter. (DANGEREUX pour le Pairs Trading)
    """
    # Conversion en tableau numpy pour la performance
    series = np.array(time_series)
    
    # On ne peut pas calculer Hurst sur une série trop courte
    if len(series) < max_lag:
        return 0.5

    lags = range(2, max_lag)
    
    # Calcul de la volatilité des différences pour différents retards (lags)
    # Formule simplifiée de la méthode Variance-Ratio
    tau = [np.sqrt(np.std(np.subtract(series[lag:], series[:-lag]))) for lag in lags]
    
    # On utilise une régression linéaire sur le log-log pour trouver la pente
    # La relation est : Log(Var) ~ 2H * Log(lag)
    
    # Y = mX + c
    Y = np.log(tau)
    X = np.log(lags)
    
    # La pente (m) de la régression correspond à l'exposant de Hurst
    try:
        poly = np.polyfit(X, Y, 1)
        H = poly[0] # Dans cette implémentation simplifiée via std dev, la pente est H directement (parfois H*2 selon la méthode exacte R/S vs DMA)
    except:
        return 0.5 # Valeur par défaut en cas d'erreur mathématique (neutralité)
    
    return H

def check_mean_reversion_quality(series):
    """
    Fonction utilitaire pour diagnostiquer une série et afficher un rapport.
    """
    H = calculate_hurst_exponent(series)
    print(f"--- ANALYSE FRACTALE (Hurst) ---")
    print(f"Exposant de Hurst : {H:.4f}")
    
    if H < 0.4:
        print(">> EXCELLENT : Forte force de rappel (Mean Reversion).")
        return True
    elif H < 0.5:
        print(">> BON : Tendance au retour à la moyenne.")
        return True
    elif H == 0.5:
        print(">> NEUTRE : Marche aléatoire pure.")
        return False
    else:
        print(">> DANGER : La série est en tendance (Trending). Ne pas faire de Mean Reversion !")
        return False