import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss
import warnings

# On ignore les warnings de statsmodels (notamment pour le test KPSS 
# qui avertit quand la p-value est hors de ses tables pré-calculées)
warnings.filterwarnings("ignore")

def test_adf(series, significance_level=0.05):
    """
    Test de Dickey-Fuller Augmenté (ADF).
    Évalue si la série temporelle possède une racine unitaire.
    
    H0 : La série a une racine unitaire (Marche aléatoire, NON-stationnaire).
    H1 : La série n'a pas de racine unitaire (Stationnaire).
    """
    series = series.dropna()
    if len(series) < 30:
        return False, 1.0
        
    # L'argument autolag='AIC' choisit automatiquement le meilleur 
    # nombre de retards (lags) p dans l'équation de la différence.
    result = adfuller(series, autolag='AIC')
    p_value = result[1]
    
    # On veut REJETER H0 pour prouver la stationnarité
    is_stationary = p_value < significance_level
    return is_stationary, p_value

def test_kpss(series, significance_level=0.05):
    """
    Test KPSS (Kwiatkowski-Phillips-Schmidt-Shin).
    Évalue la stationnarité autour d'une constante (trend deterministe).
    
    H0 : La série est stationnaire.
    H1 : La série a une racine unitaire (NON-stationnaire).
    """
    series = series.dropna()
    if len(series) < 30:
        return False, 0.0
        
    result = kpss(series, regression='c', nlags='auto')
    p_value = result[1]
    
    # On veut ÉCHOUER À REJETER H0 pour valider la stationnarité
    is_stationary = p_value >= significance_level
    return is_stationary, p_value

def is_strictly_stationary(series):
    """
    Validation Institutionnelle : Exige la confirmation croisée des deux tests.
    C'est la garantie mathématique qu'un processus de Mean-Reversion existe.
    """
    adf_stat, adf_p = test_adf(series)
    kpss_stat, kpss_p = test_kpss(series)
    
    # Logique de confirmation stricte
    if adf_stat and kpss_stat:
        status = "Strictement Stationnaire (Mean Reverting Validé)"
        is_valid = True
    elif adf_stat and not kpss_stat:
        status = "Stationnarité Différentielle (Instable)"
        is_valid = False
    elif not adf_stat and kpss_stat:
        status = "Non-Stationnaire (Dérive bornée)"
        is_valid = False
    else:
        status = "Marche Aléatoire Pure (Danger)"
        is_valid = False
        
    return is_valid, status, adf_p, kpss_p

# --- TEST INDÉPENDANT DU FICHIER ---
if __name__ == "__main__":
    import sys
    import os
    # Permet de lancer le fichier directement depuis la racine du projet
    sys.path.append(os.getcwd()) 
    from core.database_manager import QuantDatabase
    from quant_maths.kalman import KalmanRegression
    
    print("--- LABORATOIRE ÉCONOMÉTRIQUE : TEST DE STATIONNARITÉ ---")
    
    db = QuantDatabase()
    df_y = db.get_ticker_data('KO')
    df_x = db.get_ticker_data('PEP')
    
    if df_y is not None and df_x is not None:
        df = pd.DataFrame({'KO': df_y, 'PEP': df_x}).dropna()
        
        # On calcule le spread avec notre Kalman bivarié
        kalman = KalmanRegression(delta=1e-6, R=1e-3)
        spreads = []
        for i in range(len(df)):
            y, x = df['KO'].iloc[i], df['PEP'].iloc[i]
            kalman.update(x, y)
            spread = y - (kalman.get_beta() * x + kalman.get_alpha())
            spreads.append(spread)
            
        spread_series = pd.Series(spreads).dropna()
        
        # On ne teste que la période récente (ex: les 2 dernières années)
        recent_spread = spread_series.tail(504)
        
        is_valid, status, p_adf, p_kpss = is_strictly_stationary(recent_spread)
        
        print(f"\nDiagnostic du Spread KO/PEP (504 derniers jours) :")
        print(f"- P-Value ADF  : {p_adf:.4f} (< 0.05 attendu)")
        print(f"- P-Value KPSS : {p_kpss:.4f} (> 0.05 attendu)")
        print(f"-> Résultat : {status}")