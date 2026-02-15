import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from core.database_manager import QuantDatabase

class FinancialAsset:
    """
    Cette classe représente un actif financier (Action, ETF, Forex).
    Elle encapsule les données et les calculs mathématiques de base.
    """
    def __init__(self, ticker, start_date, end_date):
        self.ticker = ticker
        self.start = start_date
        self.end = end_date
        self.data = None
        self.daily_returns = None
        self.log_returns = None
        # On instancie la base de données
        self.db = QuantDatabase()

    def download_data(self, force_update=False):
        """
        Stratégie Intelligente : Local First, Network Second.
        Si force_update=True, on force le téléchargement Yahoo.
        """
        # 1. Essai de lecture locale (Si pas force_update)
        if not force_update:
            print(f"[{self.ticker}] Recherche dans la base locale...")
            local_data = self.db.get_ticker_data(self.ticker)
            
            if local_data is not None and not local_data.empty:
                # On vérifie si les dates correspondent (approximativement)
                # Si la dernière date locale est < à self.end, il faudrait mettre à jour (on verra plus tard)
                print(f"[{self.ticker}] Chargé depuis le disque (SQL).")
                self.data = local_data
                self.clean_data()
                return # On s'arrête là, pas besoin d'internet

        # 2. Si on est ici, c'est qu'on n'a pas de données ou qu'on force la mise à jour
        print(f"[{self.ticker}] Téléchargement Yahoo Finance...")
        try:
            df = yf.download(self.ticker, start=self.start, end=self.end, auto_adjust=True, progress=False)
            
            # Gestion MultiIndex (Ton correctif)
            if isinstance(df.columns, pd.MultiIndex):
                df = df['Close']
            if isinstance(df, pd.DataFrame) and df.shape[1] == 1:
                df = df.iloc[:, 0]

            self.data = df
            self.clean_data()
            
            # 3. SAUVEGARDE EN BASE pour la prochaine fois
            print(f"[{self.ticker}] Sauvegarde en base SQL...")
            self.db.save_ticker_data(self.ticker, self.data)
            
        except Exception as e:
            print(f"ERREUR CRITIQUE sur {self.ticker} : {e}")

    def clean_data(self):
        """Nettoyage des données (Ingénierie)"""
        # Suppression des NaNs (Jours fériés, bugs)
        self.data.dropna(inplace=True)
        # On pourrait ajouter ici : détection des outliers (prix aberrants)
        
    def compute_returns(self):
        """Calcule les rendements Arithmétiques ET Logarithmiques."""
        if self.data is None:
            raise ValueError("Données non chargées. Lancez download_data() d'abord.")
            
        # 1. Rendements Simples (Arithmétiques) : R_t = (P_t / P_{t-1}) - 1
        self.daily_returns = self.data.pct_change().dropna()
        
        # 2. Log Rendements (Géométriques) : r_t = ln(P_t / P_{t-1})
        # MATHS : np.log est le logarithme népérien
        self.log_returns = np.log(self.data / self.data.shift(1)).dropna()

    def get_volatility(self, annualized=True):
        """Retourne la volatilité (Standard Deviation)"""
        if self.log_returns is None:
            self.compute_returns()
            
        vol = self.log_returns.std()
        if annualized:
            vol = vol * np.sqrt(252) # Annualisation (Racine du temps)
        return vol

    def plot_analysis(self):
        """Affiche un dashboard complet de l'actif"""
        if self.log_returns is None:
            self.compute_returns()
            
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # Prix
        ax1.plot(self.data, color='navy')
        ax1.set_title(f"Prix de {self.ticker}")
        ax1.grid(True)
        
        # Distribution des Rendements (Histogramme)
        # C'est ici qu'on vérifie la "Normalité"
        ax2.hist(self.log_returns, bins=50, color='orange', alpha=0.7, density=True)
        ax2.set_title(f"Distribution des Log-Rendements (Volatilité: {self.get_volatility():.2%})")
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()


    def diagnose(self):
        """
        Effectue une analyse statistique complète (Moments 3 & 4 + Test de Normalité).
        Affiche un rapport et un graphique de distribution.
        """
        if self.log_returns is None:
            self.compute_returns()
            
        series = self.log_returns
        
        # 1. Calcul des Moments
        mu = series.mean()
        sigma = series.std()
        skew = series.skew()
        kurt = series.kurtosis() # Pandas donne l'Excess Kurtosis (Normal = 0)
        
        # 2. Test de Jarque-Bera (Test de Normalité)
        # H0 : La distribution est Normale (Gaussienne)
        jb_stat, p_value = stats.jarque_bera(series)
        is_normal = p_value > 0.05
        
        # 3. Affichage du Rapport "Ingénieur"
        print(f"\n--- DIAGNOSTIC STATISTIQUE : {self.ticker} ---")
        print(f"Période analysée : {len(series)} jours")
        print(f"1. Moyenne (Drift)    : {mu:.6f} (proche de 0 ?)")
        print(f"2. Volatilité (Sigma) : {sigma:.6f}")
        print(f"3. Skewness (Biais)   : {skew:.4f} " + ("(Danger: Négatif)" if skew < -0.5 else "(OK)"))
        print(f"4. Kurtosis (Queues)  : {kurt:.4f} " + ("(Danger: Fat Tails)" if kurt > 2.0 else "(Normal)"))
        print("-" * 30)
        print(f"Test Jarque-Bera      : p-value = {p_value:.4e}")
        if is_normal:
            print(">> CONCLUSION : Distribution NORMALE (Modèles classiques valides).")
        else:
            print(">> CONCLUSION : Distribution ANORMALE (Risque extrême sous-estimé).")
            print(">> Les modèles Gaussiens (Sharpe, Black-Scholes) doivent être ajustés.")

        # 4. Visualisation Avancée (QQ-Plot + Histogramme)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Histogramme vs Loi Normale Théorique
        count, bins, ignored = ax1.hist(series, 50, density=True, alpha=0.6, color='blue', label='Réel')
        # Courbe théorique
        plt_x = np.linspace(min(bins), max(bins), 100)
        plt_y = stats.norm.pdf(plt_x, mu, sigma)
        ax1.plot(plt_x, plt_y, linewidth=2, color='r', label='Loi Normale Théorique')
        ax1.set_title(f"Distribution des Rendements ({self.ticker})")
        ax1.legend()
        ax1.grid(True)
        
        # QQ-Plot (Quantile-Quantile Plot)
        # Si les points bleus suivent la ligne rouge, c'est normal.
        # S'ils s'écartent aux extrémités, c'est des Fat Tails.
        stats.probplot(series, dist="norm", plot=ax2)
        ax2.set_title("QQ-Plot (Détection des Fat Tails)")
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()


