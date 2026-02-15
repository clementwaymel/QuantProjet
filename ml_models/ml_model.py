import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

class MLPredictor:
    """
    Wrapper pour gérer l'entraînement et la prédiction d'un modèle Random Forest.
    """
    def __init__(self):
        # On initialise un Random Forest
        # n_estimators=100 : 100 arbres de décision
        # min_samples_leaf=10 : On empêche les arbres d'apprendre par cœur (Overfitting)
        # random_state=42 : Pour que les résultats soient reproductibles
        self.model = RandomForestClassifier(n_estimators=100, min_samples_leaf=10, random_state=42)
        
    def prepare_data(self, df_features):
        """
        Sépare les données en X (Facteurs) et y (Cible).
        Supprime les colonnes inutiles (Prix, Date, etc.)
        """
        # Liste des colonnes qu'on donne à manger à l'IA
        feature_cols = ['log_ret', 'volatility', 'rsi_norm', 'macd', 'z_score_price']
        
        # Vérification que les colonnes existent
        available_cols = [c for c in feature_cols if c in df_features.columns]
        
        X = df_features[available_cols]
        y = df_features['Target_Direction'] # 0 ou 1
        
        return X, y

    def train(self, X_train, y_train):
        """Entraîne le modèle sur le passé."""
        print(f"[IA] Entraînement sur {len(X_train)} bougies...")
        self.model.fit(X_train, y_train)
        print("[IA] Entraînement terminé.")

    def predict(self, X_test):
        """Fait des prédictions sur de nouvelles données."""
        return self.model.predict(X_test)
    
    def predict_proba(self, X_test):
        """Donne la probabilité (confiance) de la prédiction (ex: 0.55 de chance que ça monte)."""
        return self.model.predict_proba(X_test)

    def evaluate(self, y_true, y_pred):
        """Affiche le bulletin de notes du modèle."""
        acc = accuracy_score(y_true, y_pred)
        print(f"\n--- RÉSULTATS DU MODÈLE ---")
        print(f"Précision (Accuracy) : {acc:.2%}")
        print("-" * 30)
        print("Matrice de Confusion :")
        print(confusion_matrix(y_true, y_pred))
        print("-" * 30)
        print("Rapport détaillé :")
        print(classification_report(y_true, y_pred))
        return acc