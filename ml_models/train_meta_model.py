import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import os

print("--- ENTRAÎNEMENT DU MODÈLE DE META-LABELING ---")

# 1. Chargement des données
data_path = "data/ml_dataset_ko_pep.csv"
if not os.path.exists(data_path):
    print(f"Erreur : Le fichier {data_path} est introuvable.")
    exit()
    
df = pd.read_csv(data_path, index_col='date', parse_dates=True)

# 2. Séparation Features (X) / Target (y)
feature_cols = ['volatility', 'rsi_norm', 'macd', 'z_score_price']
X = df[feature_cols]
y = df['Label']

# 3. Séparation Temporelle (Train / Test Split)
# PAS DE SHUFFLE ! On garde l'ordre chronologique (Sécurité mathématique absolue)
split_idx = int(len(df) * 0.70)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

print(f"Données d'entraînement (Le Passé) : {len(X_train)} signaux")
print(f"Données de test (Le Futur Inconnu) : {len(X_test)} signaux")

# 4. Le Modèle : Random Forest avec Pondération Mathématique
# class_weight='balanced' pénalise lourdement les erreurs sur les Fat Tails (Classe 0)
# max_depth=3 empêche le modèle d'apprendre par coeur le bruit du marché
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=3,           
    min_samples_leaf=5,
    class_weight='balanced', 
    random_state=42
)

# 5. Entraînement sur le Passé
model.fit(X_train, y_train)

# 6. Évaluation sur le Futur (Out-of-Sample) avec Seuil Personnalisé
print("\n--- RÉSULTATS OUT-OF-SAMPLE (SUR LE TEST) ---")

# Au lieu d'utiliser predict() qui coupe à 50%...
# On utilise predict_proba() qui renvoie les pourcentages [Probabilité_0, Probabilité_1]
y_proba = model.predict_proba(X_test)[:, 1] # On extrait la probabilité de succès (Classe 1)

# --- RECHERCHE DU SEUIL OPTIMAL ---
# On va tester plusieurs niveaux d'exigence pour voir comment l'IA réagit
seuils_a_tester = [0.50, 0.60, 0.65, 0.70]

for seuil in seuils_a_tester:
    # Si la probabilité dépasse le seuil, on valide (1), sinon on bloque (0)
    y_pred_custom = (y_proba >= seuil).astype(int)
    
    print(f"\n>> TEST AVEC SEUIL DE CONFIANCE : {seuil * 100}%")
    print(confusion_matrix(y_test, y_pred_custom))