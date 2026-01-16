# QuantEngine : Algorithmic Trading Infrastructure

## Description
Projet d'ingénierie financière visant à construire un moteur de backtesting et d'analyse quantitative modulaire en Python.
Ce projet explore la gestion de données financières (SQL), l'analyse statistique avancée (Jarque-Bera, Cointégration) et l'optimisation de portefeuille (Markowitz, Kelly Criterion).

## Fonctionnalités Actuelles
- **Architecture OOP :** Gestion modulaire via classes `Asset` et `Portfolio`.
- **Data Engineering :** Système de cache local (SQLite) pour limiter les appels API et garantir la persistance.
- **Analyses Statistiques :** Détection de non-normalité (Fat Tails), Tests de Stationnarité (ADF).
- **Stratégies Implémentées :** - Pairs Trading (Cointégration / Mean Reversion).
  - Allocation Diversifiée (Actions / Obligations / Or).

## Stack Technique
- **Langage :** Python 3.11
- **Analyse :** Pandas, NumPy, Scipy, Statsmodels
- **Données :** Yfinance, SQLite
- **Visualisation :** Matplotlib, Seaborn

## Prochaines Étapes (Roadmap)
- Implémentation d'un Backtester "Event-Driven" (Simulation réaliste).
- Ajout de modèles de Machine Learning pour la prédiction de tendance.
