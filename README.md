# XAUUSD AI Model — v3.1 (liquidity-first, leak-free)

Modèle de recherche qui prédit, à la clôture d'une session, ce que fera la session suivante :

1. **P(sweep du PDH)** : la prochaine session casse-t-elle le plus haut de la session ?
2. **P(sweep du PDL)** : casse-t-elle le plus bas ?
3. **Excursions haut / bas** en ATR (médiane + quantile 80 %), converties en niveaux de prix.

La direction 3 classes (Bear/Range/Bull) n'est gardée que comme diagnostic : elle n'a aucun pouvoir prédictif.

> Outil de recherche, pas un signal de trading. Aucune simulation d'exécution (spread, slippage) n'est incluse.

## Résultats (walk-forward strict, données H4 publiques 2013 → fév. 2026)

Réentraînement chaque année sur tout le passé. Le paramètre du modèle a été choisi sur 2017–2021 uniquement ; **2022 → 2026 est un holdout gelé**.

| Holdout 2022–2026 (1 057 sessions) | PDH sweep | PDL sweep |
|---|---|---|
| AUC | **0.819** | **0.787** |
| Accuracy (taux de base) | 73.5 % (53 %) | 72.5 % (44 %) |
| Brier skill vs climatologie | +0.31 | +0.24 |
| Quand p ≥ 0.70 : taux de réussite / % des jours | 84 % / 32 % | 79 % / 20 % |

Tests sur d'autres fournisseurs, avec un modèle entraîné uniquement sur les données H4 :

* MT5 M15 de janv. 2025 à mai 2026 (entraînement jusqu'à fin 2024) : AUC 0.836 (PDH) / 0.749 (PDL).
* M15 UTC d'avril à sept. 2026 : AUC 0.791 / 0.795.

Ce qui **ne marche pas** (mesuré, voir `reports/`) :

* **Excursions (médiane)** : pas mieux qu'une constante (MAE skill ≈ 0). Le quantile 80 % est bien calibré (couverture 75–80 %). Utilise-le comme enveloppe de volatilité, pas comme cible.
* **GBM** : moins bon que la régression logistique. Le blend fait jeu égal.
* **Macro (US10Y, US02Y, VIX)** : ±0.003 AUC, c'est du bruit.
* **Transformer** : AUC 0.807 contre 0.817 pour la logistique sur le même test.
* **Direction 3 classes** : 34.7 % de réussite, moins que la classe majoritaire (40.8 %).

## Démarrage rapide

```bash
pip install -r requirements.txt

# 1) Données publiques de dev (ou tes propres CSV)
python public_data_bootstrap.py
python prepare_public_data.py
python build_macro_features.py --fred data/fred_sample.csv --out data/macro.csv   # optionnel

# 2) Contrôle qualité : fuseau, sessions, trous, spikes
python validate_data.py --xauusd data/xauusd_h4_2013_2026.csv --session-tz UTC --session-start 21:00

# 3) Test anti-fuite
python -m pytest tests -q
python tests/test_no_leak.py --xauusd data/xauusd_h4_2013_2026.csv --session-tz UTC --session-start 21:00

# 4) Walk-forward avec holdout gelé
python walk_forward_multitask.py --xauusd data/xauusd_h4_2013_2026.csv \
    --session-tz UTC --session-start 21:00 --holdout-start 2022-01-01 --out reports/wf_h4

# 5) Modèle final + prédiction de la prochaine session
python train_multitask.py  --xauusd data/xauusd_h4_2013_2026.csv --session-tz UTC --session-start 21:00
python predict_multitask.py --xauusd data/xauusd_h4_2013_2026.csv

# 6) Tester un modèle sauvegardé sur un autre fournisseur ou une autre période
python evaluate_model.py --model models/xauusd_multitask.joblib --xauusd data/xauusd_m15_utc_2026.csv
```

## Réglages des données (important)

| Option | Défaut | Rôle |
|---|---|---|
| `--ts-convention` | `open` | Le timestamp est l'heure d'**ouverture** de la bougie (MT5, Dukascopy, HistData, TwelveData, OANDA). Mets `close` si ton export date la clôture. |
| `--data-tz` | `UTC` | Fuseau des timestamps sans offset. **MT4/MT5 = heure serveur** : la plupart des brokers or utilisent `NY+7` (minuit serveur = 17:00 New York, suit le DST US). Sinon un fuseau IANA (`Europe/Athens`...). |
| `--session-tz` / `--session-start` | `America/New_York` / `17:00` | Définition de la bougie journalière. Le dimanche soir est rattaché au lundi. |

`validate_data.py` signale un fuseau probablement faux (pause quotidienne mal placée, sessions courtes).

## Structure

| Fichier | Rôle |
|---|---|
| `config.py` | Tous les réglages (sessions, labels, modèle) |
| `data_pipeline.py` | Chargement, fuseaux, horloge de session, agrégation, jointure macro « as-of » |
| `features.py` | Features sans unité de prix (ATR, %, ratios) et labels |
| `multitask_models.py` | Logit / GBM / blend, régressions médiane + quantile |
| `walk_forward_multitask.py` | Test principal : baselines, calibration, découpage par régime, holdout |
| `train_multitask.py` / `predict_multitask.py` | Entraînement final / prédiction avec niveaux de prix |
| `evaluate_model.py` | Score d'un modèle sur un autre dataset |
| `validate_data.py` | Qualité des données |
| `tests/test_no_leak.py` | Test de troncature : les features du jour D ne changent pas si on coupe les données après D |
| `train_baseline.py`, `walk_forward.py`, `predict_next_day.py` | Direction 3 classes (diagnostic v1) |
| `train_transformer.py` | Modèle séquentiel optionnel, comparé à la logistique sur un test gelé |

## Règle

On ne garde une source de données ou un modèle que s'il bat `core3` (logistique à 3 variables) **et** la version actuelle sur le walk-forward, sans dégrader le holdout. Le rapport affiche `beats_core3` pour chaque cible.

Comparaison avec la variante v4 « session fixed » : `COMPARISON_v4.md`. Voir `CHANGELOG.md` pour la liste des corrections et `RESEARCH_RESULTS_2026-09-29.md` pour l'historique.
