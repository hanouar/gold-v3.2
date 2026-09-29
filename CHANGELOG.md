# v3.2 — 2026-09-29

* Ajout de l'horloge broker `--data-tz NY+7` (repris de la variante v4) : minuit serveur = 17:00 New York, avec le DST américain. Sur le fichier MT5 public, l'ancien réglage `Europe/Athens` décalait d'1 h environ 10 % des bougies (mars et octobre-novembre, quand les DST européen et américain divergent). AUC sur ce fichier : 0.833 → 0.836 (PDH), 0.747 → 0.749 (PDL).
* Nouveau test unitaire pour cette horloge.
* `COMPARISON_v4.md` : benchmark v4 contre v3.1 avec le même protocole.

# v3.1 — corrections (2026-09-29)

## Bugs critiques

1. **Fuite de données H4/H1/M15** (`features.py`) : `f.resample("1D").last()` était décalé d'un jour par rapport aux barres D1 (`label="right"`). Chaque ligne voyait ~20 h de la session à prédire.
   * Effet mesuré : AUC 0.93 (faux) → 0.81 (réel).
   * Correction : toutes les timeframes sont agrégées sur la même horloge de session. Un test de troncature (`tests/test_no_leak.py`) le vérifie automatiquement.
2. **Journées fantômes du dimanche** : avec un découpage à minuit UTC, les ~2 h du dimanche soir formaient une « journée ». Le lundi la balayait 70–77 % du temps, ce qui gonflait les scores.
   * Correction : sessions définies de 17:00 New York à 17:00 New York (configurable), le dimanche soir étant rattaché au lundi.
3. **Décalage entraînement / live** : avec la fuite, le modèle était entraîné sur des features que le live n'a jamais. `predict_multitask.py` refuse aussi une session non terminée (sauf `--allow-partial`).
4. **Convention des timestamps** : le code supposait des timestamps de clôture alors que MT5 et Dukascopy exportent l'ouverture. Ajout de `--ts-convention open|close` (défaut `open`).
5. **Fuseau MT5** : l'heure serveur est gérée via `--data-tz`. `validate_data.py` détecte un fuseau incohérent.
6. **Macro** :
   * La jointure se fait sur l'heure de fin de session, colonne par colonne.
   * Tolérance de 7 jours : plus de VIX figé pendant 1 an et demi par un ffill.
   * Les valeurs FRED sont datées « observation + 36 h », c'est-à-dire quand elles sont vraiment publiées.
   * Les variations de taux se calculent en différences, pas en %.

## Améliorations du modèle

7. **Features sans unité de prix** : suppression de l'ATR et de l'ADR en dollars, qui ne généralisaient pas entre l'or à 1 200 $ et à 4 000 $. Ajouts :
   * position de la clôture dans le range (clv), distance clôture → haut/bas en ATR ;
   * sweep et rejet du jour, fréquence de sweep sur 5/20 jours ;
   * inside/outside day, ratio ATR5/ATR20, jour de la semaine suivante.
8. **Régression logistique régularisée par défaut** (C = 0.01, choisi sur 2017–2021 seulement).
   * Elle bat le GBM, qui reste disponible avec `--classifier gbm|blend`.
   * Le GBM est maintenant calibré sur des plis temporels (`TimeSeriesSplit`) au lieu de plis mélangés.
9. **Excursions** : perte `absolute_error` (médiane) au lieu de `squared_error`, plus un quantile 80 %. `predict_multitask.py` les convertit en niveaux de prix.
10. **Nouveaux labels de recherche** : `y_pdh_reject` / `y_pdl_reject` (sweep puis clôture de retour dans le range) et `add_sweep_order_labels()` (quel côté est pris en premier).

## Évaluation

11. `walk_forward_multitask.py` :
    * baselines climatologie et `core3` entraînées de la même façon ;
    * Brier skill, tables de calibration, taux de réussite quand p ≥ 0.70 ;
    * découpage par régime de volatilité et par jour ;
    * `--holdout-start` pour une période gelée ;
    * `predictions.csv` jour par jour.
12. `evaluate_model.py` : score d'un modèle sauvegardé sur un autre fournisseur ou une autre période.
13. `train_transformer.py` : découpage train / validation (early stopping) / test gelé, avec la logistique comme référence sur les mêmes lignes. Avant, le même jeu servait à l'early stopping et au score.
14. `walk_forward.py` (direction) : ajout de la référence « classe majoritaire ».
15. `build_macro_features.py` / `fetch_web_data.py` : horodatage « disponible à » au lieu de la date d'observation.
16. `prepare_public_data.py` : chaîne de préparation reproductible des données publiques.
