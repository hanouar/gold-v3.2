# Comparaison : v4 « session fixed » vs v3.1 / v3.2

Protocole identique pour les deux versions (script : `tests/bench_compare.py`) :

* mêmes fichiers de données ;
* même walk-forward annuel avec réentraînement sur tout le passé ;
* développement 2017–2021, holdout gelé 2022 → fév. 2026 ;
* tests sur d'autres fournisseurs avec un modèle entraîné uniquement sur les données H4.

Chaque version construit son propre dataset et utilise son modèle par défaut :

* **v4** : GBM calibré, 75 features ;
* **v3.1** : régression logistique, 59 features.

`core3` est la régression à 3 variables, entraînée sur le dataset de chaque version. Elle donne la même AUC des deux côtés : les sessions et les labels sont donc équivalents, et l'écart vient du modèle et des features.

Exécution sous pandas 2.2 (voir le bug pandas 3 plus bas).

## AUC (plus haut = mieux)

| Jeu de test | Cible | v4 | v3.1 | core3 |
|---|---|---|---|---|
| Dev 2017–2021 (1 290 sessions) | PDH | 0.807 | **0.829** | 0.823 |
| | PDL | 0.772 | **0.791** | 0.788 |
| Holdout 2022–2026 (1 057) | PDH | 0.795 | **0.819** | 0.814 |
| | PDL | 0.765 | **0.787** | 0.783 |
| MT5 M15 2025–mai 2026 (~335) | PDH | 0.795 | **0.833** | 0.81 |
| | PDL | 0.720 | **0.747** | 0.73 |
| M15 UTC avr.–sept. 2026 (~115) | PDH | 0.751 | **0.791** | 0.785 |
| | PDL | 0.776 | **0.795** | 0.785 |

Brier skill sur le holdout : v4 +0.26 / +0.21, contre v3.1 +0.31 / +0.24.

Erreur moyenne des excursions sur le holdout (haut / bas) : v4 0.352 / 0.371, v3.1 0.335 / 0.317. Une prédiction constante (médiane) fait 0.336 / 0.321 : v4 fait moins bien qu'une constante.

Temps du walk-forward : v4 109 s, v3.1 40 s.

**Le modèle v4 fait moins bien que la régression à 3 variables sur ses propres données, sur tous les jeux de test.**

## Audit du code v4

| Point | v4 | v3.1 |
|---|---|---|
| Fuite de données (test de troncature) | Aucune ✔ | Aucune ✔ |
| Sessions à 17:00 New York avec DST | ✔ (pandas 2.x seulement) | ✔ |
| **pandas 3.x** | ✘ `offset` ignoré en silence : sessions à minuit NY, journées du dimanche de retour, 2 tests sur 3 échouent. Or `requirements.txt` accepte pandas ≥ 2.0. | ✔ testé sous 2.2 et 3.0 |
| Horloge broker NY+7 | ✔ bonne idée | ✘ → **repris dans v3.2** |
| Alignement des bougies H4 sur la session | ✘ 1 h de décalage en heure d'été : les H4 enjambent 17:00 | ✔ |
| Features H1/M15 calculées à partir de données H4 | ✘ copies trompeuses des H4 | ✔ ignorées |
| Features en niveau de prix ($) | ✘ ATR et ADR absolus | ✔ tout en ATR ou % |
| Modèle de classification | GBM | Régression logistique (choisie sur la période dev) |
| Excursions | Perte au carré, pire qu'une constante | Médiane + quantile 80 % calibré |
| Macro | Valeurs figées sans limite d'âge, variations des taux en % | Limite de 7 jours, différences |
| Prédiction sur une session non terminée | ✘ utilise la dernière ligne telle quelle | ✔ détectée et exclue |
| Walk-forward : références, holdout, calibration | ✘ | ✔ |

## Conclusion

1. Le **constat principal de v4 est juste** : des sessions à 17:00 New York au lieu de minuit UTC améliorent les résultats. v3.1 le faisait déjà.
2. Le **modèle v4 est moins bon** : −0.02 à −0.04 d'AUC selon le jeu de test, et il ne bat même pas la régression à 3 variables.
3. v4 a **cassé la compatibilité pandas 3**, sans erreur visible : les sessions deviennent fausses.
4. Seul apport à reprendre, **fait dans v3.2** : l'horloge broker **NY+7**. Sur le fichier MT5, `Europe/Athens` décalait ~10 % des bougies d'1 h en mars et en octobre-novembre.
