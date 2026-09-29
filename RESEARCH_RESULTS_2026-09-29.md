# Research checkpoint — 2026-09-29

This file records the exploratory tests run while building v3. These are **research diagnostics**, not a production performance claim.

## Data assembled

- Long Gold daily history: 2001-06-03 onward from a public market-history repository.
- 2025–2026 broker daily Gold data used as a recent bridge.
- Public 15-minute Gold sample aggregated to daily to extend the test through 2026-09-25.
- Public mirrors of US 10Y, US 2Y and VIX were inspected; a weekly public macro panel supplied broad USD, real 10Y yield and S&P 500 context.

## Finding 1 — next-day 3-class direction is weak

Target: next-day close return in ATR units:

- Bearish <= -0.20 ATR
- Range between -0.20 and +0.20 ATR
- Bullish >= +0.20 ATR

A leak-aware expanding logistic baseline was close to random for three classes:

- Walk-forward 2014–2026 Daily-only accuracy: ~35.7%.
- A 2026 stitched out-of-sample test through late September produced ~34.1% accuracy.
- Adding a small set of macro features to a simple linear classifier did not improve the overall result in the exploratory 2021–2026 test.

Conclusion: do **not** make next-day Bull/Range/Bear the main objective and do not add deeper networks merely to force this label.

## Finding 2 — liquidity sweep targets are materially more learnable

The model was reframed around the daily decision tree:

- Will tomorrow trade above today's high? (`PDH sweep`)
- Will tomorrow trade below today's low? (`PDL sweep`)
- How far can upside/downside extend in ATR units?

Using a regularized binary baseline with technical features and training only on data available before the test year:

### Validation 2025

- PDH sweep: accuracy ~76.4%, AUC ~0.839.
- PDL sweep: accuracy ~67.8%, AUC ~0.758.

### Out-of-sample 2026 through 2026-09-25

- PDH sweep: accuracy ~67.3%, AUC ~0.756.
- PDL sweep: accuracy ~65.4%, AUC ~0.725.
- Realized next-day PDH sweep base rate: ~47.6%.
- Realized next-day PDL sweep base rate: ~48.6%.
- Mean upside excursion: ~0.49 ATR.
- Mean downside excursion: ~0.52 ATR.

This is substantially more promising than the 3-class close-direction target.

## Important caveats

1. The 2025/2026 price series is stitched across public datasets/providers. A production run should standardize one primary provider and verify overlapping bars.
2. The reported sweep model is still a baseline. It needs stricter walk-forward calibration, transaction/session logic and source-quality checks.
3. Macro releases with revisions (CPI/PCE, etc.) require point-in-time/ALFRED vintages before they enter a historical backtest.
4. Consensus forecasts and FedWatch probabilities need timestamped historical snapshots; current-vintage values must never be backfilled into the past.
5. Options/gamma history remains optional/premium and should not block v1.

## v3 modeling priority

1. PDH sweep probability.
2. PDL sweep probability.
3. Expected upside excursion in ATR.
4. Expected downside excursion in ATR.
5. Sweep -> reclaim / acceptance labels from M15/H1.
6. Direction of daily close only as a secondary output.
7. Macro/news as incremental features only if they improve out-of-sample metrics.

## Production gate

Do not call the model "ready" until it passes:

- provider overlap checks;
- strict walk-forward test;
- probability calibration;
- regime breakdown;
- news-day vs non-news-day breakdown;
- spread/slippage simulation;
- a frozen final out-of-sample period that was not used for tuning.

---

# Addendum v3.1 — audit et corrections (2026-09-29)

**Les chiffres ci-dessus datent d'avant correction.** Le code v3 contenait une fuite de données : les features H4/H1/M15 incluaient la session à prédire. Les barres du dimanche gonflaient aussi les résultats sur les données M15 en UTC. Avec le code v3 d'origine sur des données H4 2013–2026, le walk-forward donnait une AUC de 0.93–0.94, ce qui est impossible. Voir `CHANGELOG.md`.

Résultats après correction (données H4 publiques d'un seul fournisseur, sessions à 21:00 UTC, walk-forward annuel) :

| | Dev 2017–2021 | Holdout gelé 2022–fév. 2026 |
|---|---|---|
| PDH AUC (logit) | 0.829 | 0.819 |
| PDH AUC (core3, 3 variables) | 0.823 | 0.814 |
| PDL AUC (logit) | 0.791 | 0.787 |
| PDL AUC (core3) | 0.787 | 0.783 |
| Brier skill PDH / PDL | +0.33 / +0.25 | +0.31 / +0.24 |
| MAE skill excursions haut / bas | 0.00 / −0.03 | 0.00 / +0.01 |

Tests sur d'autres fournisseurs, avec un modèle entraîné uniquement sur les données H4 :

* MT5 M15 de 2025 à mai 2026 : AUC 0.833 / 0.747.
* M15 UTC d'avril à sept. 2026 : AUC 0.791 / 0.795.

Conclusions :

1. Le signal PDH/PDL est réel et stable (tous régimes de volatilité, tous jours, 3 fournisseurs). Il vient surtout d'où la session clôture dans son range. Les autres features n'ajoutent qu'environ +0.005 AUC.
2. GBM, Transformer et macro ne battent pas la régression logistique régularisée.
3. La taille des excursions n'est pas prévisible avec ces features. Seule l'enveloppe au quantile 80 % est utile.
4. Prochain levier probable : l'information intraday de la session suivante (réaction de l'Asie ou de Londres au niveau). Cela change l'heure de décision et demande des données M15 ou M5 longues d'un seul fournisseur.
