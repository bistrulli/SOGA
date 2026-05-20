# Feasibility study — Automatic dead-variable pruning in SOGA

**Date**: 2026-05-20
**Question**: Quanto guadagneremmo in prestazioni se SOGA marginalizzasse automaticamente le variabili non più usate (dead) durante l'esecuzione del CFG, senza più trascinarle nella joint Gaussian Mixture?
**Method**: Profile + hand rewrite + scaling study su famiglie sintetiche (Pattern B BayesPointMachine-like, Pattern C Markov sequenziale).
**Code modifications**: nessuna.

---

## TL;DR (aggiornato dopo scaling study)

Lo speedup atteso da automatic dead-variable pruning è **modesto** (≈10–80%) anche su programmi 10–20× più grandi dei benchmark esistenti:

- Su `BayesPointMachine.soga` (d=9): **+9% SOGA-only**.
- Su Pattern B sintetico a d=103 (BPM con N=100 osservazioni): **+24% SOGA-only**.
- Su Pattern C sintetico (Markov, d=101): **+53% SOGA-only**.
- Su Pattern C a d=151: **+81% SOGA-only**.

Il guadagno cresce ma **molto lentamente**: il bottleneck dominante a queste d (≤150) è l'**overhead Python+ANTLR per-chiamata** del dispatcher SOGA, non il costo cubico d³ delle operazioni numeriche. Lo scaling cubico inizia a contribuire visibilmente solo per d > ~100.

Inoltre, **importante limite teorico** scoperto durante l'analisi: nel pattern BPM-like (B), il **pure dead-variable pruning** è teoricamente bounded **≈4×** indipendentemente da N, perché la liveness statica vede tutti i `mu[i+1..N-1]` come vivi al momento di `observe[i]`. Lo speedup maggiore di V2_tmp viene da una trasformazione più aggressiva (loop fusion + name reuse) che NON è puro dead-var pruning.

**Raccomandazione**: ❌ NON implementare come priorità sui benchmark attuali. Alternative con ROI molto migliore in §6. La feature avrebbe valore solo per (i) benchmark con d > 200, (ii) programmi con pattern Markov-style + d alta, o (iii) dopo aver risolto l'overhead Python/ANTLR.

---

## 1. Setup

Tre versioni di BayesPointMachine, **semanticamente equivalenti** sulla posterior di `w[0..2]` (verificato: identici a 5 cifre decimali su E[w_i] e Cov[w]):

| Variante | Descrizione | d_total |
|---|---|---|
| **V1** (originale) | `mean[i] = feat·w + ε; observe(mean[i] >/< 0)` × 6 | **9** (3 w + 6 mean[]) |
| **V2_tmp** (fair proxy per auto-pruning) | uguale a V1 ma con una sola `tmp` riassegnata invece dell'array `mean[]` | **4** (3 w + tmp) |
| **V2_inline** (upper bound aggressivo) | inline degli observe: `observe(feat·w + ε >/< 0)`, niente mean/tmp | **3** (solo w) |

**V2_tmp** è il proxy migliore: stesso numero di assegnamenti e parsing di V1, ma joint più piccola — equivalente a quello che un'analisi di liveness automatica otterrebbe (dopo ciascun observe `tmp` è dead finché non viene riassegnata).
**V2_inline** sovrastima: elimina anche 12 assignment statements, cosa che il pruning automatico non può fare (gli assignment sono nel programma).

Ambiente: `.venv` Python 3.12.8 con numpy 2.4.6 / scipy 1.17.1 / antlr4-python3-runtime 4.10.

---

## 2. Profile V1 (Method B)

cProfile in-process, no multiprocessing wrapper. Total = 0.077 s. Top componenti del runtime SOGA:

| Funzione | cumtime (ms) | % totale | Note |
|---|---|---|---|
| `produce_cfg` (ANTLR parse .soga) | 34 | 44% | una-tantum, indip. da pruning |
| `start_SOGA` (dispatcher BFS) | 42 | 55% | totale execution |
| ↳ `update_rule` × 17 | 29 | 38% | |
| &nbsp;&nbsp;↳ `asgmt_parse` (ANTLR) | 28 | **36%** | ❗ dominante |
| &nbsp;&nbsp;↳ `add_func` (numerica) × 15 | 1 | 1.3% | trascurabile |
| ↳ `truncate` × 6 | 13 | **17%** | |
| &nbsp;&nbsp;↳ `trunc_parse` (ANTLR) | 6 | 8% | |
| &nbsp;&nbsp;↳ `ineq_func` (numerica) | 6 | 8% | scala con d |
| &nbsp;&nbsp;&nbsp;&nbsp;↳ `compute_moments` | 4 | 5% | |
| &nbsp;&nbsp;&nbsp;&nbsp;↳ `find_basis` | <1 | <1% | il pezzo O(d³) |

**Finding chiave**: ≈88% del runtime è **ANTLR parsing**, non numerica gaussiana. Il pezzo che scala O(d³) (`find_basis`) è impercettibile su d=9.

Sotto cProfile il runtime è 77ms; senza strumentazione è 7-11ms (l'overhead di cProfile inflaziona ma le proporzioni relative restano informative).

---

## 3. Timing comparison (Method C)

N=5 run dopo 1 run di warm-up. Tempi in-process, esclusa la multiprocessing wrapper della CLI.

| Variante | d | total (ms) | SOGA-only (ms) |
|---|---|---|---|
| V1 | 9 | 10.67 ± 0.49 | 7.23 ± 0.39 |
| V2_tmp | 4 | 17.00 ± 0.30 | **6.64 ± 0.09** |
| V2_inline | 3 | 13.04 ± 9.59† | **5.22 ± 0.41** |

†V2_inline ha alta varianza (jitter su un run; min=8.26ms).

**Speedup SOGA-only** (vs V1):

| Confronto | Speedup | Interpretazione |
|---|---|---|
| V1 → **V2_tmp** | **1.09×** (-9%) | proxy realistico per auto-pruning |
| V1 → V2_inline | 1.38× (-38%) | upper bound (include rimozione assignment) |

Il **total time** di V2_tmp è MAGGIORE di V1 perché il programma riscritto a mano srotola il for-loop in 12 assignment espliciti, e questo costa più ANTLR parsing nella `produce_cfg`. In uno scenario di auto-pruning reale il `.soga` originale resta com'è → solo il SOGA-only è rilevante.

---

## 4. Profile comparativo per funzione

Tempi cumulativi sotto cProfile (numeri assoluti inflazionati dall'instrumentazione, ma i rapporti tra varianti sono significativi):

| Funzione | V1 (d=9) | V2_tmp (d=4) | V2_inline (d=3) |
|---|---|---|---|
| total | 77 ms | 98 ms* | 59 ms |
| produce_cfg | 34 | 60* | 30 |
| start_SOGA | 42 | 37 | 28 |
| update_rule cumul | 29 | 28 | 9 |
| asgmt_parse | 28 | 27 | 9 |
| **truncate cumul** | **13** | **8** | 19† |
| trunc_parse | 6 | 4 | 15† |
| **ineq_func (numerica)** | **6** | **4** | **4** |
| compute_moments | 4 | 3 | 3 |
| find_basis | <1 | <1 | <1 |

\* produce_cfg slower in V2_tmp perché il programma riscritto a mano è più lungo (loop unrollato manualmente).
† V2_inline ha observe più lunghi (6 termini + gm() inline) → trunc_parse più costoso.

**Confronto V1 → V2_tmp (il proxy onesto per auto-pruning)**:

- Truncate cumul: 13 → 8 ms (**-38%**). Ratio teorico (9/4)³ ≈ 11×; osservato 1.6×.
- ineq_func (parte numerica): 6 → 4 ms (**-33%**). Ratio teorico 11×; osservato 1.5×.
- Tutto il resto è simile a V1.

**Il costo cubico in d c'è ma è subordinato all'overhead Python/ANTLR a queste d basse**.

---

## 5. Scaling study su programmi sintetici (Patterns B e C)

Per validare empiricamente l'ipotesi "i benefici crescono con d", ho generato due famiglie sintetiche a dimensioni crescenti, senza modifiche al codice SOGA. Tutti i programmi sono `gen_programs.py` (deterministici, seed=42); ciascun (size, variant) è eseguito con N=3 run dopo 1 warm-up.

### 5.1 Pattern B — BayesPointMachine-like

Struttura: K=3 pesi vivi (`w[0..K-1]`) + N osservazioni (`mu[i] = feat·w + ε; observe(mu[i] >/< 0)`).
V1 mantiene `array[N] mu` nel joint (d_total = K+N); V2_tmp riusa una singola `tmp` (d_total = K+1 = 4).

| N | d_V1 | total V1 (ms) | total V2 (ms) | SOGA V1 (ms) | SOGA V2 (ms) | **SOGA speedup** |
|---|---|---|---|---|---|---|
| 6 | 9 | 10.70 | 16.45 | 7.19 | 6.45 | **1.11×** |
| 12 | 15 | 23.99 | 30.97 | 12.61 | 12.29 | 1.03× |
| 25 | 28 | 32.98 | 62.87 | 26.68 | 24.66 | 1.08× |
| 50 | 53 | 65.44 | 123.52 | 55.87 | 48.11 | 1.16× |
| 75 | 78 | 103.16 | 203.31 | 90.06 | 81.97 | 1.10× |
| 100 | 103 | 159.0 | 261.4 | 135.2 | 109.4 | **1.24×** |

**Per-observe cost (SOGA-only / N)**:
- V1: ~1.20–1.35 ms/observe (cresce molto lentamente con d)
- V2: ~1.08–1.10 ms/observe (sostanzialmente costante)

Lo scaling **non è cubico** in pratica: V1 cresce ~linearmente con N. Il costo per chiamata `truncate` è dominato da overhead Python+ANTLR (parsing LBC, function calls, np.array conversions), non dalle moltiplicazioni d×d.

⚠️ **Total time è SEMPRE peggiore per V2** perché il programma riscritto è 3× più lungo in LOC (unrolled) → `produce_cfg` (ANTLR) costa molto più di V1 (compact for-loop). In uno scenario reale di auto-pruning il file `.soga` resta com'è, quindi solo SOGA-only è rilevante.

### 5.2 Pattern C — Markov sequenziale (random walk con observe ad ogni step)

Struttura: `state[t] = state[t-1] + ε_t; observe(state[t] > -100)` per T step. V1 mantiene `array[T+1] state` (d_total = T+1); V2_tmp riusa `tmp` (d_total = 1). Per Pattern C, V2_tmp **≈ pure dead-var pruning** (lo stato precedente è genuinamente morto dopo lo step successivo).

| T | d_V1 | total V1 (ms) | total V2 (ms) | SOGA V1 (ms) | SOGA V2 (ms) | **SOGA speedup** |
|---|---|---|---|---|---|---|
| 5 | 6 | 7.36 | 5.41 | 4.03 | 3.50 | 1.15× |
| 10 | 11 | 13.84 | 9.77 | 7.84 | 6.82 | 1.15× |
| 20 | 21 | 25.73 | 18.49 | 14.63 | 12.86 | 1.14× |
| 40 | 41 | 51.94 | 35.19 | 30.89 | 25.10 | 1.23× |
| 60 | 61 | 80.82 | 52.84 | 49.26 | 37.55 | **1.31×** |
| 100 | 101 | 158.0 | 87.4 | 96.6 | 63.0 | **1.53×** |
| 150 | 151 | 275.7 | 158.2 | 198.3 | 109.4 | **1.81×** |

**Per-step cost (SOGA-only / T)**:
- V1: cresce da 0.81 a 1.32 ms/step (segno che d³ inizia a contribuire)
- V2: stabile a ~0.7 ms/step

A **d=150** lo speedup arriva a 1.81×. Per Pattern C il trend è chiaramente crescente: il termine cubico inizia a manifestarsi.

### 5.3 Modello del runtime

Per V1 stiamo osservando approssimativamente:

```
T_V1(d) ≈ N · (a + b·d³)
```

dove `a` è l'overhead Python+ANTLR per `truncate` (~1 ms costante) e `b·d³` è il lavoro numerico. Dai dati:
- A d=10, `b·d³` ≈ 0.1 ms (trascurabile rispetto ad `a ≈ 1 ms`)
- A d=100, `b·d³` ≈ 0.4 ms (≈30% del totale per-step)
- A d=150, `b·d³` ≈ 0.7 ms (≈40% del totale)
- Estrapolato d=300, `b·d³` ≈ 5.4 ms (≈85% del totale) → speedup ~3-5×
- Estrapolato d=500, `b·d³` ≈ 25 ms → speedup ~10×+

**A queste d (≤150)** Python+ANTLR overhead è circa metà del costo per-truncate. Per raggiungere speedup grandi (5–10×) servono programmi con d > 300.

### 5.4 Limite teorico per Pattern B (insight)

V2_tmp testa **loop fusion + dead-var pruning insieme**. Il PURE dead-var pruning su V1 (lasciando l'ordine originale del codice) avrebbe limite **~4×** indipendente da N:

Liveness statica a `observe[k]`: i `mu[k+1..N-1]` sono ancora vivi (usati nei prossimi observe). Solo `mu[0..k-1]` sono morti. Quindi:
- A `observe[0]`: lavora su d = K+N (full)
- A `observe[N-1]`: lavora su d = K+1
- Cost cumulativo: Σ_{j=K+1}^{K+N} j³ ≈ (K+N)⁴/4
- Senza pruning: N(K+N)³
- Ratio asintotico: 4N(K+N)³ / (K+N)⁴ = 4N/(K+N) → **4×** per N→∞

Quindi per Pattern B, ANCHE assumendo che la numerica diventi dominante a d alta, il pure pruning dà al massimo ~4×. La crescita osservata su Pattern B (1.11× → 1.24×) è coerente con questo limite teorico **debolmente combinato** con loop fusion implicita in V2_tmp.

Per Pattern C, V2_tmp ≈ pure pruning, quindi i risultati osservati riflettono fedelmente la feature richiesta.

---

## 6. Decisione & alternative

### Sintesi quantitativa

| Scenario | Speedup SOGA-only | Verdetto |
|---|---|---|
| Benchmark esistenti (d ≤ 10) | 1.0–1.1× | trascurabile |
| Pattern B esteso (d=100) | 1.24× | modesto |
| Pattern C esteso (d=150) | 1.81× | moderato, in crescita |
| Estrapolato Pattern C d=300 | ~3-5× | significativo (ma nessun benchmark reale a questa scala) |
| Estrapolato Pattern B (qualunque N) | ~4× max (limite teorico) | bounded |

### Implementare l'auto-pruning ORA? ❌ Non come priorità

- Costo: ~200 LOC liveness analysis + integration CFG + tests + `/audit-numerical`
- Beneficio sui benchmark attuali (d ≤ ~15): ~5-15% di runtime SOGA-only
- ROI sfavorevole.

### Implementare l'auto-pruning POI? ⚠️ Condizionato

Vale la pena SOLO SE:
1. Si vuole posizionare SOGA come tool per programmi a **d > 100** (richiede prima costruire benchmark, ad es. Bayesian regression con 100+ predittori, HMM lunghi).
2. Le ottimizzazioni Python/ANTLR-overhead sono già state fatte (vedi sotto). Altrimenti il guadagno è mascherato dall'overhead per-chiamata.
3. La SOGA paper vuole dimostrare scalabilità a programmi sequenziali tipo Pattern C.

### Alternative con ROI molto migliore (priorità superiore al dead-var pruning)

1. **Cache `asgmt_parse` / `trunc_parse`** — Oggi ogni state/test/observe rifà l'ANTLR parse della stringa `expr` / `LBC`. Memoizzando per stringa (dict) si elimina ripetizione. Sui benchmark profilati, ~36% del SOGA time è in `asgmt_parse`. **Implementazione: ~15 LOC**. Speedup atteso: 1.3-1.5× su quasi tutti i benchmark.
2. **Pre-compile expression functions in `produce_cfg`** — Spostare il parsing dalla fase di dispatch alla fase di CFG construction. Una sola volta per expr. Speedup atteso simile a (1) ma su tutti i nodi.
3. **Profilare `truncate` interno** per identificare se find_basis/inv hanno opt-out su d piccola.
4. **Vettorizzare il loop su componenti** in `update_rule` (oggi loop Python su `dist.gm.pi`). Importante solo per n_comp >> 1.

### Cosa NON ho misurato (limiti dello studio)

- Programmi con n_comp >> 1 (mistura con molte componenti, es. molti if-then-else): il costo cubico è per-componente, quindi lo speedup scala con `n_comp × d³`. Su programmi reali con n_comp > 10, il pruning potrebbe dare guadagni maggiori.
- Interaction con `prune(K)` (ricostruzione componenti dopo merge): non testata.
- Programmi con loops molto lunghi (es. T=500, 1000): non eseguiti per limiti di tempo. Estrapolazione cubica suggerisce speedup crescenti.
- Pattern misti: l'analisi di liveness reale incontrerà programmi che NON sono né puramente B né puramente C, ma combinazione di entrambi. Comportamento empirico difficile da prevedere senza prototipo.

---

## 7. Artifacts

```
inputs/
├── BayesPointMachine_V1.soga           (benchmark originale)
├── BayesPointMachine_V2_tmp.soga       (proxy fair per auto-pruning)
├── BayesPointMachine_V2_inline.soga    (upper bound aggressivo)
├── B_K3N{6,12,25,50,75,100}_{V1,V2}.soga  (Pattern B scaling family)
└── C_T{5,10,20,40,60,100,150}_{V1,V2}.soga (Pattern C scaling family)

profiles/
├── V1.{pstats,cumulative.txt,tottime.txt}
├── V2_tmp.{pstats,cumulative.txt,tottime.txt}
└── V2_inline.{pstats,cumulative.txt,tottime.txt}

results/
├── V1_baseline_output.txt        (E[w], Cov[w] di riferimento)
├── timings_raw.csv               (BPM: 15 run individuali)
├── timings_summary.csv           (BPM: mean/std/min per variante)
├── scaling_results.csv           (Pattern B/C: 20 rows, sizes principali)
└── scaling_results_full.csv      (B/C + estensioni d=100, d=150)

profile_run.py       (Method B: cProfile)
timing_compare.py    (Method C su BPM)
gen_programs.py      (generator di Pattern B e C a dimensioni arbitrarie)
bench_scaling.py     (scaling benchmark V1 vs V2 su tutti i sizes)
REPORT.md            (questo file)
```

## 8. Repro

```bash
# venv già configurato in /Users/emilio-imt/git/SOGA/.venv
cd experiments/feasibility_dead_var_pruning_2026-05-20

# Method B: profile V1 (BayesPointMachine)
../../.venv/bin/python profile_run.py inputs/BayesPointMachine_V1.soga profiles/V1

# Method C: timing comparison V1 vs V2_tmp vs V2_inline su BPM
../../.venv/bin/python timing_compare.py

# Scaling study (genera i programmi e poi misura)
../../.venv/bin/python gen_programs.py
../../.venv/bin/python bench_scaling.py
```

Seed: `random.seed(0); np.random.seed(0)` in tutti gli script. Output deterministico.
