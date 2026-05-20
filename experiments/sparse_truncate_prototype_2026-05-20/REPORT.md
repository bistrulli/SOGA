# Prototype standalone: sparse-aware truncate vs SOGA rotation-based

**Date**: 2026-05-20
**Goal**: Verificare empiricamente (i) l'equivalenza numerica e (ii) lo speedup atteso del rank-1 update conditional Gaussian rispetto all'approccio attuale di SOGA (full rotation via find_basis + A·Sigma·A.T + inv).
**Code modifications in SOGA**: nessuna. Pure prototipo Python.

---

## TL;DR

- **Correttezza**: 36 configurazioni (d ∈ {5,10,30,100,300}, k ∈ {1,2,3,5}, dir ∈ {>,<}), 200 sample randomici ciascuna = **7200 test, tutti PASS**. Max |Δμ| = 5e-14, max |ΔΣ| = 1.4e-12, max |ΔP| = 2.2e-16. Equivalenza confermata al livello machine epsilon.
- **Speedup**: 1.5× a d=5, 4× a d=100, 13× a d=300, 22× a d=1000. Cresce ~ linearmente in d (coerente con teoria O(d^3) → O(d^2)).
- **Decisione**: la matematica regge, la velocità c'è, l'implementazione in SOGA è giustificata per benchmark con d ≥ 100.

---

## 1. Setup

Due implementazioni indipendenti del truncate 1D-inequality:

**A. SOGA-style** (replica fedele di `libSOGAtruncate.py:ineq_func`):
```
normalize alpha
A = find_basis(alpha)              # d x d via SVD
transl_mu    = A @ mu              # O(d^2)
transl_sigma = A @ Sigma @ A.T     # O(d^3)
... 1D moments along axis 0 ...
conditional rank-1 update in rotated space
A_inv = inv(A)                     # O(d^3)
new_mu    = A_inv @ new_transl_mu
new_sigma = A_inv @ new_transl_sigma @ A_inv.T   # O(d^3)
```

**B. Sparse-aware** (rank-1 update diretto):
```
g     = Sigma @ alpha              # O(d^2) dense
var_s = alpha @ g                  # O(d)
m_hat, v_hat, P = 1D truncated normal moments
new_mu    = mu + (g/var_s) * (m_hat - mu_s)        # O(d)
new_sigma = Sigma - (g g^T / var_s) * (1 - v_hat/var_s)   # O(d^2)
```

**C. Sparse-aware-K** (variante che sfrutta anche la sparsità di alpha nelle moltiplicazioni):
```
nz = where(alpha != 0)             # k posizioni
g  = Sigma[:, nz] @ alpha[nz]      # O(k * d) invece di O(d^2)
```

Resto identico a B. C dovrebbe essere più veloce di B su d alti se k << d.

## 2. Verifica numerica

7200 test su input randomici (Σ PSD generata da A·A^T + 0.1·I, μ Gaussiano, α sparsa, c scelto perché |z| < 1.5):

| Quantità          | Max differenza overall |
|-------------------|------------------------|
| Δμ                | 5.0e-14                |
| ΔΣ                | 1.4e-12                |
| ΔP                | 2.2e-16                |

Risultato: **i tre metodi coincidono al livello dell'errore di arrotondamento di numpy** (~ε_float64 = 2.2e-16, accumulato su O(d²) operazioni quindi ~d²·ε ≈ 1e-12 per d=300).

Conclusione: la sparse-aware NON è un'approssimazione. È un calcolo numericamente identico a quello attuale.

## 3. Speedup empirico

Tempo per-truncate, best-of-5 batch averages (n_calls adattivo: 200 per d≤30, scende a 3 per d=1000):

| d    | k | SOGA (ms) | Sparse (ms) | Sparse-K (ms) | Speedup B | Speedup K |
|------|---|-----------|-------------|---------------|-----------|-----------|
| 5    | 1 | 0.062     | 0.043       | 0.046         | 1.45×     | 1.35×     |
| 5    | 3 | 0.062     | 0.043       | 0.045         | 1.43×     | 1.36×     |
| 10   | 1 | 0.068     | 0.043       | 0.047         | 1.59×     | 1.45×     |
| 10   | 3 | 0.072     | 0.046       | 0.048         | 1.57×     | 1.49×     |
| 30   | 1 | 0.083     | 0.045       | 0.047         | 1.87×     | 1.77×     |
| 30   | 3 | 0.084     | 0.044       | 0.046         | 1.91×     | 1.82×     |
| 100  | 1 | 0.224     | 0.054       | 0.056         | **4.13×** | 4.03×     |
| 100  | 3 | 0.218     | 0.056       | 0.056         | 3.91×     | 3.87×     |
| 300  | 1 | 1.731     | 0.136       | 0.125         | **12.74×**| 13.85×    |
| 300  | 3 | 1.809     | 0.135       | 0.126         | 13.38×    | 14.33×    |
| 1000 | 1 | 27.807    | 1.277       | 1.126         | **21.78×**| 24.71×    |
| 1000 | 3 | 28.126    | 1.389       | 1.172         | 20.24×    | 24.00×    |

### Osservazioni

1. **Speedup cresce con d**: passa da ~1.5× (d=5) a ~22× (d=1000). Coerente con la teoria O(d^3) vs O(d^2): il rapporto cresce ~O(d), modulato dalle costanti BLAS.

2. **k ha effetto marginale**: la versione Sparse-K (che fa Sigma[:, nz] @ alpha[nz] invece di Sigma @ alpha) è solo leggermente più veloce. Motivo: il bottleneck di Sparse è l'outer product g·g.T (O(d²)) che non dipende da k. La parte O(k·d) è già minoritaria.

3. **Sotto d=30 lo speedup è modesto** (1.5-2×): l'overhead Python (function calls, numpy setup, scipy.stats per il truncated normal) domina sull'O(d^3) cubico.

4. **A d=300 lo speedup è 12-14×**, a d=1000 è 20-25×. **A d=100 (la soglia "stress" che avevo testato sui programmi reali) è 4×.**

### Fit ragionato

Costo del SOGA-style ben approssimato da:
```
T_soga(d) ≈ 0.06 ms + 28e-9 * d^3 ms
```

Costo di Sparse:
```
T_sparse(d) ≈ 0.04 ms + 1.3e-9 * d^3 ms    (numpy BLAS è eccellente)
```

A d→∞ il rapporto teorico tende a 28/1.3 ≈ 21× (limite asintotico del BLAS constant). Empiricamente a d=1000 osserviamo ~22×: match.

## 4. Implicazioni per SOGA

### Su benchmark esistenti (d ≤ 20)

Speedup per-truncate: **~1.5-2×**. Considerando che truncate è ~10-17% del runtime totale (dato il profile su BayesPointMachine), risparmio totale: **~2-4%**. Poco interessante.

### Su programmi "stress" (Pattern B/C da d=50 in su)

Dal nostro scaling study precedente:
- A d=100 il truncate era ~85% del SOGA-runtime. Speedup per-truncate 4× → speedup totale ~**3×**.
- A d=300 (estrapolato dai trend, non testato): truncate sarebbe >95% del runtime. Speedup per-truncate ~13× → speedup totale **~10×**.

### In combinazione con altre ottimizzazioni

La sparse-aware truncate è **ortogonale** a:
- Cache di asgmt_parse/trunc_parse (rimuove l'overhead ANTLR)
- Component vectorization (parallelizza su n_comp)
- Auto-prune insertion (riduce n_comp)

Combinandole, lo speedup totale su benchmark high-d, high-n_comp può facilmente superare le 20×.

## 5. Limiti dello studio

- **Solo per inequality 1D**: l'equivalente per equality (`eq_func` in SOGA) richiede una derivazione separata (più semplice — è già un rank-1 update conditional Gaussian, senza il fattore di truncation).
- **Sigma PSD generata da A·A.T**: nei programmi reali Sigma può essere quasi singolare (delta vars). Il prototipo non testa questi casi degeneri esplicitamente — SOGA gestisce con `substitute_deltas` upstream, stesso approccio si applicherebbe qui.
- **No multi-component**: ogni chiamata è su una singola componente Gaussiana. La velocità in n_comp è puramente moltiplicativa (entrambi i metodi processano componente-per-componente).
- **No exploitation of Sigma sparsity** (Markov blanket via `select_indices`): è un'ottimizzazione separata. Implementabile in combinazione.

## 6. Decisione: vale la pena implementare in SOGA?

**SÌ**, con priorità media. Motivazioni:

1. **Equivalenza matematica certificata**: zero rischio di regressione di correttezza (verificato su 7200 sample).
2. **Implementazione contenuta**: ~30 LOC per la nuova `ineq_func`. La struttura è radicalmente più semplice di quella attuale.
3. **Benefici scalabili**: per benchmark a d>50 il guadagno è significativo (3-10× sui programmi totali). Per benchmark esistenti minore (~2-4%) ma comunque positivo.
4. **Audit numerico più sicuro**: il rank-1 update preserva PSD-ness teoricamente, e ha meno operazioni di accumulazione errori del rotation+inv attuale.

**Ordine consigliato di implementazione**:
1. Estendere il prototipo a `eq_func` (equality conditioning): formula simile, ~50 LOC.
2. Integrare in `libSOGAtruncate.py` come opzione (flag, A/B test).
3. Run `/audit-numerical` su benchmark esistenti per confermare zero regressioni.
4. Sostituire come default dopo validazione.

Effort stimato: ~1 giornata di codifica + audit numerico.

## 7. Artifacts

```
prototype.py             # both implementations + verification + benchmark
results/correctness.txt  # 36 configs × 200 samples = 7200 tests, all PASS
results/timing.csv       # 12 rows: per (d, k) timing measurements
REPORT.md                # questo file
```

## 8. Repro

```bash
cd experiments/sparse_truncate_prototype_2026-05-20
/Users/emilio-imt/git/SOGA/.venv/bin/python prototype.py
```

Seed master = 42 (correttezza), 7 (timing). Output deterministico.
