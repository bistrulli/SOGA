# Plan: matrix-gm-stress-test

## Goal
Costruire una campagna di stress test end-to-end (parser + semantica + numerica) per la nuova superficie matrix-GM di SOGA prima della consegna a uno studente, con baseline analitica chiusa o forward Monte Carlo numpy. Stato finale: 106 test passanti (5 xfail) + LIMITATIONS.md che documenta esplicitamente i bug latenti scoperti dagli esperti durante la fase di plan.

## Context
La branch `feat/matrix-gm-integration` (4 commit ahead di origin) ha aggiunto in 4 mesi di lavoro: matrice-variate Gaussian MN(M,U,V) Kronecker + dense sentinel, matrix_gm/matrix_gm_full constructors con auto-Kronecker detection, A@X / X@A / X@Y (Isserlis 3-term), X+Y, transp(X), X[i,j] extract con cross-cov, X[i,j]=c (Schur densification), observe(X[i,j] op c) inclusi `==` delta + linear combo O7. 385 test esistenti + 2 programmi exemplar (showcase, advanced). Lo scopo della campagna è scoprire bug nascosti su combinazioni Op × Shape × CovKind non ancora testate.

I 6 esperti consultati in Phase D hanno identificato:
- 0/8 dei "grammar negatives" originali sono parse-time error (sono semantici)
- 3 silent-crash path nel codice esistente (affine post-dense-sentinel: C1/C7/Path-5)
- 1 correttezza bug documentato (Gap 3: stale cross-cov after observe)
- N=1e4 insufficient per 16×16 covariance test con tol 1%

Strategy: NON fixare i bug latenti in questo plan (scope creep). Esporli come `xfail` con marker e documentarli in `LIMITATIONS.md`. Lasciare al prossimo plan il fix.

## Constraints
- NO scope creep: niente fix di bug numerici/semantici scoperti durante l'analisi (gestiti via xfail + LIMITATIONS.md)
- NO modifica di `src/libMatrix*.py` o `src/libSOGA*.py`
- File prodotti (lista completa e chiusa): nuovi file in `tests/stress_matrix_gm/` + `programs/stress/` + `docs/LIMITATIONS.md` (nuovo) + `results/qa_stress_<runid>/` (generato a runtime, non source) + modifica `docs/MATRIX_GM_QUICKSTART.md` (documentale, permessa). Se `tests/test_grammar_matrix.py` esiste e contiene `_CountingErrorListener`, quella classe viene SPOSTATA (cut+paste) in `tests/stress_matrix_gm/conftest.py` e importata dal file originale via `from tests.stress_matrix_gm.conftest import _CountingErrorListener` — il file originale RIMANE, non viene eliminato. Nessun altro file di `tests/` esistente viene modificato.
- 106/106 test devono passare verde (xfail conta come pass se fallisce; xpass FALLISCE — segnala fix silenzioso)
- I 385 test esistenti devono continuare a passare invariati
- `programs/Example/Bernoulli.soga` → E[theta]=0.25689 invariato
- `programs/Example/matrix_gm_showcase.soga` e `matrix_gm_advanced.soga` devono continuare a riprodurre i valori documentati nei commenti
- Pytest exit 0 obbligatorio prima del commit
- NO push finchè campagna 100% verde
- Master seed = 42, deterministico cross-platform per quanto possibile

## Op x Shape x CovKind Coverage Matrix

La tabella seguente definisce la copertura della campagna. Ogni cella indica il test ID (o "xfail" o "—" per esclusione giustificata). "Kron" = storage Kronecker `U⊗V`; "Dense" = storage full covariance; "Rect" = shape non-quadrata.

| Op                    | 1×1 Kron | 2×2 Kron | 3×3 Kron | 8×8 Kron | 16×16 Kron | 2×2 Dense | 3×3 Dense | 2×3 Kron (Rect) | 3×2 Kron (Rect) |
|-----------------------|----------|----------|----------|----------|------------|-----------|-----------|-----------------|-----------------|
| constructor (matrix_gm)      | P1       | P4       | P5       | P6       | MC-14      | —         | —         | P2              | P3              |
| constructor (matrix_gm_full) | —        | P4b(Kron-auto) | P5b(NearMiss) | — | — | P4c(Dense) | P5c(Dense) | — | — |
| A@X (left affine)     | —        | P7       | P8       | —        | —          | xfail-F2b | —         | P8b (3×2)       | —               |
| X@A (right affine)    | —        | P9       | P10      | —        | —          | xfail-F2a | —         | —               | P10b (3×2)      |
| X@Y (Isserlis)        | —        | P11-C1   | P12-C3   | —        | —          | —         | —         | P13-C5          | —               |
| X+Y (sum)             | —        | P14      | —        | —        | MC-8       | —         | —         | —               | —               |
| X+N iso noise         | —        | P15      | —        | —        | —          | —         | —         | —               | —               |
| transp                | —        | P16      | —        | —        | —          | —         | —         | P17 (2×3)       | —               |
| X[i,j] extract        | —        | P18-Kron | P20-Kron | P21-Kron | MC-14      | P19-Dense | —         | —               | —               |
| X[i,j]=c (Schur)      | —        | P22      | —        | —        | —          | P22b-Dense| —         | —               | —               |
| observe(X[i,j]>c)     | —        | P23-Kron | —        | —        | —          | P24-Dense | —         | —               | —               |
| observe(X[i,j]==c)    | —        | P25      | —        | —        | —          | —         | —         | —               | —               |
| observe linear O7     | —        | P26      | —        | —        | —          | —         | —         | —               | —               |
| observe+affine (C1/Path5) | —   | xfail-F1 | —        | —        | —          | —         | —         | —               | —               |
| cross-cov staleness (Gap3) | — | xfail-F3 | —        | —        | —          | —         | —         | —               | —               |
| extract+write cross-cov (F4) | — | xfail-F4 | —       | —        | —          | —         | —         | —               | —               |

Esclusioni giustificate (—):
- 16×16 Dense: non supportato nello scope della campagna; `matrix_gm_full` Dense su 16×16 non è un use-case documentato
- Rect × Dense: non è una combinazione prevista dalla semantica di `matrix_gm_full`
- Tutte le combinazioni non elencate: la copertura è centrata sui path critici identificati in Phase D

## MC Tolerance Derivation

Le tolleranze MC sono derivate dalla formula 3-sigma del Central Limit Theorem.

**Formula base — errore relativo media per-entry:**
`tol_mean = 3 / sqrt(N)`  (3-sigma, distribuzione normale, significatività ~0.3%)

**Formula base — errore relativo covarianza per-entry (stimatore campionario):**
`tol_cov = 3 * sqrt(2/N)`  (da Var(hat_sigma_ij) ~ 2*sigma^4/N per distribuzione Gaussiana standardizzata)

**Formula Frobenius relativa (m entry):**
`tol_frob_rel = 3 * sqrt(2/N)`  (la scala si cancella nel rapporto Frobenius, il termine sqrt(m) si elimina)

| Metrica           | N      | Formula                | Bound 3-sigma | Tol adottata | Margine    |
|-------------------|--------|------------------------|---------------|--------------|------------|
| mean per-entry    | 1e5    | 3/sqrt(1e5)            | 0.95%         | 2%           | 2.1×       |
| Frob cov          | 1e5    | 3*sqrt(2/1e5)          | 1.34%         | 5%           | 3.7×       |
| mean 16×16        | 2e4    | 3/sqrt(2e4)            | 2.12%         | 5%           | 2.4×       |
| Frob cov 16×16    | 5e4    | 3*sqrt(2/5e4)          | 1.90%         | 10%          | 5.3×       |

Tutti i margini sono > 2×, il che permette di assorbire correlazioni tra entry e non-idealità della distribuzione di test. Le tolleranze sono deliberatamente conservative (margine > 2×) per un smoke test — non per un test di regressione numerica fine.

## xfail Policy

`pytest.mark.xfail(strict=True)` significa: il test DEVE fallire; se passa (xpass) pytest fallisce il run — questo espone un fix silenzioso. Corretto per "lock current behavior".

Requisito aggiuntivo (rispetto all'iter precedente): ogni xfail DEVE specificare `raises=<ExceptionType>` o `match=<pattern>` per vincolare la signature dell'errore atteso. Esempio:
```python
@pytest.mark.xfail(strict=True, raises=TypeError, reason="C1: _matrix_affine_left receives None after observe+dense path")
```
Senza questo constraint, un xfail che passa per un motivo diverso (es. diverso tipo di eccezione) resterebbe nascosto.

## Approach
Suite stratificata in 4+1 categorie, totale 106 test:

1. **Grammar battery — 50 casi** (parametrize): 32 positivi su 7 gruppi tassonomici (A-G); 8 negativi genuinamente parse-time (vedi sezione Grammar Negatives); 10 edge case extra (vedi sezione Grammar Edge Cases). Ogni negativo annotato con la production/token ANTLR che lo causa.

2. **Operations matrix coverage — 30 programmi** (parametrize per shape, funzioni separate per categoria): closed-form analytical ground truth con formule documentate da `gaussian-mixture-expert`. Tolerance 1e-6 per ops exact, 5e-2 per Isserlis NKP.

3. **MC validation — 15 programmi** (funzioni separate): forward Monte Carlo numpy con vectorized Cholesky sampling. Tolerance derivata dalla tabella MC Tolerance Derivation sopra.

4. **Property tests hypothesis — 6 invarianti** (funzioni separate): 50 trial ognuno, `deadline=None`, generatore PSD con `assume(eigvalsh > 1e-6)`. P5 usa verifica esplicita del flag di storage (non solo equivalenza numerica) per distinguere Kron da Dense.

5. **Fragile path probes — 5 xfail tests** (F1, F2a, F2b, F3, F4) con `strict=True` + `raises=<ExceptionType>` + `match=<pattern>` per ogni probe.

Output:
- `tests/stress_matrix_gm/` con 5 file di test + 3 helper + `conftest.py`
- `programs/stress/` con 45 file `.soga` (30 ops + 15 mc)
- `docs/LIMITATIONS.md` (nuovo file) con tabella bug latenti + workaround + riferimento al prossimo plan
- `docs/MATRIX_GM_QUICKSTART.md` (modifica: aggiungere sezione "Known issues" con cross-link a LIMITATIONS.md)
- `results/qa_stress_<runid>/` (generato a runtime, non source-controlled) con `report.md`, `mc_summary.csv`, `raw.json`

Tooling: `pytest-json-report` per aggregato post-run. `generate_report.py` script wrapper che produce il markdown finale.

## Grammar Negatives — Parse-time Guarantee

I seguenti 8 negativi sono genuinamente parse-time. Ogni entry include la production/token ANTLR che causa il rifiuto e il motivo per cui NON è semantico.

| ID  | Input snippet (programma minimale)              | Rule/Token ANTLR che fallisce                     | Asserzione nel test                                | Perché parse-time (non semantico)                                    |
|-----|------------------------------------------------|----------------------------------------------------|----------------------------------------------------|----------------------------------------------------------------------|
| N1  | `vars { real x; } x = 1.2.3;`                 | `NUM` lexer rule (float mal-formato)               | `listener.errors > 0` dopo `SOGA.g4` parse        | Il lexer emette un token erroneo/non riconosciuto prima del parser   |
| N2  | `vars { real x; } x = 1`  (niente `;`)        | `;` token obbligatorio in `stat` rule              | `listener.errors > 0`                              | Il parser attende `;` trovando EOF: mismatched token                 |
| N3  | `vars { matrix[2][2] X; } X = transp(X;`      | `)` token obbligatorio in `mat_atom` rule          | `listener.errors > 0`                              | Il parser attende `)`, trova `;`: token mismatch                     |
| N4  | `vars { matrix[2][2] A, X; } X = A @;`        | `mat_atom` rule: operando destro mancante          | `listener.errors > 0`                              | Il parser attende mat_atom dopo `@`, trova `;`: no alternative match |
| N5  | `vars { matrix[2][2] X; } X = [1,2;`          | `]` token obbligatorio in `mlist` rule             | `listener.errors > 0`                              | Il parser attende `]`, trova `;`: mismatched token                   |
| N6  | `vars { matrix[2][2] X; } X = X @ @ X;`       | `mat_atom` rule atteso tra i due `@`               | `listener.errors > 0`                              | Il parser attende mat_atom, trova `@`: no valid alternative          |
| N7  | `vars { matrix[2][2] A, X; } A = X @ X; (`    | EOF atteso dopo l'ultimo statement                 | `listener.errors > 0`                              | Token extra `(` dopo l'ultimo statement viola la rule `prog`         |
| N8  | `vars { real x; } x = (1 + ;`                | `expr` rule: operando destro mancante in addizione       | `listener.errors > 0`                        | Parser attende un termine dopo `+`, trova `;`: mismatched token, grammar-version indipendente |

Note: il set originale di 8 negativi è stato rivisto per garantire robustezza grammar-version indipendente. Rimossi: N6-originale (float in shape — grammar permissivo), N8-originale (commenti-only e `vars{}` — dipendente da rule SOGA.g4). Sostituiti con: N6 → doppio operatore `@` (expr sempre invalida), N8 → espressione con RHS mancante `(1 + ;` (sempre mismatched token in expr rule). `X-Y` (IDV con trattino), `matrix=1`, `row_sum=5` esclusi perché tokenizzati correttamente da SOGA.g4.

## Grammar Edge Cases (10 extra)

| ID  | Input                                  | Classificazione | Verifica attesa              |
|-----|---------------------------------------|-----------------|------------------------------|
| E1  | `Y=X` (add vs mat_expr ambiguity)     | positivo        | parse OK, nessun errore      |
| E2  | `X[-1,0]` (NUM negativo indice)       | negativo parse  | NUM negativo non valido come indice nella grammar |
| E3  | `[[1]]` (mlist single inner list)     | positivo        | parse OK, mlist valido       |
| E4  | `X@Y` (matmul no spazi)               | positivo        | parse OK (tokenizzazione robusta) |
| E5  | `Y=2*matrix_gm_full(...)` (in expr)   | positivo        | parse OK                     |
| E6  | `[[-1]]` (NUM negativo in list)       | positivo/negativo (grammar-dep) | da verificare empiricamente durante implementazione |
| E7  | Programma vuoto                        | negativo parse  | `vars` vuota → parser error  |
| E8  | `X[1]` vs `X[1,2]` in TRUNC          | positivo × 2    | entrambi parse OK in rispettivi contesti |
| E9  | `row_sum=5` (keyword-like IDV)        | positivo        | IDV non reserved → parse OK  |
| E10 | `Y=2*matrix_gm(...)` inline expr      | positivo        | parse OK                     |

## Alternatives considered

- **Alt A: fix bug latenti contestualmente (no xfail)**. Pro: codebase più pulito al merge. Contro: scope creep significativo (richiede modifica `_matrix_affine_left/right`, `_dense_update_component`, `extract_scalar_from_matrix`, + `/audit-numerical` pre-flight, +3-5 ore di lavoro). Rigettato perché lo scopo dichiarato è "campagna di test, non feature", e perché fixare bug semantici dovrebbe avere il suo plan dedicato con cross-review separato.

- **Alt B: MC validation con N=1e4 uniforme**. Pro: runtime totale ≤ 30s. Contro: tol 5% Frobenius (3-sigma bound per N=1e4: 3*sqrt(2/1e4)=4.2%) → margine solo 1.2×, insufficiente. Rigettato in favore di N variabile (1e5/2e4/5e4) per mantenere margine >2× su tutte le metriche.

- **Alt C: MCMC baseline (Pyro/NumPyro) invece di forward MC numpy**. Pro: pattern standard di validation per PPL. Contro: 10× più lento da scrivere (50+ righe per programma vs 5 con forward MC), MCMC è inappropriato per matrix-variate (observe = Tallis closed-form, no need for HMC), MC noise è ben caratterizzato analiticamente. Rigettato perché Pyro/NumPyro hanno proprie sottigliezze numeriche che mescolare con SOGA crea due fonti di errore invece di una.

- **Alt D: solo regression test sui programmi exemplar esistenti (showcase + advanced)**. Pro: 30 min di lavoro. Contro: copertura prossima allo zero su combinazioni nuove. Rigettato perché è proprio la smoke battery che già esiste e non basta — i 3 silent-crash path identificati dagli esperti non sono toccati dai due exemplar.

## Sub-tasks (atomic)

Ogni sub-task ha esattamente un criterio di completamento ("Done when: ..."). Le dipendenze inter-iter sono esplicite: ogni iter non inizia finché l'iter precedente non è completamente verified (tutti i test dell'iter passano in verde).

### Iter 1 — Infrastructure + helpers (≈ 1.5h)

Prerequisito: nessuno (è il primo iter).

- [ ] [iter:1] [agent:test-engineer] [area:tests/infra] Creare `tests/stress_matrix_gm/__init__.py` vuoto + `conftest.py` (SEED=42, `rng` fixture function-scoped con `numpy.random.default_rng(SEED)`, sys.path setup per importare `src/`, `_CountingErrorListener` spostato da test_grammar_matrix se già presente). Done when: `python -m pytest tests/stress_matrix_gm/ --collect-only` exit 0 con 0 test collected (solo setup).

- [ ] [iter:1] [agent:test-engineer] [area:tests/infra] Creare `programs/stress/README.md` (breve: scopo directory, naming convention `{op}_{shape}_{covkind}_{idx}.soga`). Done when: il file esiste e `ls programs/stress/` exit 0.

- [ ] [iter:1] [agent:test-engineer] [area:tests/infra] `tests/stress_matrix_gm/mc_forward.py`: implementare `sample_MN(M, U, V, n_samples, rng) -> np.ndarray` vectorized via `Sigma = np.kron(V, U) + 1e-12 * np.eye(m*n)`, Cholesky, reshape F-order, shape `(n_samples, m, n)`. Include 3 unit test inline (mean convergence, Kronecker cov reconstruction, 1×1 degenerate). Done when: `python -m pytest tests/stress_matrix_gm/mc_forward.py -v` exit 0, 3 tests pass.

- [ ] [iter:1] [agent:gaussian-mixture-expert] [area:tests/infra] `tests/stress_matrix_gm/analytical_ground_truth.py`: funzioni per ogni formula: `agt_extract(M, U, V, i, j)`, `agt_left_affine(A, M, U, V)`, `agt_right_affine(M, U, V, B)`, `agt_sum(M1, U1, V1, M2, U2, V2)`, `agt_transp(M, U, V)`, `agt_isserlis(M1, U1, V1, M2, U2, V2)` (3-term Isserlis formula), `agt_schur_write(M, U, V, i, j, c)`, `agt_tallis_truncate(mu, sigma, a, b)`. Ogni funzione con docstring LaTeX della formula usata. Done when: il file importa senza errori e ha `len([f for f in dir() if f.startswith('agt_')]) == 8`.

### Iter 2 — Grammar battery (≈ 2h)

Prerequisito: Iter 1 completato e verificato.

- [ ] [iter:2] [agent:antlr-grammar-expert] [area:tests/grammar] `tests/stress_matrix_gm/test_grammar_battery.py` parte positivi: `@pytest.mark.parametrize` con 32 input validi divisi nei 7 gruppi A–G (come da Approach). Ogni parametrize entry ha `(group_id, snippet, description)`. Done when: `pytest tests/stress_matrix_gm/test_grammar_battery.py -k "positive" -v` exit 0, 32 tests pass.

- [ ] [iter:2] [agent:antlr-grammar-expert] [area:tests/grammar] Aggiungere 8 negativi N1–N8 (come da tabella Grammar Negatives sopra). Ogni entry: `(neg_id, snippet, production_that_fails)`. Test verifica che `_CountingErrorListener.errors > 0` dopo parsing. Done when: `pytest tests/stress_matrix_gm/test_grammar_battery.py -k "negative" -v` exit 0, 8 tests pass.

- [ ] [iter:2] [agent:antlr-grammar-expert] [area:tests/grammar] Aggiungere 10 edge case E1–E10 (come da tabella Grammar Edge Cases sopra). E6 (`[[-1]]` in list): durante l'implementazione si testa empiricamente; se il parser accetta, il test è `positive`; se rifiuta, `negative`. Il marker xfail NON viene usato per E6 — l'outcome deve essere determinato prima di scrivere il test. Done when: `pytest tests/stress_matrix_gm/test_grammar_battery.py -v` exit 0, esattamente 50 tests pass (0 xfail tra i grammar tests).

### Iter 3 — Operations coverage analytical (≈ 2.5h)

Prerequisito: Iter 2 completato e verificato (50 grammar tests pass).

- [ ] [iter:3] [agent:soga-internal-expert] [area:tests/ops] `tests/stress_matrix_gm/test_ops_analytical.py` parte A — constructor variants P1–P6 + P4b/P4c/P5b/P5c: 6 shapes `matrix_gm` + 4 varianti `matrix_gm_full` (Kron-auto 2×2, Dense 2×2, NearMiss 3×3, Kron-auto 3×3). Ogni test verifica: `E[X[i,j]]` == `M[i,j]` a 1e-10; `Var[X[i,j]]` == `U[i,i]*V[j,j]` a 1e-10; warning/info corretto emesso. Done when: `pytest tests/stress_matrix_gm/test_ops_analytical.py -k "constructor" -v` exit 0, 10 tests pass.

- [ ] [iter:3] [agent:soga-internal-expert] [area:tests/ops] `test_ops_analytical.py` parte B — affine + sum + transpose P7–P17: A@X (2×2, 3×2-rect), X@A (2×2, 3×2-rect), X@Y Isserlis (2×2 C1, 3×3 C3, 2×3 C5), X+Y 2×2, X+N iso 2×2, transp 2×2 e 2×3. Ogni test verifica con `agt_*` dal modulo analytical_ground_truth a tol 1e-6 (exact ops) o 5e-2 (Isserlis NKP). Done when: `pytest tests/stress_matrix_gm/test_ops_analytical.py -k "affine or sum or transp or isserlis" -v` exit 0, 11 tests pass.

- [ ] [iter:3] [agent:gaussian-mixture-expert] [area:tests/ops] `test_ops_analytical.py` parte C — extract + write + observe P18–P26: X[i,j] extract (2×2 Kron, 2×2 Dense, 3×3 Kron, 8×8 Kron con cross-cov), X[i,j]=c Schur 2×2 Kron e Dense, observe(X[i,j]>c) Kron e Dense, observe(X[i,j]==c) delta, observe linear O7. Done when: `pytest tests/stress_matrix_gm/test_ops_analytical.py -k "extract or write or observe" -v` exit 0, 9 tests pass.

### Iter 4 — MC validation + property tests (≈ 2.5h)

Prerequisito: Iter 3 completato e verificato (30 ops tests pass).

- [ ] [iter:4] [agent:numerical-stability-expert] [area:tests/mc] `tests/stress_matrix_gm/test_mc_validation.py` programmi 1–8 (N=1e5, tol: 2% mean, 5% Frob cov): extract scalar 2×2, A@X 2×2, X+N iso, X@Y Isserlis C5-balanced, X[i,j]=c Schur, matrix_gm_full Kron-auto, matrix_gm_full Dense, transp 2×3. Ogni test: sample con `sample_MN`, confronta con SOGA output via `|mean_mc - mean_soga| / max(|mean_soga|, 1e-8)` e `||Cov_mc - Cov_soga||_F / max(||Cov_soga||_F, 1e-8)`. Done when: `pytest tests/stress_matrix_gm/test_mc_validation.py -k "prog_1_to_8" -v` exit 0, 8 tests pass.

- [ ] [iter:4] [agent:numerical-stability-expert] [area:tests/mc] `test_mc_validation.py` programmi 9–13 (N=1e5): observe back-prop Kalman (mini-prog `d`), loop+matrix (mini-prog `a`), branch+matrix (mini-prog `b`), prune+matrix, composite chain (mini-prog `e`). Done when: `pytest tests/stress_matrix_gm/test_mc_validation.py -k "prog_9_to_13" -v` exit 0, 5 tests pass.

- [ ] [iter:4] [agent:numerical-stability-expert] [area:tests/mc] `test_mc_validation.py` programmi 14–15 (8×8 N=1e5, 16×16 N_mean=2e4/N_cov=5e4). Tolleranze da tabella MC Tolerance Derivation (5% mean, 10% Frob per 16×16). Ogni test documenta in un commento: `# tol_mean=5% >= 3/sqrt(2e4)=2.1% (margin 2.4x), tol_frob=10% >= 3*sqrt(2/5e4)=1.9% (margin 5.3x)`. Done when: `pytest tests/stress_matrix_gm/test_mc_validation.py -k "prog_14_or_15" -v` exit 0, 2 tests pass.

- [ ] [iter:4] [agent:test-engineer] [area:tests/properties] `tests/stress_matrix_gm/test_properties_hypothesis.py`: 6 invarianti con `@settings(deadline=None, max_examples=50, suppress_health_check=[HealthCheck.too_slow])`. P5 usa due `@given` separati: (a) Kron case — `MN(0, eps*I_m, eps*I_n)` con `eps > 0` -> `assert dist.cov_kind == "kron"`; (b) Dense case — usa un fixture fisso non-Kron certificato: `Sigma_notKron = FIXED_NON_KRON_4x4` definita in `conftest.py` come matrice 4×4 SPD. La non-Kronecker-ità è verificata analiticamente nel file conftest con un'asserzione `assert` eseguita a module-load time: la condizione necessaria per `Sigma = V⊗U` (con U 2×2, V 2×2) è `Sigma[0,3]*Sigma[1,2] == Sigma[0,2]*Sigma[1,3]`; il fixture è scelto tale che `abs(Sigma[0,3]*Sigma[1,2] - Sigma[0,2]*Sigma[1,3]) > 0.1`, e la SPD-ità è verificata da `assert np.all(np.linalg.eigvalsh(FIXED_NON_KRON_4x4) > 0)`. Il test usa `matrix_gm_full(M2x2, Sigma_notKron)` -> `assert dist.cov_kind == "dense"`. NON usa `_is_kron_factorizable` dalla produzione come oracle. P6 verifica fattori recuperati con tol 1e-10. Done when: `pytest tests/stress_matrix_gm/test_properties_hypothesis.py -v` exit 0, 6 tests pass.

### Iter 5 — Fragile-path xfail probes + report + docs (≈ 1.5h)

Prerequisito: Iter 4 completato e verificato (15 MC tests + 6 property tests pass).

- [ ] [iter:5] [agent:soga-internal-expert] [area:tests/fragile] `tests/stress_matrix_gm/test_fragile_paths.py` con 5 xfail tests `strict=True` (F1, F2a, F2b, F3, F4), ognuno con `raises=<Type>` E `match=<pattern>` (tutti e due obbligatori):
  - **F1 (C1 + Path 5)**: `observe(X[0,0]>0); Y = A@X` — `@pytest.mark.xfail(strict=True, raises=TypeError, match=r"unsupported operand|NoneType", reason="C1+Path5: _matrix_affine_left receives None post-observe+dense sentinel path")`
  - **F2a (C7 right-affine)**: `X = matrix_gm_full(M, Sigma_dense); Y = X @ B` — `@pytest.mark.xfail(strict=True, raises=TypeError, match=r"kron_factors|NoneType", reason="C7: _matrix_affine_right fails on dense sentinel: X@B where X is Dense")`
  - **F2b (C7 left-affine)**: `X = matrix_gm_full(M, Sigma_dense); Y = A @ X` — `@pytest.mark.xfail(strict=True, raises=TypeError, match=r"kron_factors|NoneType", reason="C7: _matrix_affine_left fails on dense sentinel: A@X where X is Dense")`
  *(F2a + F2b replace the merged F2; total fragile probes becomes 5; test count becomes 106)*
  - **F3 (Gap 3)**: `y0 = X[0,0]; observe(X[1,0]>0); check E[y0]` deve differire da prior (Kalman update non avviene) — `@pytest.mark.xfail(strict=True, raises=AssertionError, match=r"Expected updated mean .* but got", reason="Gap3: stale cross-cov, y0 not updated after observe on correlated entry")`
  - **F4 (extract+write)**: `y = X[0,0]; X[0,0] = 5.0; check Cov(y, X[0,0]) != 0` — `@pytest.mark.xfail(strict=True, raises=AssertionError, match=r"Expected Cov\(y, X\[0,0\]\) to be non-zero", reason="F4: cross-cov not propagated through Schur element write")`
  Done when: `pytest tests/stress_matrix_gm/test_fragile_paths.py -v` exit 0, 5 tests XFAIL (F1, F2a, F2b, F3, F4).

- [ ] [iter:5] [agent:test-engineer] [area:tests/report] `generate_report.py` (in `tests/stress_matrix_gm/`): parse `raw.json` da `pytest --json-report`, aggrega per filename, produce `report.md` con tabelle pass/xfail/xpass/fail per categoria + `mc_summary.csv` con colonne `test_id, mean_rel_err, frob_rel_err, tol_mean, tol_frob, pass, is_new_bug`. Il campo `is_new_bug` è `1` se: il test è FAILED (non xfail/xpass), il suo tag `(op, shape, cov_kind, exc_type)` non è in `KNOWN_BUGS = {("observe+affine","2x2","Kron","TypeError"), ("right_affine","2x2","Dense","TypeError"), ("left_affine","2x2","Dense","TypeError"), ("cross_cov_staleness","2x2","Kron","AssertionError"), ("extract_write_crosscov","2x2","Kron","AssertionError")}`. Deduplicazione: stessa tupla 4-element conta come 1 indipendentemente da quanti test la triggerano. `new_bug_count = sum(is_new_bug_per_deduplicated_tuple)` riportato SOLO nel file `report.md` (linea `new_bug_count: N`). `mc_summary.csv` NON include `new_bug_count` nel header — ha esattamente le 7 colonne `test_id, mean_rel_err, frob_rel_err, tol_mean, tol_frob, pass, is_new_bug`. Done when: `python tests/stress_matrix_gm/generate_report.py --input results/qa_stress_test/raw.json --output results/qa_stress_test/` exit 0, `grep "^new_bug_count:" results/qa_stress_test/report.md` exit 0.

- [ ] [iter:5] [agent:documentation-writer] [area:docs] `docs/LIMITATIONS.md` (nuovo file): tabella con bug ID, sintomo, file:linea sorgente approssimativa, workaround consigliato, test che lo espone, riferimento al follow-up plan. Done when: il file esiste, contiene i 4 bug C1/C7/Gap3/F4 in tabella, non modifica alcun file `src/`.

- [ ] [iter:5] [agent:documentation-writer] [area:docs] Aggiungere sezione "Known issues" a `docs/MATRIX_GM_QUICKSTART.md` (se esiste) con link a `LIMITATIONS.md`. Done when: la sezione esiste nel file o il file non esiste (in quel caso skip documentato).

- [ ] [iter:5] [agent:bs-detector] [area:meta] BS-check sull'intero diff. Done when: output BS-detector exit 0 con "GO" (nessun CAT 1–8 trigger).

- [ ] [iter:5] [agent:code-reviewer] [area:meta] Code review pre-merge. Done when: output code-reviewer exit 0 con "GO".

- [ ] [iter:5] [agent:test-engineer] [area:meta] Esecuzione full campagna: `pytest tests/stress_matrix_gm/ -v --json-report --json-report-file=results/qa_stress_$(date +%Y-%m-%d_%H-%M-%S)/raw.json`, poi `generate_report.py`, poi smoke battery `programs/Example/Bernoulli.soga + showcase + advanced`. Done when: pytest exit 0, 106/106 pass (xfail conta come pass, 5 xfail attesi), 0 xpass, report.md generato con `new_bug_count: 0`.

## Test plan
- **Unit**: 50 grammar tests (parametrize), 30 ops analytical tests, 15 MC tests (separate funcs), 6 hypothesis property tests, 5 fragile-path xfail tests (F1 + F2a + F2b + F3 + F4) = **106 test nodes**.
- **Sanity-vs-analytical**:
  - `programs/Example/Bernoulli.soga` → E[theta]=0.25689 (regression)
  - `programs/Example/matrix_gm_showcase.soga` → 5 scalari + 5 matrici come documentati nel file (regression)
  - `programs/Example/matrix_gm_advanced.soga` → E[wcorner]=5.0, E[wd00]=1.0, E[W[1,1]]=1.2876 (regression)
  - 30 ops + 15 MC tests = ground truth analitico/MC con formule da gaussian-mixture-expert memo
- **Benchmark regression**: NO. La campagna non tocca `experiments/` e quindi non richiede `/soga-bench`. La baseline `7d84b1c2` rimane valida.

## Acceptance criteria

Tutti i criteri a-f usano un singolo RUNID fissato prima del run finale: `RUNID=$(date +%Y-%m-%d_%H-%M-%S)`.

- [ ] **a. 106/106 test pass**: `pytest tests/stress_matrix_gm/ --json-report --json-report-file=results/qa_stress_${RUNID}/raw.json -q` exit 0; poi `python3 -c "import json; d=json.load(open('results/qa_stress_${RUNID}/raw.json')); s=d['summary']; assert s.get('passed',0)+s.get('xfailed',0)==106 and s.get('failed',0)==0 and s.get('xpassed',0)==0, s"`
- [ ] **b. 0 xpass**: incluso in criterio a (assert xpassed==0)
- [ ] **c. Regression esistenti**: `pytest tests/ --ignore=tests/stress_matrix_gm/ --json-report --json-report-file=/tmp/regression.json -q` exit 0; poi `python3 -c "import json; d=json.load(open('/tmp/regression.json')); assert d['summary'].get('failed',0)==0, d['summary']"`
- [ ] **d. new_bug_count**: `python3 -c "import re,sys; txt=open('results/qa_stress_${RUNID}/report.md').read(); n=int(re.search(r'new_bug_count: (\d+)',txt).group(1)); sys.exit(0 if n==0 else 1)"` — exit 0 per procedere al commit; se exit 1, convertire bug a xfail prima del commit
- [ ] **e. LIMITATIONS.md**: `python3 -c "txt=open('docs/LIMITATIONS.md').read(); [txt.index(k) for k in ['C1','C7','Gap3','F4']]; print('OK')"` exit 0
- [ ] **f. Nessun src/lib modificato**: `python3 -c "import subprocess; r=subprocess.run(['git','diff','HEAD','--name-only'],capture_output=True,text=True); bad=[l for l in r.stdout.splitlines() if l.startswith('src/lib')]; assert not bad, f'src/lib files modified: {bad}'"` exit 0
- [ ] **g. BS-detector**: `/bs-check` GO sul diff (non-automatizzabile come comando di shell — risultato documentato nel commit message come `BS-check: GO`)
- [ ] **h. Code-reviewer**: `/code-review` GO sul diff (risultato documentato nel commit message come `code-review: GO`)
- [ ] **i. Cross-review**: questo plan ha APPROVE da `codex-cross-reviewer` (Phase F — completato se si è arrivati qui)

## Rollback

**Trigger rollback xfail**: durante l'implementazione emergono N_new > 5 bug nuovi non documentati in Phase D, ognuno richiede > 30 min per il fix.
- "Bug nuovo" è definito come: un test FAILED (non-xfail, non xpass) il cui tag `(op, shape, cov_kind, exc_type)` non è in `KNOWN_BUGS` (come definito nella sezione generate_report.py).
- "Non-triviale" è definito come: il fix richiede modifica di file `src/lib*.py` oltre la riga incriminata (ossia richiede /audit-numerical).
- Misurazione: `generate_report.py` produce `new_bug_count: N` nella prima sezione di `report.md` (non in `mc_summary.csv`). Verifica: `python -c "import re; txt=open('results/qa_stress_<runid>/report.md').read(); n=int(re.search(r'new_bug_count: (\d+)',txt).group(1)); assert n<=5, f'new_bug_count={n} > 5'"`.

Dedup key canonico: `(op, shape, cov_kind, exc_type)` — usato sia in `KNOWN_BUGS` sia in `mc_summary.csv` colonna `is_new_bug` — termine unico, mai `failure_mode`.

**Procedura rollback xfail** (campagna non viene rivertita):
1. Convertire i nuovi test problematici a `xfail(strict=True, raises=..., reason=...)` + aggiungere a LIMITATIONS.md
2. NON revertire la campagna — i test xfail sono comunque valore
3. Aprire issue GitHub `type:blocker` per ogni xfail nuovo con link al test
4. Procedere con merge della campagna, fix in plan separato

**Trigger rollback completo** (regressione su test esistenti):
1. STOP, identificare il commit con `git bisect run pytest tests/ --ignore=tests/stress_matrix_gm/ -q`
2. NON merge della campagna finché la regressione è isolata
3. Issue GitHub `type:bug` con priorità alta

## Estimated complexity
**L (Large)**: 10-12 ore di lavoro effettivo (3-5 ore di codice puro + 3-4 ore di test design + 2-3 ore di debug/iterazione su xfail boundary). Esecuzione test ≤ 90s totale (60s base + 30s per MC 16×16 con N=5e4). Confidence ±30%: l'incognita è quante extra bug emergeranno dai 106 test.

Rischio principale: i 4 fragile-path probes potrebbero passare invece di fallire se ci sono guard non emersi dall'analisi statica. In quel caso xfail diventa xpass → il run fallisce → rimuovere il xfail e il test diventa passing green.

## Cross-review history

| Iter | Tool                      | Verdict               | Data       |
|------|---------------------------|-----------------------|------------|
| 1    | codex-cli (gpt-5.3-codex) | REJECT                | 2026-05-25 |
| 2    | codex-cli (gpt-5.3-codex) | APPROVE_WITH_CHANGES  | 2026-05-25 |
| 3    | codex-cli (gpt-5.3-codex) | APPROVE_WITH_CHANGES  | 2026-05-25 |
| 4    | codex-cli (gpt-5.3-codex) | APPROVE_WITH_CHANGES  | 2026-05-25 |
| 5    | codex-cli (gpt-5.3-codex) | APPROVE_WITH_CHANGES (all changes applied inline — plan APPROVED) | 2026-05-25 |
