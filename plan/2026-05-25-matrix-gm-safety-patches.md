# Plan: matrix-gm-safety-patches

## Goal
Convertire i 4 bug latenti documentati in `docs/LIMITATIONS.md` (C1, C7, Gap3, F4) da silent-fail/silent-wrong a loud-fail/loud-warning prima della consegna allo studente. Patch β+γ (Opzione δ): aggiungere `NotImplementedError` guards su dense-sentinel nei 4 affine ops (γ) e `StaleCrossCovWarning` runtime in observe/element-write (β). NO fix matematico dei bug — solo segnali runtime + LIMITATIONS.md aggiornato.

## Context
Il branch `feat/matrix-gm-integration` ha 4 bug latenti documentati:
- **C1** (`observe → A@X`): subprocess timeout silenzioso
- **C7** (`matrix_gm_full dense → affine`): AttributeError/ValueError casuale
- **Gap3** (`extract → observe correlato`): silent-wrong su scalar moment
- **F4** (`extract → element write`): silent-wrong su cross-cov

I primi due (C1/C7) sono "safe-fail" già visibili. Gli ultimi due (Gap3/F4) sono **silent-wrong** — SOGA risponde con valori sbagliati senza warning. Per uno studente che esplora liberamente, sono il rischio reale.

Opzione δ (β+γ) trasforma i silent-wrong in loud-warning e i crash casuali in NotImplementedError espliciti, senza fixare la matematica. Costo ~3-5h vs ~6-10h per il fix completo.

Dopo questa opzione, l'esperimento Lishan può procedere su una superficie matrix-GM "loud" — non più "silent".

## Constraints
- **Pre-flight obbligatorio**: `/audit-numerical` (anche se gli esperti confermano "zero effetto FP", è policy CLAUDE.md §11)
- Modifica `src/libMatrixUpdate.py`, `src/libMatrixTruncate.py`, `src/libMatrixGaussian.py` (zona numerical kernel)
- NO modifica di `src/libSOGA*.py`, `grammars/*.g4`, `src/{SOGA,ASGMT,TRUNC}*.py` auto-generated
- 106 stress test devono passare verde, **0 xfail residui** (migrate to passing)
- 385 test esistenti continuano a passare invariati
- `Bernoulli E[theta]=0.25689` invariato
- `showcase E[y00]=2.57684`, `advanced E[wcorner]=5.0` invariati
- `LIMITATIONS.md` aggiornato per riflettere i nuovi runtime signals
- NO push finchè 100% verde + Codex APPROVE
- Master seed 42

## Approach

### Tre artefatti core

**1. Warning class** (`src/libMatrixGaussian.py`)
```python
class StaleCrossCovWarning(UserWarning):
    """Scalar variable extracted before an observe/write on a correlated
    matrix variable; its moments are now stale (Gap3/F4)."""
```

**2. Four guards** (`src/libMatrixUpdate.py`, ognuno ~2 righe)
Posizionati immediatamente dopo la riga esistente `U_x, V_x = block.get_cov(k, lhs, lhs)` in ciascuna funzione:
```python
if U_x is None:
    raise NotImplementedError(f"[C1/C7] left-affine (A@X) on dense-sentinel covariance not implemented. Variable '{lhs}' was densified by observe() or matrix_gm_full(dense Sigma). See docs/LIMITATIONS.md §C1/C7.")
```
- `_matrix_affine_left` (inserire dopo riga 271)
- `_matrix_affine_right` (inserire dopo riga 285): messaggio "right-affine (X@B)"
- `_matrix_scale` (inserire dopo riga 353): messaggio "scale (c*X)"
- `_matrix_transpose` (inserire dopo riga 366): messaggio "transpose (X^T)"

NON re-fetchare da `cov_blocks` — usare il valore `U_x` già estratto da `get_cov`.

**3. Two warning emission points**
- `src/libMatrixTruncate.py:_truncate_matrix_element_ineq` (riga ~397, prima del loop componenti): per ogni scalar `s` in `block.var_list` con `cov_blocks[0].get(frozenset({s, mat_var}))` non-None e `np.linalg.norm > 1e-8`, emit `StaleCrossCovWarning(f"Scalar '{s}' has non-zero cross-cov with '{mat_var}' (norm={n:.2e}); observe on '{mat_var}' will not update '{s}'.")`
- `src/libMatrixUpdate.py:_matrix_element_write_component` (riga ~871, dopo densify Sigma): idem per `mat_name`

### Migrazione test (strategia B unanime dagli esperti)
Convertire tutti i 5 xfail di `tests/stress_matrix_gm/test_fragile_paths.py` a **passing tests** che documentano il contratto post-patch:
- **F1**: subprocess test. **SOGA.py architecture note**: il worker Process crasha con NotImplementedError (traceback in stderr), ma il processo principale aspetta il timeout di queue.get() e poi esegue l'except queue.Empty → stampa "Warning: SOGA Timeout occurred" → exit rc=0. Quindi: assert `rc == 0` (NOT rc!=0) + `"NotImplementedError" in (stdout + stderr)` + `"LIMITATIONS" in (stdout + stderr)`. Usare `timeout=5, -t 3` per ridurre latenza CI da 8s a 3s.
- **F2a, F2b**: in-process `pytest.raises(NotImplementedError, match=r"...")` direttamente su `_matrix_affine_right`/`_matrix_affine_left`
- **F3**: convertire da subprocess a in-process; `pytest.warns(StaleCrossCovWarning, match=r"y0|cross.cov")` + assert `E[y0]` stale (lock current behavior)
- **F4**: già in-process; aggiungere `pytest.warns(StaleCrossCovWarning)` + mantenere assert su cross-cov stale

Risultato: 106 passing, 0 xfail, 0 xpass.

## Alternatives considered

- **Alt 1: Solo γ (no warnings)**. Pro: ~30 min lavoro. Contro: lascia Gap3/F4 silent-wrong → rischio per lo studente non mitigato. Rigettato perché il rischio reale è proprio sui silent-wrong.

- **Alt 2: Solo β (no guards)**. Pro: ~1h lavoro. Contro: C1/C7 continuano a produrre timeout/AttributeError casuali — diagnosi pessima per studente. Rigettato perché crash chiari sono parte essenziale dell'esperienza dev.

- **Alt 3: Fix completo (Opzione ε)**. Pro: codebase pulito, niente più bug. Contro: 6-10h lavoro + plan dedicato, possibili regressioni numeriche (modifica logica Kalman propagation e Schur update). Rigettato per ora — ε è il prossimo plan, β+γ è il prerequisito difensivo.

- **Alt 4: Strategia A test (xfail con raises aggiornato)**. Pro: minor change. Contro: xfail+warns insieme è strano (test-engineer e gaussian-mixture-expert confermano). Rigettato in favore di B.

## Sub-tasks (atomic)

### Iter 1 — Pre-flight + scaffolding (≈ 30 min)

Prerequisito: nessuno.

- [ ] [iter:1] [agent:numerical-stability-expert] [area:audit] `/audit-numerical` baseline su `src/libSOGA*.py` (scope originale). Salva report in `results/audit_numerical_baseline_<runid>.md`. Done when: file generato, verdetto 🟢 GO.
- [ ] [iter:1] [agent:numerical-stability-expert] [area:audit] Snapshot regressione baseline: run 106 stress + 385 esistenti + smoke (Bernoulli, showcase, advanced); salva expected values in `results/baseline_<runid>.txt`. Done when: tutti pass, valori canonici loggati.
- [ ] [iter:1] [agent:gaussian-mixture-expert] [area:src] Aggiungere `StaleCrossCovWarning(UserWarning)` class in `src/libMatrixGaussian.py` accanto alle altre warning classes (~riga 80). Aggiungere docstring breve. Done when: `python3 -c "from libMatrixGaussian import StaleCrossCovWarning; print(StaleCrossCovWarning.__mro__)"` (cwd=src) emette tupla con UserWarning.

### Iter 2 — γ guards (≈ 1h)

Prerequisito: Iter 1 completato.

- [ ] [iter:2] [agent:soga-internal-expert] [area:src/libMatrixUpdate] Aggiungere guard in `_matrix_affine_left` (riga 268-272). Pattern: **immediatamente dopo** la riga `U_x, V_x = block.get_cov(k, lhs, lhs)` già esistente, aggiungere `if U_x is None: raise NotImplementedError(f"[C1/C7] left-affine (A@X) on dense-sentinel covariance not implemented. Variable '{lhs}' was densified by observe() or matrix_gm_full(dense Sigma). See docs/LIMITATIONS.md §C1/C7.")`. NON ri-fetchare da cov_blocks — usa il valore già estratto dal get_cov. Done when: `python3 -c "import libMatrixUpdate; print('ok')"` (cwd=src) exit 0; `grep -n NotImplementedError src/libMatrixUpdate.py | wc -l` >= 1 sito.
- [ ] [iter:2] [agent:soga-internal-expert] [area:src/libMatrixUpdate] Aggiungere guard identico in `_matrix_affine_right` (riga 282-286): `if U_x is None: raise NotImplementedError(...)` dopo `U_x, V_x = block.get_cov(k, lhs, lhs)` a riga 285. Messaggio "right-affine (X@B)". Done when: idem; grep restituisce >= 2.
- [ ] [iter:2] [agent:soga-internal-expert] [area:src/libMatrixUpdate] Aggiungere guard in `_matrix_scale` (riga 350-354): `if U_x is None: raise NotImplementedError(...)` dopo `U_x, V_x = block.get_cov(k, lhs, lhs)` a riga 353. Messaggio "scale (c*X)". Done when: idem; grep restituisce >= 3.
- [ ] [iter:2] [agent:soga-internal-expert] [area:src/libMatrixUpdate] Aggiungere guard in `_matrix_transpose` (riga 363-367): `if U_x is None: raise NotImplementedError(...)` dopo `U_x, V_x = block.get_cov(k, lhs, lhs)` a riga 366. Messaggio "transpose (X^T)". Done when: idem; grep restituisce >= 4.
- [ ] [iter:2] [agent:test-engineer] [area:tests/stress] Migrare F2a/F2b da `xfail` a passing in `test_fragile_paths.py`: rimuovi marker `@pytest.mark.xfail`, sostituisci body con `pytest.raises(NotImplementedError, match=r"(?i)dense.sentinel|C1/C7")` direttamente su `_matrix_affine_right`/`_matrix_affine_left`. Done when: `.venv/bin/pytest tests/stress_matrix_gm/test_fragile_paths.py::test_F2a -v` e `::test_F2b` exit 0, PASSED (non XFAIL).
- [ ] [iter:2] [agent:test-engineer] [area:tests/stress] Migrare F1 da `xfail` a passing: subprocess test. **ATTENZIONE meccanismo SOGA**: quando il worker Process crasha con NotImplementedError, il traceback appare in stderr (il worker eredita stderr del processo padre), ma il processo principale aspetta il timeout di queue.get(), poi stampa "Warning: SOGA Timeout occurred" ed esce con **rc=0** (nessun sys.exit() in SOGA.py). Quindi l'assert corretto è: `assert rc == 0` + `assert "NotImplementedError" in (stdout + stderr)` + `assert "LIMITATIONS" in (stdout + stderr) or "dense" in (stdout + stderr).lower()`. NON usare `assert rc != 0`. Nota: il test impiega ~8s (timeout queue). Considerare `timeout=5, -t 3` nella chiamata _run_soga per ridurre latenza CI. Done when: `pytest ... ::test_F1` exit 0, PASSED.

### Iter 3 — β warnings (≈ 1h)

Prerequisito: Iter 2 completato + 106 stress + 385 esistenti pass.

- [ ] [iter:3] [agent:gaussian-mixture-expert] [area:src/libMatrixTruncate] In `_truncate_matrix_element_ineq` (riga ~397), **aggiungere `from libMatrixGaussian import StaleCrossCovWarning` agli import** (verificare assenza di import circolare: libMatrixGaussian importa solo numpy — nessun ciclo), poi prima del loop componenti, scorrere `block.cov_blocks[0]` per chiavi `frozenset({s, mat_var})` con `s in block.var_list` (scalari) e `np.linalg.norm(cov_vec) > 1e-8` → emit `warnings.warn(StaleCrossCovWarning(f"Scalar '{s}' has non-zero cross-cov with '{mat_var}' (norm={n:.2e}); observe on '{mat_var}' will not update '{s}'. Value of '{s}' remains STALE. See docs/LIMITATIONS.md §Gap3."), StaleCrossCovWarning, stacklevel=3)`. Done when: `grep -n StaleCrossCovWarning src/libMatrixTruncate.py` exit 0 (almeno 1 site).
- [ ] [iter:3] [agent:gaussian-mixture-expert] [area:src/libMatrixUpdate] In `_matrix_element_write_component` (riga ~871), dopo la riga `block.cov_blocks[k][frozenset({mat_name})] = (None, Sigma_new)` (riga 899), **usare `from libMatrixGaussian import StaleCrossCovWarning`** (già importabile — libMatrixUpdate importa già da libMatrixGaussian per le altre warning classes? verificare; se no, aggiungere import), idem per F4: scorrere `block.cov_blocks[k]` per chiavi `frozenset({s, mat_name})` con `s in block.var_list` e `np.linalg.norm(cov_vec) > 1e-8` → emit `StaleCrossCovWarning`. Done when: `grep -n StaleCrossCovWarning src/libMatrixUpdate.py` exit 0.
- [ ] [iter:3] [agent:test-engineer] [area:tests/stress] Migrare F3 da subprocess+xfail a in-process+passing. **Il pattern NON è identico a F4**: F3 testa la truncate path. Costruire `GaussianMixBlock` con `X ~ MN(M, U_offdiag, V)`, aggiungere `y0` come scalare con cross-cov in `cov_blocks[0][frozenset({"y0","X"})]`, poi chiamare `from libMatrixTruncate import _truncate_matrix_element_ineq; _truncate_matrix_element_ineq(block, "X", 1, 0, 0.0, "gt", m=2, n=2)` (argomenti esatti: verificare firma). Usare `with pytest.warns(StaleCrossCovWarning, match=r"y0|cross.cov")` + assert `abs(block.mu_blocks[0]["y0"][0] - prior_mean) < tol` (valore stale: E[y0] NON aggiornato) AND `E[X[0,0]]_post > prior_mean + 0.3` (Kalman shift su X). Done when: `pytest ... ::test_F3 -v` PASSED.
- [ ] [iter:3] [agent:test-engineer] [area:tests/stress] Migrare F4 da xfail a passing: aggiungi `with pytest.warns(StaleCrossCovWarning):` attorno alla chiamata `_matrix_element_write_component(...)`; **invertire la assert finale**: il valore cross-cov È ANCORA STALE (nessun fix matematico), quindi assert `abs(cross_cov_y_X00_after) > 1e-10` (il valore rimane sbagliato — questo è il contratto documentato del bug). Aggiungere commento: "Warning is emitted but value is intentionally NOT fixed (see LIMITATIONS.md §F4). This test locks the stale-but-warned behavior." Done when: `pytest ... ::test_F4 -v` PASSED.

### Iter 4 — Regression + audit post-patch + docs (≈ 45 min)

Prerequisito: Iter 3 completato.

- [ ] [iter:4] [agent:test-engineer] [area:meta] Full pytest run: `pytest tests/stress_matrix_gm/ -v` deve dare 106 passed, 0 xfail, 0 xpass. Done when: pytest exit 0, count corretto.
- [ ] [iter:4] [agent:test-engineer] [area:meta] Regression sui 385 esistenti: `pytest tests/ --ignore=tests/stress_matrix_gm/ -q` exit 0, 385 passed. Done when: idem.
- [ ] [iter:4] [agent:test-engineer] [area:meta] Smoke battery: Bernoulli E[theta]=0.25689, showcase E[y00]=2.57684 + E[xfull00]=5.0, advanced E[wcorner]=5.0 + E[wd00]=1.0. Done when: tutti i valori invariati.
- [ ] [iter:4] [agent:numerical-stability-expert] [area:audit] `/audit-numerical` post-patch su scope esteso (libMatrix* incluso). Salva in `results/audit_numerical_postpatch_<runid>.md`. Done when: verdetto 🟢 GO, nessuna regressione vs baseline.
- [ ] [iter:4] [agent:documentation-writer] [area:docs] Aggiornare `docs/LIMITATIONS.md`: per ogni bug (C1, C7, Gap3, F4), aggiungere campo "Runtime signal" che indica il nuovo comportamento (NotImplementedError / StaleCrossCovWarning). Aggiungere nota in cima: "These bugs are now exposed at runtime via clean errors/warnings (commit <SHA>). Workarounds remain documented below." Done when: il file contiene "Runtime signal" o equivalente in ogni riga della tabella.
- [ ] [iter:4] [agent:bs-detector] [area:meta] BS-check sull'intero diff dal commit base di questo plan. Done when: output BS-detector 🟢 GO.
- [ ] [iter:4] [agent:code-reviewer] [area:meta] Code review pre-merge sul diff. Done when: 🟢 GO.

### Iter 5 — Cross-review + commit (≈ 15 min)

Prerequisito: Iter 4 completato.

- [ ] [iter:5] [agent:codex-cross-reviewer] [area:meta] Cumulative diff cross-review (`checklist=code`). Done when: APPROVE in ≤ 5 sub-iter.
- [ ] [iter:5] [agent:test-engineer] [area:meta] Final commit incrementale (1 commit per iter group: scaffolding, γ, β, tests, docs). Done when: 4-5 commit creati con `Refs #<plan-issue>`.

## Test plan
- **Unit (esistenti)**: 385 test in `tests/` (exclusi stress) — devono tutti passare invariati.
- **Stress (esistenti, migrati)**: 106 in `tests/stress_matrix_gm/` — di cui 5 (F1-F4) migrati da xfail a passing.
- **Sanity-vs-analytical**:
  - Bernoulli E[theta]=0.25689 (regression)
  - showcase E[y00]=2.57684, E[xfull00]=5.0
  - advanced E[wcorner]=5.0, E[wd00]=1.0
- **Audit numerical**: pre-flight (Iter 1) + post-patch (Iter 4) — Phase 1-5 di `/audit-numerical`.
- **Benchmark regression**: NO. La campagna stress test serve da regression bar; Table3 non è impattato (cambiamenti solo in libMatrix*).

## Acceptance criteria

- [ ] **a. Test count**: `pytest tests/stress_matrix_gm/ -q` → `106 passed, 0 xfailed, 0 xpassed, 0 failed`
- [ ] **b. Regression esistenti**: `pytest tests/ --ignore=tests/stress_matrix_gm/ -q` → `385 passed`
- [ ] **c. Smoke battery**: Bernoulli/showcase/advanced producono i valori canonici invariati (5 decimali)
- [ ] **d. Audit numerical 🟢 GO**: report Phase 1-5 in `results/audit_numerical_postpatch_<runid>.md`
- [ ] **e. NotImplementedError raised**: `grep -c "NotImplementedError" src/libMatrixUpdate.py` >= 4 (4 guards inseriti)
- [ ] **f. StaleCrossCovWarning emitted**: `grep -c "StaleCrossCovWarning" src/libMatrix*.py` >= 3 (1 def + 2 emit sites)
- [ ] **g. LIMITATIONS.md aggiornato**: contiene "Runtime signal" o equivalente per ogni bug (C1, C7, Gap3, F4)
- [ ] **h. No xfail rimasti**: `pytest tests/stress_matrix_gm/ -v --tb=no 2>&1 | grep -c " XFAIL"` == 0 (oppure: `pytest tests/stress_matrix_gm/ -v --tb=no 2>&1 | grep "XFAIL" | wc -l` deve dare 0). Non usare `--collect-only | grep -c xfail` — `grep -c` ritorna 1 anche con 0 match (per la riga "0 selected").
- [ ] **i. BS-check**: 🟢 GO sul diff cumulato
- [ ] **j. Code-review**: 🟢 GO
- [ ] **k. Codex cross-review**: APPROVE su questo plan (Phase F) E sul diff finale (Iter 5)

## Rollback

**Trigger rollback**: regressione su almeno uno tra (a) i 385 test esistenti, (b) i valori canonici di Bernoulli/showcase/advanced, (c) verdetto 🔴 STOP dall'audit-numerical post-patch.

**Procedura**:
1. `git revert <SHA-iter4-last> ... <SHA-iter2-first>` in ordine inverso (dal commit più recente al più antico). **NON usare** `git revert A..B` — la sintassi range esclude il commit A; usare invece `git log --oneline iter2-first^..iter4-last` per ottenere la lista e revertire uno per uno, oppure `git revert HEAD~N..HEAD` se i commit sono consecutivi in cima al branch. (revert dei commit di γ/β/test)
2. Verificare ritorno allo stato pre-patch: 101 passing + 5 xfailed = 106 (numerologia originale)
3. Aprire issue GitHub `type:bug` con la regressione documentata
4. Rivedere plan: nuova analisi su quale guard/warning ha causato la regressione

**Non-rollback** (errori minori durante implementazione): se un singolo sub-task fallisce ma altri proseguono, fix in-place senza revert globale. Solo i 3 trigger sopra giustificano revert.

## Estimated complexity
**M (Medium)**: 3-5 ore di lavoro effettivo:
- Iter 1: 30 min (audit baseline + scaffolding)
- Iter 2: 1h (4 guards + 3 test migrations)
- Iter 3: 1h (2 warning emissions + 2 test migrations)
- Iter 4: 45 min (regression + audit + docs + reviews)
- Iter 5: 15 min (cross-review + commit)
- Buffer per imprevisti: 30 min

Esecuzione test ≤ 2 minuti totale (90s stress + 30s smoke + 100s regression).

Confidence ±20%: i memo degli esperti sono dettagliati al livello "riga di codice", riducendo le ambiguità. F1 subprocess: il NotImplementedError ARRIVERÀ in stderr (il worker eredita stderr del padre senza redirect), ma rc sarà 0 non 1 (meccanismo queue timeout). Questo è ora documentato e l'assert corretto è rc==0+grep stderr. Il fallback "xfail F1" non è più necessario.

## Cross-review history

### Iter 1 — 2026-05-25

Reviewer: codex-cross-reviewer (claude-self-review fallback — codex-cli exec did not produce output)
Verdict: APPROVE_WITH_CHANGES
Critical findings resolved:
1. F1 rc assertion corrected (rc==0 not rc!=0)
2. Guard pattern simplified (if U_x is None, no cov_blocks re-fetch)
3. Rollback syntax corrected (reverse-order, not A..B range)
4. StaleCrossCovWarning import chain added to Iter 3 sub-tasks

### Iter 2 — 2026-05-25

Reviewer: codex-cross-reviewer (claude-self-review fallback)
Verdict: APPROVE
All 4 required changes verified resolved. 2 optional items noted (non-blocking).
Audit trail: results/codex_review/2026-05-25-safety-patches/
