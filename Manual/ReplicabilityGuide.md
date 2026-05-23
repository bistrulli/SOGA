# Replicability Instructions

## Contents of the Package

- The folder `experiments` contains the scripts and data to reproduce the main results of the paper _SOGA: Inference of Probabilistic Programs by Second-order Gaussian Approximation_ (i.e., Table 2, Table 3, Table 4, Table 5, Table 6).
- The folder `grammars` contains the file with the grammar of SOGA (SOGA.g4) and the two sub-grammars ASGMT (ASGMT.g4) and TRUNC (TRUNC.g4).
- The folder `programs` contains the scripts of the models analyzed in the paper, divided by tools and experimental campaigns; in particular, the scripts of the SOGA programs analyzed in the paper can be found in 'programs/SOGA/SensitivityExp'.
- The folder `src` contains the code implementing the tool SOGA, whose usage is described below;
- The folder `tools` is used to collect the implementation of the tools with which SOGA is compared. 
- The folder `jinjaTemplate` contains the templates of the latex files used to reproduced the paper's results 

## Smoke Test
- For veryfing that all the Tables can be reproduced without problems, run the following commands:
```bash
cd /root/SOGA/experiments
python3 reproduce.py --exp var    --smoke   #Smoke test of Table 2
python3 reproduce.py --exp branch --smoke   #Smoke test of Table 3
python3 reproduce.py --exp cmp    --smoke   #Reproduces Table 4
python3 reproduce.py --exp par    --smoke   #Reproduces Table 5
python3 reproduce.py --exp prune  --smoke   #Reproduces Table 6
```

- After executing each command, a Table[2-6].pdf file and the corresponding .tex will be generated within the folder `/root/SOGA/experiments/results/latexResult/`, marking the test as passed. Please note that during the smoke test, SOGA will run experiments with a timeout of 2 seconds, meaning that the produced data will not be consistent with the ones reported in the tool paper, as they serve only to verify that everything is set up and ready for the full evaluation process.

## Experiments Replication

- For reproducing all the Tables issue the following commands:

```bash
cd /root/SOGA/experiments
python3 reproduce.py --exp var    #Reproduces Table 2
python3 reproduce.py --exp branch #Reproduces Table 3
python3 reproduce.py --exp cmp    #Reproduces Table 4
python3 reproduce.py --exp par    #Reproduces Table 5
python3 reproduce.py --exp prune  #Reproduces Table 6
```

- After executing each command a Table[2-6].pdf file and the corresponding .tex will be generated wihin the folder 

```bash
/root/SOGA/experiments/results/latexResult/
```

- The maximum time required to run all the evaluation is ~20h observed when all the experiments reach the timeout of 600s. The expected time is around 10h.

- To copy a generated Table from the container to the host machine, issue the following command

```bash
#Here we assume our goal is to copy Table2.pdf from the container to the host machine
docker cp SOGA:/root/SOGA/experiments/results/latexResult/Table2.pdf ~/Table2.pdf
```
## Implementation Detail

The module `producecfg.py`contains the classes definition for CFG objects and the function produce_cfg, that extracts a CFG from a program script in a .txt file. 

The module `libSOGA.py` contains the function start_soga, which is used to invoke SOGA on a CFG object and the function SOGA, which, depending on the type of the visited node, calls the functions needed to update the current distribution. 

Such functions are contained in the auxiliary modules:
- `libSOGAtruncate.py`, containing functions for computing the resulting distribution when a truncation occurs (in conditional or observe instructions);
- `libSOGAupdate.py`, containing functions for computing the resulting distribution after applying an assignment instruction;
- `libSOGAmerge.py`, containing functions for computing the resulting distribution when a merge or a prune instruction is encountered;

Additional functions for general purpose are defined in the module `libSOGAshared.py`, which is imported by all previous libraries.

`libSOGAtruncate.py` exposes three interchangeable code paths for the inequality truncate, selected at runtime by two module-level flags (`USE_SPARSE_TRUNCATE` and `USE_VECTORIZE_TRUNCATE`, set from `start_SOGA(sparse_truncate=..., vectorize_truncate=...)`, in turn fed by the `--sparse-truncate` and `--vectorize-truncate` CLI flags):

- `_ineq_func_classic` is the default path. It builds a d x d rotation A via `find_basis` (SVD), computes `A * Sigma * A.T`, inverts A, then back-projects, for an overall O(d^3) cost per Gaussian-mixture component.
- `_ineq_func_sparse` performs the same truncation through a rank-1 conditional Gaussian update on the original (mu, Sigma), exploiting the fact that the rotation A is structurally block-sparse (identity on the d-k coordinates where the LBC coefficient vector alpha is zero). Asymptotic cost drops to O(d^2) per component.
- `_ineq_truncate_vectorized_impl` applies the same rank-1 conditional Gaussian update to every component in one batched numpy call. The per-component Python `for` loop in `truncate()` is bypassed entirely; the dispatcher in `truncate()` calls `_truncate_vectorized(dist, trunc_rule)` and returns the aggregated `(norm_factor, new_dist)` directly. Numpy BLAS handles SIMD and multi-core under the hood. Per-component delta-variable masks and the outer loop over aux `gm()` combinations are preserved. The equality counterpart is `_eq_truncate_vectorized_impl`. All three paths are mathematically equivalent: the unit and integration test suite confirms `max |delta mu|, |delta Sigma|` at machine-epsilon level (scaled with sqrt(n_comp) for BLAS accumulation order).

Parsing of the scripts, expressions and truncations is performed using ANTLR. Definition of the respective grammars can be found in the files `grammars/SOGA.g4`, `grammars/ASGMT.g4` and `grammars/TRUNC.g4`.

## Grammar Regeneration (ANTLR 4.10)

The Python parser/lexer files in `src/` are auto-generated from grammars in `grammars/`. **Never hand-edit** `src/*Lexer.py`, `src/*Parser.py`, `src/*Listener.py`, `src/*Visitor.py` — they will be overwritten.

### Pinned version

The ANTLR runtime is pinned to **4.10** in `requirements.txt`:
```
antlr4-python3-runtime==4.10
```
The generator jar must also be 4.10 (`antlr-4.10-complete.jar`). Using a different jar version will produce runtime incompatibilities.

### Regeneration sequence

```bash
# Download the jar if not already present:
# curl -O https://www.antlr.org/download/antlr-4.10-complete.jar

ANTLR_JAR=/path/to/antlr-4.10-complete.jar

# Regenerate all three grammars into src/
java -jar $ANTLR_JAR -Dlanguage=Python3 -visitor -listener grammars/SOGA.g4  -o src/
java -jar $ANTLR_JAR -Dlanguage=Python3 -visitor -listener grammars/ASGMT.g4 -o src/
java -jar $ANTLR_JAR -Dlanguage=Python3 -visitor -listener grammars/TRUNC.g4 -o src/
```

After regeneration, the `grammars/` source files and `src/` generated files must stay in sync. Run `scripts/check_grammar_sync.sh` to verify (added in M1.5 of the matrix-GM integration branch).