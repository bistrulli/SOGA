
## Reusability: how to write your SOGA model
A formal description of SOGA's grammar can be found in the file grammars/SOGA.g4.  
A SOGA model is encoded in a .soga file.

### Data
At the beginning of your file you can declare data. These are arrays that can be accessed but cannot be overwritten by the program. To declare data use the keyword `data` before declaring an array. For example:

`data obs = [0., 1., 0., 0., 1.];`

can be accessed at any point of the program using `obs[i]` where `i` is an integer index (indexing starts from 0).

NOTE: currently SOGA does not support index arithmetic. Arrays and data can only be accessed using a single variable or a number, not expressions such as `i+1`.

### Programs

SOGA supports 5 types of instruction: assignments, conditionals, loops, observe and prune.

In the following:

- `var` is any variable name. You do not need to declare scalar variables in advance, as SOGA infers them automatically when parsing the program. For array variables declare `array[size] var` before using the array, for example `array[10] y;`. Array values are accessed using the usual notation `var[i]` where `i` is an integer index ;

- `const` is either a constant (i.e. a number) or a data value;

- `dist` is a distribution. In SOGA all distributions are approximated by Gaussian Mixtures, which are declared as `gm(pi_list, mu_list, sigma_list)` where `pi_list` is a list of scalar weights summing to 1, `mu_list` is a list of scalar means and `sigma_list` is a list of scalar standard deviations. For example a standard normal distribution can be assigned using `gm([1.], [0.], [1.])` or the shortcut `gauss(0,1)`. Supported primitives for assigning distributions are: `gauss(mu, sigma)`, `bernoulli(p)`, `uniform([a,b], C)`, `beta([a,b], C)`, `laplace(mu, sigma, C)`, `exprnd(lambda, C)` where `C` is the number of components of the approximating mixture and `mu`, `sigma`, `a`, `b`, `lambda` are distribution-specific parameters. 

- `block` is any sequence of instructions.

* Assignments assign a program variable with a value. An assignment can be either be linear or non-linear. A linear assignment has the form

`var_name = const + const*var + ... + const*var;`

A non-linear assignment has the form:

` var_name = const*var*var; `  

At the R.H.S. of an assignment `var` can also be a distribution `dist`. To assign more complex expressions use subsequent assignments. For example:

` z = x + y + 1;`

`z = x*z;`

* Conditionals are expressed by if-then-else statements. The main structure is:

` if bexpr { block } else { block } end if;`

Here `bexpr` is a Boolean expression which can either be of the form:

`const*var + ... + const*var (< | <= | >=| > ) const`

where at the L.H.S. `var` can also be a `dist`, or of the form:

` var == const`.

For example:

`if x == 0 { y = 1; } else { y = 2*x +1; } end if;`

* Loops are expressed by *bounded* for statement. The main structure is:

`for var in range(const) { block } end for;`

For example:

`for i in range(10) { x = obs[i]; } end for;`

* Observe are expressed as

`observe(bexpr)` where `bexpr` is as in the conditional statement, except that it cannot contains `dist`.

For example:

`observe(x + y > 0)`

`observe(y == obs[0])`

* Prune instructions have the effect of pruning the current distribution, which is represented as a Gaussian Mixtures, up to a user-defined number of components. They can be inserted in any point of the program to improve scalability using `prune(K)`.

For example, `prune(30)` will prune the current distribution up to 30 components.

You can find a commented SOGA model in the folder `programs/Example/Bernoulli.soga`. Other SOGA models are available in the folder `programs/SOGA/`

## Performance considerations

SOGA's `truncate` step is the most expensive numerical operation: it runs every time an `observe(...)` or a conditional branch is encountered, once per Gaussian-mixture component. Two opt-in CLI flags provide progressively more aggressive replacements for the default truncate path.

### `--sparse-truncate` (experimental)

The default implementation in `libSOGAtruncate.py` allocates a d x d rotation matrix and inverts it, which costs O(d^3) per component (d is the total number of program variables tracked in the joint distribution).

The `--sparse-truncate` CLI flag activates an alternative implementation that produces the same result (verified to machine precision) by means of a rank-1 conditional Gaussian update, with O(d^2) cost per component.

**When it helps in practice.**
- Models with a moderate-to-large joint dimension (roughly `d > 30`). Below that, fixed per-call overhead dominates and the speedup is small (`~1.1x` on `BayesPointMachine.soga`, `d=9`).
- Models with many `observe` statements: the saving is per-truncate, so it compounds with the number of conditioning events.
- Models with many Gaussian-mixture components (e.g. many branches): the saving is also per component.

**When it does not help.**
- Single-variable random walks (`d=1`, `RandomWalkGauss10.soga` and similar): the rotation is already trivial and there is nothing to save.
- Programs whose total runtime is dominated by ANTLR parsing rather than numerics (typically small `.soga` files with `d <= 10`).

**Correctness.** The two paths are equivalent up to floating-point rounding (`max |delta E[var]| < 1e-15` on the canonical benchmarks). The flag is therefore safe to enable, but the default is `--sparse-truncate` off while we collect soak data.

### `--vectorize-truncate` (experimental)

`--sparse-truncate` reduces the cost per component but leaves intact the Python `for k in range(n_comp)` loop over components. For mixtures with many components (Bernoulli, ClinicalTrial, ClickGraphPrune), that loop becomes the new bottleneck because each Python iteration carries ~50-200 us of function-call overhead.

`--vectorize-truncate` removes the loop. It stacks all components into 3D tensors `(n_comp, d, d)` and applies the rank-1 conditional Gaussian update across the entire mixture in a single numpy call. Numpy BLAS handles SIMD and multi-core under the hood, so the speedup is largest on high-`n_comp` programs.

**Implies `--sparse-truncate`.** The math is identical to the sparse path, just batched.

**A/B benchmark figures (canonical SOGA suite, 21 programs, geometric mean):**

- `--sparse-truncate` over default: **1.51x**
- `--vectorize-truncate` over default: **3.17x**
- `--vectorize-truncate` over `--sparse-truncate`: **2.10x** (median 1.40x, max 12.66x)

**Programs where `--vectorize-truncate` is dramatically faster than `--sparse-truncate`:**

| Program | n_comp | sparse (ms) | vectorize (ms) | speedup |
|---|---|---|---|---|
| `ClinicalTrial.soga` | 5795 | 1547 | 122 | **12.66x** |
| `Bernoulli.soga` | 1954 | 524 | 47 | **11.10x** |
| `RandomWalkUnif10.soga` | 271764 | 52636 | 7274 | **7.24x** |
| `RandomWalkGauss10.soga` | 1024 | 192 | 46 | **4.19x** |
| `SurveyUnbias.soga` | 128 | 41 | 11 | **3.89x** |
| `CoinBias.soga` | 64 | 21 | 8 | **2.70x** |

`RandomWalkUnif10.soga` is the most striking case: the default truncate path times out, the sparse path completes in 52 s, and the vectorized path completes in 7 s. Equivalence holds: `max |delta E[var]|` across the three paths is 2.78e-07.

**When it does not help.**
- Mixtures with `n_comp < 10`: the Python loop overhead it eliminates is small, and the tensor setup adds a fixed cost. `--vectorize-truncate` can be marginally slower than `--sparse-truncate` here (worst measured: 0.95x on BayesPointMachine).

**Example.**

```bash
# default (classic)
python3 src/SOGA.py -f programs/SOGA/Bernoulli.soga

# rank-1 per-component
python3 src/SOGA.py -f programs/SOGA/Bernoulli.soga --sparse-truncate

# batched rank-1 across all components (fastest on this benchmark)
python3 src/SOGA.py -f programs/SOGA/Bernoulli.soga --vectorize-truncate
```

A/B/C benchmark figures across the entire canonical suite are reproducible via the scripts in `experiments/sparse_truncate_prototype_2026-05-20/` (`bench_all_canonical.py` for the 2-way comparison, `bench_all_3way.py` for the 3-way).