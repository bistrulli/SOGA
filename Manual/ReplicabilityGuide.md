# Replicability Instructions

## Contents of the Package

- The folder `experiments` contains the scripts and data to reproduce the main results of the paper _SOGA: Inference of Probabilistic Programs by Second-order Gaussian Approximation_ (i.e., Table 2, Table 3, Table 4, Table 5, Table 6).
- The folder `grammars` contains the file with the grammar of SOGA (SOGA.g4) and the two sub-grammars ASGMT (ASGMT.g4) and TRUNC (TRUNC.g4).
- The folder `programs` contains the scripts of the models analyzed in the paper, divided by tools and experimental campaigns; in particular, the scripts of the SOGA programs analyzed in the paper can be found in 'programs/SOGA/SensitivityExp'.
- The folder `src` contains the code implementing the tool SOGA, whose usage is described below;
- The folder `tools` is used to collect the implementation of the tools with which SOGA is compared. 
- The folder `jinjaTemplate` contains the templates of the latex files used to reproduced the paper's results 


## Experiments Replication

- For reproducing all the Tables issue the following commands:

```bash
cd /root/SOGA/experimemts
python3 reproduce.py --exp var #Reproduces Table 2
python3 reproduce.py --exp branch #Reproduces Table 3
python3 reproduce.py --exp cmp #Reproduces Table 4
python3 reproduce.py --exp par #Reproduces Table 5
python3 reproduce.py --exp prune #Reproduces Table 6
```

- After executing each command a Table[2-6].pdf file and the corresponding .tex will be generated wihin the folder 

```bash
/root/SOGA/experimemts/results/latexResult/
```

## Implementation Detail

The module `producecfg.py`contains the classes definition for CFG objects and the function produce_cfg, that extracts a CFG from a program script in a .txt file. 

The module `libSOGA.py` contains the function start_soga, which is used to invoke SOGA on a CFG object and the function SOGA, which, depending on the type of the visited node, calls the functions needed to update the current distribution. 

Such functions are contained in the auxiliary modules:
- `libSOGAtruncate.py`, containing functions for computing the resulting distribution when a truncation occurs (in conditional or observe instructions);
- `libSOGAupdate.py`, containing functions for computing the resulting distribution after applying an assignment instruction;
- `libSOGAmerge.py`, containing functions for computing the resulting distribution when a merge or a prune instruction is encountered;

Additional functions for general purpose are defined in the module `libSOGAshared.py`, which is imported by all previous libraries.

Parsing of the scripts, expressions and truncations is performed using ANTLR. Definition of the respective grammars can be found in the files `grammars/SOGA.g4`, `grammars/ASGMT.g4` and `grammars/TRUNC.g4`.