# Dynamic soft sets as metric paths

Reference implementation and reproducibility code for the paper

> **Dynamic soft sets as metric paths from crisp rigidity and bounded variation to graded differential calculus**

Every number, table and figure in the paper is produced by the code in this repository.
All data are synthetic and generated with fixed random seeds, so a run is fully
deterministic. One command regenerates everything:

```bash
pip install -r requirements.txt
python run_all.py
```

A full run takes about half a minute on a laptop.

## What the code implements

A dynamic soft set is treated as a path in a space of *snapshots* (soft sets with an active
parameter set). The code implements

* the fixed-coordinate snapshot metric `d_alpha` (Definition 4.1) and its graded extension `D_alpha`;
* the union-normalised distance and the counterexample showing that it violates the triangle
  inequality (Proposition 4.4);
* union, restricted intersection and relative complement of snapshots, with their Lipschitz constants
  (Proposition 4.7);
* the sampled-variation profile of Algorithm 1: activation, deactivation, expansion and contraction
  counts, increments, speeds and cumulative variation;
* the duration-sensitive time average (Definition 6.15) and the level snapshots of the layer-cake
  representation (Theorem 6.3);
* the two numerical experiments of Section 8: grid consistency of the sampled variation
  (Experiment A) and change localisation under observation noise (Experiment B).

## Repository layout

```
dss.py                 reference implementation (python dss.py runs a self-test)
verify_examples.py     exact checks of Proposition 4.4, Theorem 4.2, Proposition 4.7,
                       Examples 5.10, 6.18-6.20, 8.1 and 8.2
simulation.py          Experiments A and B (Tables 5 and 6) and Figures 1, 3 and 4
run_all.py             runs the three scripts above and stores their logs in results/
figures/               figures of the paper (600-dpi PNG, as used in the manuscript, and vector PDF)
figures_src/           TikZ source of Figure 2 (flowchart of Algorithm 1)
results/               console logs and simulation_results.json from a reference run
requirements.txt       Python dependencies
```

## Figure and table map

| Paper item | Produced by | Output |
|---|---|---|
| Proposition 4.4, Theorem 4.2, Proposition 4.7 | `verify_examples.py` | `results/verify_examples_output.txt` |
| Examples 5.10, 6.18-6.20, 8.1, 8.2; Table 4 | `verify_examples.py` | `results/verify_examples_output.txt` |
| Tables 5 and 6 | `simulation.py` | `results/simulation_output.txt`, `results/simulation_results.json` |
| Fig. 1 | `simulation.py` | `figures/fig_graded_example.png` |
| Fig. 2 | `figures_src/fig_algorithm_flow.tex` | `figures/fig_algorithm_flow.png` |
| Fig. 3 | `simulation.py` | `figures/fig_grid_consistency.png` |
| Fig. 4 | `simulation.py` | `figures/fig_localisation_trace.png` |

`verify_examples.py` ends with `ALL CHECKS PASSED` and exits with status 0 when every check
holds, and with status 1 otherwise.

## Requirements

Python 3.10 or later with numpy, matplotlib and scikit-learn (`requirements.txt`).
The code has been run with identical results under Python 3.10 (numpy 1.24, matplotlib 3.7,
scikit-learn 1.3) and Python 3.13 (numpy 2.5, matplotlib 3.11, scikit-learn 1.9).

Figure 2 is built with LaTeX (TikZ) and poppler-utils:

```bash
cd figures_src
pdflatex fig_algorithm_flow.tex
pdftoppm -png -r 600 -singlefile fig_algorithm_flow.pdf ../figures/fig_algorithm_flow
```

## License

MIT License; see `LICENSE`.
