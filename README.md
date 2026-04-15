# DVSG
A Python module to calculate the kinematic disturbance parameter $\Delta V_{\star-g}$ (pronounced 'DVSG').

## Background
Following **Powley et al. (2026)**, a galaxy's $\Delta V_{\star-g}$ value is defined as:

$$
\Delta V_{\star-g} = \frac{1}{N} \sum_{j} \left| {V_{\star,\text{norm}}^{j} - V^{j}_{g,\text{norm}}} \right|
$$

where:
- $V_{\star,\text{norm}}$ is the **normalised stellar velocity map**
- $V_{g,\text{norm}}$ is the **normalised gas velocity map**
- $\sum_{j} \left| {V_{\star,\text{norm}}^{j} - V^{j}_{g,\text{norm}}} \right|$ is the **sum** over all spaxels/bins, $j$, of the **absolute difference** between the normalised stellar and gas velocity maps
- $N$ is the **number of bins/spaxels** contributing towards the sum

A lower $\Delta V_{\star-g}$ value implies greater similarity between a galaxy's stellar and gas kinematics, whereas a higher $\Delta V_{\star-g}$ implies a greater degree of kinematic disturbance. For more detailed information about the steps to calculate $\Delta V_{\star-g}$, please refer to Powley et al. (submitted.).

## Overview

The `calculations` module contains functions to calculate the $\Delta V_{\star-g}$ value of a galaxy.

The `modelling` module contains the `MapModel` class, which can create basic mock data and test how $\Delta V_{\star-g}$ changes when velocity maps are artificially rotated.

The `plotting` module contains functions that can be used to quickly produce science plots from a $\Delta V_{\star-g}$ calculation.

The `helpers` module contains MaNGA-specific loading and bin utilities used by the plateifu-based workflows.

A demonstration of how to use the MaNGA workflow to calculate $\Delta V_{\star-g}$ and visualise the results is provided in `/demo/dvsg_demo.ipynb`.

## Development

If you want to install the development version:

```bash
# Clone the repository
git clone https://github.com/jmpowley/dvsg.git
cd dvsg

# Editable install in a pinned conda environment
python -m pip install -e . --no-deps

# Editable install in a plain pip environment
python -m pip install -e ".[manga]"
```

For local MaNGA development, some packages (e.g. `sdss-marvin`) can require stricter pins when setting up notebook environments. To install `dvsg` with strict pins (e.g. `packaging=20.9`,
`setuptools<81`, `wheel<0.46`), prefer:
```bash
python -m pip install -e /path/to/dvsg --no-deps
```

### Environment variables

Some functions in `helpers` rely on a local installation of MaNGA Data Analysis Pipeline (DAP) products. If you would like to use these functions, you will need to define the environment variables outlined in the [DAP documentation](https://sdss-mangadap.readthedocs.io/en/latest/execution.html#local-environment-setup-for-survey-level-manga-analysis).

## Citation

If you use this code in your research, please cite Powley et al. (2026).

If you use this code in your research, please cite Powley et al. (2026).

```bibtex

@article{2026arXiv260411905P, 
    author = {{Powley}, Jonah M. and {Smethurst}, Rebecca J. and {Lintott}, Chris J. and {G{\'e}ron}, Tobias}, 
    title = "{Introducing $ΔV_{\star-g}$: a new universal kinematic disturbance parameter}", 
    journal = {arXiv e-prints}, 
    pages = {arXiv:2604.11905}, 
    year = {2026},
}
```
