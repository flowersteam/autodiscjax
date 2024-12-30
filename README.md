# AutoDiscJax

*Autodiscjax* is a python library built on top of [jax](https://jax.readthedocs.io/en/latest/index.html) and [equinox](https://github.com/patrick-kidger/equinox) 
to facilitate automated exploration and simulation of computational models of 
biological processes (such as gene, proteins or metabolites networks).
It provides several already-implemented modules and pipelines to organize experimentation on these biological network pathways using
curiosity-driven learning and exploration algorithms.

## Installation
```
pip install autodiscjax
```

## Why use AutoDiscJax?
AutoDiscJax follows two main design principles:
1) Everything is a *module*, where a module is simply a parametrized function that takes inputs and returns outputs (and log_data). All autodiscjax modules `adx.Module` are implemented as equinox modules `eqx.Module`, which essentially allows to represent the function as a callable PyTree (and hence to be compatible with jax transformations) while keeping an intuitive API for model building (python class with a \__call\__ method). The only add-on with respect to equinox is that when instantiating a `adx.Module`, the user must specify the module's outputs PyTree structure, shape and dtype.
2) An experiment *pipeline* defines (i) how modules interact sequentially and exchange information, and (ii) what information should be collected and saved in the experiment *history*.


AutoDiscJax provides a handful of already-implement modules and pipelines to
1) Simulate biological networks while intervening on them according to our needs
2) Automatically organize experimentation in those systems, by implementing a variety of exploration approaches such as random, optimization-driven and curiosity-driven search
3) Analyze the discoveries of the exploration method, for instance by testing their robustness to various perturbations

Finally, AutoDiscJax takes advantage of JAX mains features (just-in-time compilation, automatic vectorization and automatic differentation) which are especially advantageous for parallel experimentation and computational speedups, as well as gradient-based optimization.

## License
The project is licensed under the MIT license.

## Acknowledgements
AutoDiscJax is inspired by:
- the [auto_disc](https://github.com/flowersteam/adtool/tree/prod/libs/auto_disc) library purpose and structure
  (by the FLOWERS team) 
- the [equinox](https://github.com/patrick-kidger/equinox) library module definition (by Patrick Kidger)

## See Also
Library to parse and convert SBML models into python models written in JAX: [SBMLtoODEjax](https://github.com/flowersteam/sbmltoodejax)


## Citation
```
 @article{Etcheverry_2024, 
 title={AI-driven Automated Discovery Tools Reveal Diverse Behavioral Competencies of Biological Networks}, 
 url={http://dx.doi.org/10.7554/eLife.92683.3}, 
 DOI={10.7554/elife.92683.3}, 
 publisher={eLife Sciences Publications, Ltd}, 
 author={Etcheverry, Mayalen and Moulin-Frier, Clément and Oudeyer, Pierre-Yves and Levin, Michael}, 
 year={2024}}
 ```
