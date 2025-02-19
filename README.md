# GoML


GoML (Global Optimization using Machine Learning) is a [Julia](https://julialang.org/) package that allows for the solution of global optimization problems using mixed-integer (MI) linear and convex approximations. It is an implementation of the methods detailed in this [paper](https://arxiv.org/abs/2311.01742) and submitted to the Journal of Global Optimization (JOGO). GoML is licensed under [the MIT License](https://github.com/margaor/GoML.jl/blob/master/LICENSE). This framework is an extension of [OCHaGOn](https://github.com/1ozturkbe/OCTHaGOn.jl).

GoML relies on the [JuMP.jl](https://github.com/jump-dev/JuMP.jl) modeling language in its backend, and it develops MIO approximations using [Interpretable AI](https://www.interpretable.ai/), with a free academic license. The problems can then be solved by JuMP-compatible solvers, depending on  the type of approximation. GoML's default solver is [Gurobi](https://www.gurobi.com/), which is free with an academic license as well. 


<!-- [Documentation](https://1ozturkbe.github.io/OCTHaGOn.jl/) is available and under development.  
If you have any burning questions or applications, or are having problems with OCTHaGOn, please [create an issue](https://github.com/1ozturkbe/OCTHaGOn.jl/issues)!  -->


# Installation
The framework has been tested on Windows 10 with Julia 1.6.2 and an Interpretable AI version of 2.2 and a Gurobi version of 8. For installations, go to the directory of the repository and run:
```
julia --project=./
```
Once you enter the Julia repl, do:
```
]activate test
```
Then, once inside the Julia package manager do:
```
instantiate 
```

# Running the software
For running the benchmarks detailed in detailed the [paper](https://arxiv.org/abs/2311.01742), do the following:
```
julia --project=./
```
Once you enter the Julia repl, do:
```
]activate test
```
Then, hit backspace to revert back to the julia terminal and do:
```
include("test/benchmarks/allbench.jl")
```
The results will be exported in dump/benchmarks into a CSV format.

# Citation
If you use this package in your work, please use the following citation:
```
﻿@Article{Bertsimas2025,
  author={Bertsimas, Dimitris
  and Margaritis, Georgios},
  title={Global optimization: a machine learning approach},
  journal={Journal of Global Optimization},
  year={2025},
  month={Jan},
  day={01},
  volume={91},
  number={1},
  pages={1-37},
  issn={1573-2916},
  doi={10.1007/s10898-024-01434-9},
  url={https://doi.org/10.1007/s10898-024-01434-9}
}
```
