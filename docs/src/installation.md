# Installation

As an overview of the installation steps, the current version of GoML works with Julia 1.6.2, with Interpretable AI as its back-end for constraint learning, and Gurobi as its default solver. Please follow the instructions below to get GoML working on your machine. 

## Installing required software

### Julia

Please find instructions for installing Julia on various platforms [here](https://julialang.org/downloads/). GoML is compatible with Julia 1.5 and 1.6, but is frequently tested on Julia 1.6.2, making that the most robust version. However, if you have an existing Julia v1.6.2, we highly recommend you install a clean v1.6.x before proceeding. 

### Interpretable AI

GoML requires an installation of Interpretable AI (IAI) for its various machine learning tools. Different builds of IAI are found [here](https://docs.interpretable.ai/stable/download/), corresponding to the version of Julia used. IAI requires a pre-built system image of Julia to replace the existing image (```sys.so``` in Linux and ```sys.dll``` in Windows machines), thus the need for a clean install of Julia v1.6.x. For your chosen v1.6, please replace the system image with the one you downloaded. Then request and deploy an IAI license (free for academics) by following the instructions [here](https://docs.interpretable.ai/stable/installation/). 

### Gurobi
Gurobi is a mixed-integer optimizer that can be found [here](https://www.gurobi.com/downloads/).

## Quickest build

Once the above steps are complete, we recommend using the following set of commands as the path of least resistance to getting started. 

Navigate to where you would like to put GoML, and call the following commands to instantiate and check all of the dependencies. 

```
git clone https://github.com/margaeor/GoML.jl.git
cd GoML.jl
julia --project=.
using Pkg
Pkg.instantiate()
```

Call the following to precompile all packages and load OCTHaGOn to your environment:

```julia
include("src/GoML.jl")
using .GoML
```

Alternatively, you can test your installation of OCTHaGOn by doing the following in a new Julia terminal:

```julia
using Pkg
Pkg.activate("test")
Pkg.instantiate()
include("test/load.jl")
include("test/src.jl")
```

Please see `test/demos` for some examples of how to use the framework. The experiments paper can be reproduced by running `include("test/benchmarks/allbench.jl")` after `julia --project=./`.
