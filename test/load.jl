#=
load:
- Julia version: 1.5.1
- Author: Berk
- Date: 2020-09-04
=#
# if ! @isdefined GoML

    using Pkg
    Pkg.activate("test")

    using DataFrames
    using JuMP
    using BARON
    using CPLEX
    using CSV
    # using Gurobi
    # using Ipopt
    using JLD
    using MathOptInterface
    using Test
    using Random

    include("../src/GoML.jl")
    #using Revise
    using .GoML
    global MOI = MathOptInterface
    global BARON_SILENT = with_optimizer(BARON.Optimizer, OutputFlag = 0)
    global CPLEX_SILENT = GoML.CPLEX_SILENT
    # global GUROBI_SILENT = with_optimizer(Gurobi.Optimizer, OutputFlag = 0, Gurobi.Env())
    Random.seed!(1);
    MOI.Silent() = true;

    include(GoML.PROJECT_ROOT * "/test/tools/gams.jl");

    include(GoML.PROJECT_ROOT * "/test/tools/sagemark.jl");

    include(GoML.PROJECT_ROOT * "/test/tools/models.jl")

    include(GoML.DATA_DIR * "speed_reducer.jl")

    include(GoML.DATA_DIR * "oos.jl")

    include(GoML.DATA_DIR * "afpm.jl")

    include(GoML.DATA_DIR * "polynomial.jl")

    include(GoML.BARON_DIR * "gear.jl")

    include(GoML.BARON_DIR * "minlp.jl")

    include(GoML.BARON_DIR * "minlp_demo.jl")

    include(GoML.BARON_DIR * "nlp1.jl")

    include(GoML.BARON_DIR * "nlp2.jl")

    include(GoML.BARON_DIR * "nlp3.jl")

    include(GoML.BARON_DIR * "pool1.jl")

# end