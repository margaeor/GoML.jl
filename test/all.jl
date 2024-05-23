#=
all_tests:
- Julia version: 
- Author: Berk
- Date: 2020-07-01
Note: All tests should be run from a julia REPR within the GoML folder, using:
      julia --project=.
      include("test/all.jl")
      
      To see coverage, run with:
      julia --project=. --code-coverage=tracefile-%p.info --code-coverage=user
      include("test/all.jl")
=#


include("load.jl");


@testset "GoML" begin
    include(string(GoML.PROJECT_ROOT, "/test/src.jl"))

    include(string(GoML.PROJECT_ROOT, "/test/imports.jl"))

    include(string(GoML.PROJECT_ROOT, "/test/learners.jl"))

#     include(string(GoML.PROJECT_ROOT, "/test/cbf.jl"))

    include(string(GoML.PROJECT_ROOT, "/test/algorithms.jl"))

#     include(string(GoML.PROJECT_ROOT, "/test/lse.jl"))

end

# Other tests to try later

# include("/test/test_transonic.jl");