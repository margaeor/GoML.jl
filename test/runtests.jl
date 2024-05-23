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

using GoML

N = 5
M = 1

m = JuMP.Model(with_optimizer(CPLEX_SILENT))
@variable(m, x[1:N+1])
@objective(m, Min, x[N+1])

# Initialize GlobalModel
gm = GoML.GlobalModel(model = m, name = "qp")

Q = randn(N, N)
c = randn(N)
expr = :((x) -> -x'*$(Q)*x - $(c)'*x)

lbs = push!(-5*ones(N), -800)
ubs = push!(5*ones(N), 0)
lbs_dict = Dict(x .=> lbs)
ubs_dict = Dict(x .=> ubs)

GoML.bound!(gm, Dict(var => [lbs_dict[var], ubs_dict[var]] for var in gm.vars))
        

# Add constraints
#GoML.add_nonlinear_constraint(gm, expr, vars=x[1:N])
GoML.add_nonlinear_constraint(gm, expr, vars=x[1:N], expr_vars=[x[1:N]])

GoML.globalsolve!(gm)


# include("load.jl");

# @testset "GoML" begin
#     include(string(GoML.PROJECT_ROOT, "/test/src.jl"))

#     include(string(GoML.PROJECT_ROOT, "/test/imports.jl"))

# #     include(string(GoML.PROJECT_ROOT, "/test/cbf.jl"))

#     include(string(GoML.PROJECT_ROOT, "/test/algorithms.jl"))

# #     include(string(GoML.PROJECT_ROOT, "/test/lse.jl"))

# end

# Other tests to try later

# include("/test/test_transonic.jl");