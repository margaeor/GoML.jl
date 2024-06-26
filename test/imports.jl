"""
Makes sure all sage benchmarks import properly.
For now, just doing first 5 out of 25, since polynomial examples are not in R+.
"""
function test_sagemark_to_GlobalModel()
    idxs = 1:5; # 25
    # max_min = [26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38]
    for idx in idxs
        ex = sagemark_to_GlobalModel(idx);
    end
    idx = 1;
    gm = sagemark_to_GlobalModel(idx, false);
    gm_lse = sagemark_to_GlobalModel(idx, true);
    classify_constraints.([gm, gm_lse])

    # Testing proper importing using sagemark #1
    gm = sagemark_to_GlobalModel(1, false);
    gm_lse = sagemark_to_GlobalModel(1, true);

    bounds = Dict(all_variables(gm) .=> [[0.1, 1], [5., 10.], [8., 15.], [0.01, 1.], [-Inf, Inf]])
    @test all(get_bounds(gm)[var] ≈ bounds[var] for var in all_variables(gm))
    inp = Dict(all_variables(gm) .=> [1,1.9,3,3.9, 10.]);
    log_inp = Dict(vk => log(val) for (vk, val) in inp);

    @test gm_lse.bbls[1](log_inp)[1] ≈ gm.bbls[1](inp)[1] ≈ [inp[gm.vars[3]] ^ 0.8 * inp[gm.vars[4]] ^ 1.2][1]

    # Checking GoMLException for sampling unbounded model
    uniform_sample_and_eval!(gm)
    @test true
end

""" Testing that problems are correctly imported, with some random checking. """
function test_gams_to_GlobalModel()
    filenums = [2.15, 2.16, 2.17, 2.18] # [3.2, "3.10", 3.13, 3.15, 3.16, 3.18, 3.25]
    filenames = ["problem" * string(filenum) * ".gms" for filenum in filenums]
    gms = Dict()
    for filename in filenames
        try
            gm = GAMS_to_GlobalModel(GoML.GAMS_DIR, filename)
#             println("    Problem NL constraints: " * string(length(gm.bbls)))
            types = JuMP.list_of_constraint_types(gm.model)
            if !isempty(types)
                total_constraints = sum(length(all_constraints(gm.model, type[1], type[2])) for type in types)
#                 println("    Problem total constraints: " * string(total_constraints))
            end
        catch
            throw(GoMLException(filename * " has an import issue."))
        end
    end

    # Testing problem import.
    filename = "problem3.13.gms"
    gm =  GAMS_to_GlobalModel(GoML.GAMS_DIR, filename)
    x = gm.model[:x]
    @test length(gm.vars) == 8
    @test all(bound == [0,100] for bound in values(get_bounds(Array{JuMP.VariableRef}(flat(gm.model[:x])))))
    @test length(gm.bbls) == 1
end

# function test_load_fits()
#     gm = sagemark_to_GlobalModel(3, false)
#     set_optimizer(gm, CPLEX_SILENT)
#     # Testing finding bounds of bounded model
#     @test isnothing(get_unbounds(gm.bbls))
#     @test isnothing(find_bounds!(gm))
#     # Test reloading
#     load_fit(gm)
#     @test all([get_param(bbl, :reloaded) == true for bbl in gm.bbls])
#     add_tree_constraints!(gm)
#     optimize!(gm) 
#     @test gm.cost[end] <= -120
# end

test_sagemark_to_GlobalModel()

test_gams_to_GlobalModel()

# test_load_fits()