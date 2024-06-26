function test_baron_solve(m::JuMP.Model = gear(false))
    set_optimizer(m, BARON_SILENT)
    optimize!(m)
    sol = solution(m)
    @test true
end

function test_speed_params(gm::GlobalModel = minlp(true), solver = CPLEX_SILENT)
    set_optimizer(gm, solver)   
    bbl = gm.bbls[1]
    uniform_sample_and_eval!(bbl)    
    
    # Trying different speed parameters
    ls_num_hyper_restarts = [1, 3]
    ls_num_tree_restarts = [3, 5]
    tree_mat = [[], []]
    time_mat = [[], []]
    for i=1:length(ls_num_hyper_restarts)
        for j=1:length(ls_num_tree_restarts)
            t1 = time()
            params = Dict(:ls_num_hyper_restarts => ls_num_hyper_restarts[i],
                          :ls_num_tree_restarts => ls_num_tree_restarts[j],
                          :max_depth => 2)
            learn_constraint!(bbl; params...)
            push!(time_mat[i], time() - t1)
            push!(tree_mat[i], bbl.learners[end])
        end
    end
    @test true
end

function test_classify_gradients()
    gm = minlp(true)
    bbr = gm.bbls[3]
    uniform_sample_and_eval!(gm)
    idxs = collect(1:10)
    classify_curvature(bbr, idxs)
    @test !any(ismissing.(bbr.curvatures[idxs]))
    classify_curvature(bbr)
    @test all(bbr.curvatures .> 0)
    update_vexity(bbr)
    @test bbr.convex == true
    @test bbr.local_convexity == 1.
end

function test_infeasibility_cuts()
    gm = sagemark_to_GlobalModel(15, false)
    set_param(gm, :ignore_accuracy, true)
    uniform_sample_and_eval!(gm)
    learn_constraint!(gm)
    add_tree_constraints!(gm)
    optimize!(gm)
    bbc_idxs = [bbl isa BlackBoxClassifier for bbl in gm.bbls]
    add_infeasibility_cuts!(gm)
    optimize!(gm)
    while abs(gm.cost[end] - gm.cost[end-1]) > get_param(gm, :abstol)
        add_infeasibility_cuts!(gm)
        optimize!(gm)
    end
    @test true
end

function test_feasibility_sample()
    gm = speed_reducer()
    uniform_sample_and_eval!(gm)
    [set_param(bbl, :threshold_feasibility, 0.3) for bbl in gm.bbls if bbl isa BlackBoxClassifier]
    @test any(check_feasibility(gm) .!= 1)
    feasibility_sample(gm)
    @test all(check_feasibility(gm) .== 1)
end

function test_survey_method(gm::GlobalModel = minlp(true))
    uniform_sample_and_eval!(gm)
    bbrs = [bbl for bbl in gm.bbls if bbl isa BlackBoxRegressor]
    surveysolve(gm)
    bbcs = [bbl for bbl in gm.bbls if bbl isa BlackBoxClassifier]
    bbc_idxs = [x isa BlackBoxClassifier for x in gm.bbls]
    add_infeasibility_cuts!(gm)
    optimize!(gm)
    while abs(gm.cost[end] - gm.cost[end-1]) > get_param(gm, :abstol)
        add_infeasibility_cuts!(gm)
        optimize!(gm)
    end
    @test true
end

function test_concave_regressors(gm::GlobalModel = gear(true))
    gm = gear(true)
    uniform_sample_and_eval!(gm)
    bbrs = [bbl for bbl in gm.bbls if bbl isa BlackBoxRegressor]
    if !isempty(bbrs)
        update_vexity.(bbrs)  
    end
    bbr = bbrs[1]
    
    # Checking number of constraints
    types = JuMP.list_of_constraint_types(gm.model)
    init_constraints = sum(length(all_constraints(gm.model, type[1], type[2])) for type in types)
    surveysolve(gm) # 1st tree (ORT)
    actual = bbr.actuals[end]
    optim = bbr.optima[end]
    learn_constraint!(bbr, "upper" => minimum(bbr.actuals)) # 2nd tree (Upper OCT)
    learn_constraint!(bbr, "reg" => 40, 
                        regression_sparsity = 0, max_depth = 3) # 3th tree (Regressor on upper bounded samples)

    # Trying to add and remove individual constraints in random order to make sure no constraints accidentally remain. 
    for i=1:length(bbr.learners)
        update_tree_constraints!(gm, bbr, i)
        for j=1:length(bbr.learners)
            update_tree_constraints!(gm, bbr, j)
            for k = 1:length(bbr.learners)
                update_tree_constraints!(gm, bbr, k)
                treekeys = collect(keys(bbr.active_trees))
                treevalues = collect(values(bbr.active_trees))
                if length(treevalues) >= 2
                    @test Pair(treevalues[1].first, treevalues[2].first) in GoML.valid_pairs
                elseif length(treevalues) == 1
                    @test treevalues[1].first in GoML.valid_singles || treevalues[1].first == "upper"
                end
                clear_tree_constraints!(gm)
                n_constraints = sum(length(all_constraints(gm.model, type[1], type[2])) for type in JuMP.list_of_constraint_types(gm.model))
                @test n_constraints == init_constraints
            end
        end
    end
    update_tree_constraints!(gm, bbr, 2)
    update_tree_constraints!(gm, bbr, 3)
    @test active_lower_tree(bbr) == 3
    @test active_upper_tree(bbr) == 3  
    optimize!(gm)
    clear_tree_constraints!(gm)
    @test init_constraints == sum(length(all_constraints(gm.model, type[1], type[2])) for type in JuMP.list_of_constraint_types(gm.model))
end

function test_descent()
    gm = minlp(true)
    uniform_sample_and_eval!(gm) # descent requires some samples
    x0 = DataFrame(string.(gm.vars) .=> [1, 0, 1, 0, 1, 0, 5.8])
    append!(gm.solution_history, x0)
    append!(gm.cost, 5.69)
    feas_gap(gm, x0)
    descend!(gm)
    @test isapprox(gm.cost[end], 6.09; atol = 3)

    gm = pool1(true)
    uniform_sample_and_eval!(gm)
    x0 = DataFrame(string.(gm.vars) .=> [4.0, 3.0, 1.0, 4.0, 0, 7, 0])
    append!(gm.solution_history, x0)
    append!(gm.cost, 100)
    feas_gap(gm, x0)
    descend!(gm)
    @test isapprox(gm.cost[end], 23; atol = 3)

    gm = nlp3(true)
    uniform_sample_and_eval!(gm)
    x0 = DataFrame(string.(gm.vars) .=>
    [1728.71, 16000.0, 69.9795, 3056.32,  2000.0,  91.323, 94.7197, 11.5857, 2.26271, 151.159, -1600.81])
    append!(gm.solution_history, x0)
    append!(gm.cost, -1600)
    feas_gap(gm, x0)
    descend!(gm)
    @test isapprox(gm.cost[end], -1161; atol = 4)
end

function test_recipe()
    gm = nlp2(true)
    globalsolve_and_time!(gm)
    @test isapprox(gm.cost[end], 201; atol = 3)
end

test_baron_solve()

test_speed_params()

test_classify_gradients()

test_infeasibility_cuts()

test_feasibility_sample()

test_survey_method()

test_concave_regressors()

test_descent()