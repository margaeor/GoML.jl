""" Finds the linear min/max bounds of JuMP.VariableRefs."""
function find_linear_bounds!(gm::GlobalModel; bbls::Array{BlackBoxLearner} = gm.bbls, all_bounds::Bool = false)
    unbounds = get_unbounds(bbls)
    if all_bounds
        unbounds = get_bounds(bbls)
    end
    if isnothing(unbounds)
        return
    end
    clear_tree_constraints!(gm)
    m = gm.model
    orig_objective = JuMP.objective_function(m)
    new_bounds = copy(unbounds)
    @showprogress 0.5 "Finding bounds..." for var in collect(keys(unbounds))
        if isinf(unbounds[var][1]) || all_bounds
            @objective(m, Min, var);
            JuMP.optimize!(m);
            if termination_status(m) == MOI.OPTIMAL
                new_bounds[var][1] = getvalue(var);
            end
        end
        if isinf(unbounds[var][2]) || all_bounds
            @objective(m, Max, var);
            JuMP.optimize!(m);
            if termination_status(m) == MOI.OPTIMAL
                new_bounds[var][2] = getvalue(var);
            end
        end
    end
    # Revert objective, bounds and return new_bounds
    @objective(m, Min, orig_objective)
    bound!(gm, new_bounds)
    return new_bounds
end

"""
    find_bounds!(gm::GlobalModel; bbls::Array{BlackBoxLearner} = [], all_bounds::Bool = false)

Finds the outer variable bounds of GlobalModel by solving only over the linear constraints
and listed bbls.
TODO: improve! Only find bounds of non-binary variables.
"""
function find_bounds!(gm::GlobalModel; bbls::Array{BlackBoxLearner} = gm.bbls, all_bounds::Bool = false)
    linear_bounds = find_linear_bounds!(gm, bbls = bbls, all_bounds = all_bounds)
    return linear_bounds
end

""" 
    ridge_regress(X::DataFrame, Y::Array; solver = CPLEX_SILENT, rho::Float64 = 0., weights = ones(length(Y)))

Performs ridge regression on data. 
"""
function ridge_regress(X::DataFrame, Y::Array; solver = CPLEX_SILENT, rho::Float64 = 0., weights = ones(length(Y)))
    m = JuMP.Model(with_optimizer(solver))
    normalized_X, lbs, ubs = normalized_data(X);
    @variable(m, x[1:size(X,2)])
    @variable(m, offset)
    @objective(m, Min, sum((Y - normalized_X*x .- offset).^2) + rho*sum(x.^2))
    status = optimize!(m)
    return getvalue(offset), getvalue.(x)./(ubs-lbs)
end

""" 
    u_regress(X::DataFrame, Y::Array; solver = CPLEX_SILENT)

Finds upper regressors of data that are conservative. 
"""
function u_regress(X::DataFrame, Y::Array; solver = CPLEX_SILENT)
    if size(X, 1) < 2*size(X, 2)
        @warn("Upper regression doesn't have enough data, thus returning constant bounds. ")
        return maximum(Y), zeros(size(X,2))
    end
    m = JuMP.Model(with_optimizer(solver))
    @variable(m, α[1:size(X, 2)])
    @variable(m, α0)
    @variable(m, err[1:length(Y)] >= 0)
    @constraint(m, [i=1:length(Y)], sum(Array{Float64}(X[i, :]).* α) + α0 >= Y[i])
    @constraint(m, err .>= (Matrix(X) * α .+ α0 - Y))
    @constraint(m, err .>= -(Matrix(X) * α .+ α0 - Y))
    @objective(m, Min, sum(err.^2))
    try
        optimize!(m)
        return getvalue(α0), getvalue.(α)
    catch
        @warn("Infeasible u_regress, returning constant bounds.")
        return maximum(Y), zeros(size(X,2))
    end
end

""" 
    l_regress(X::DataFrame, Y::Array; solver = CPLEX_SILENT)

Finds lower regressors of data that are conservative. 
"""
function l_regress(X::DataFrame, Y::Array; solver = CPLEX_SILENT)
    if size(X, 1) < 2*size(X, 2)
        @warn("Lower regression doesn't have enough data, thus returning constant bounds. ")
        return minimum(Y), zeros(size(X,2))
    end
    m = JuMP.Model(with_optimizer(solver))
    @variable(m, β[1:size(X, 2)])
    @variable(m, β0)
    @variable(m, err[1:length(Y)] >= 0)
    @constraint(m, [i=1:length(Y)], sum(Array{Float64}(X[i, :]).* β) + β0 <= Y[i])
    @constraint(m, err .>= (Matrix(X) * β .+ β0 - Y))
    @constraint(m, err .>= -(Matrix(X) * β .+ β0 - Y))
    @objective(m, Min, sum(err.^2))
    try
        optimize!(m)
        return getvalue(β0), getvalue.(β)
    catch 
        @warn("Infeasible l_regress, returning constant bounds.")
        return minimum(Y), zeros(size(X,2))
    end
end

""" 
    svm(X::DataFrame, Y::Array, threshold = 0; solver = CPLEX_SILENT)

Finds the unregularized SVM split, where threshold is the allowable error. 
"""
function svm(X::Matrix, Y::Array, threshold = 0; solver = CPLEX_SILENT)
    m = JuMP.Model(with_optimizer(solver))
    @variable(m, error[1:length(Y)] >= 0)
    @variable(m, β[1:size(X, 2)])
    @variable(m, β0)
    for i=1:length(Y)
        @constraint(m, threshold + error[i] >= Y[i] - β0 - sum(X[i,:] .* β))
        @constraint(m, threshold + error[i] >= -Y[i] + β0 + sum(X[i,:] .* β))
    end
    @objective(m, Min, sum(error))
    optimize!(m)
    return getvalue(β0), getvalue.(β)
end

""" 
    reweight(X::Matrix, mag::Float64 = 10)

Gaussian reweighting of existing data by proximity to previous solution.
Note: mag -> Inf results in uniform reweighting. 
Returns:
- weights: weights of X rows, by Euclidian distance
"""
function reweight(bbl::BlackBoxLearner, sol::DataFrame, mag::Float64 = 10.)
    n_samples, n_features = size(bbl.X);
    bounds = get_bounds(bbl.vars)
    bound_dist = [abs(bounds[var][1] - bounds[var][2]) for var in bbl.vars]
    vecsol = [sol[1,strvar] for strvar in string.(bbl.vars)]
    distances = []
    for i = eachrow(bbl.X)
        push!(distances, sum((values(i) .- vecsol).^2 ./ bound_dist))
    end
    weights = exp.(-1/mag*distances);
    return weights
end

"""
    add_infeasibility_cuts(gm::GlobalModel)

Adds cuts reducing infeasibility of BBC inequalities. 
"""
function add_infeasibility_cuts!(gm::GlobalModel)
    #TODO: CHECK REGRESSION EQUALITIES.
    var_vals = solution(gm)
    cut_count = 0
    feas_tol = get_param(gm, :tighttol)
    for i=1:length(gm.bbls)
        # Inequality Classifiers
        if get_param(gm.bbls[i], :gradients) && gm.bbls[i] isa BlackBoxClassifier && !gm.bbls[i].equality
            bbc = gm.bbls[i]
            if bbc.feas_gap[end] <= -feas_tol
                leaf = bbc.active_leaves[1]
                @assert length(bbc.active_leaves) == 1
                rel_vals = var_vals[:, string.(bbc.vars)]
                eval!(bbc, rel_vals)
                Y = bbc.Y[end]
                update_gradients(bbc, [size(bbc.X, 1)])
                cut_grad = bbc.gradients[end, :]
                push!(bbc.mi_constraints[leaf], 
                    @constraint(gm.model, sum(Array(cut_grad) .* (bbc.leaf_variables[leaf][2] .- 
                                        (Array(rel_vals)' .* bbc.leaf_variables[leaf][1]))) + 
                                        Y * bbc.leaf_variables[leaf][1] + bbc.relax_var >= 0))
                cut_count += 1                        
            end
            for ll in bbc.lls
                if ll.feas_gap[end] <= -feas_tol
                    leaf = ll.active_leaves[1]
                    @assert length(ll.active_leaves) == 1
                    rel_vals = DataFrame(Array(var_vals[:, string.(ll.vars)]), string.(bbc.vars))
                    eval!(bbc, rel_vals)
                    Y = bbc.Y[end]
                    update_gradients(bbc, [size(bbc.X, 1)])
                    cut_grad = bbc.gradients[end, :]
                    push!(ll.mi_constraints[leaf], 
                    @constraint(gm.model, sum(Array(cut_grad) .* (ll.leaf_variables[leaf][2] .- 
                                        (Array(rel_vals)' .* ll.leaf_variables[leaf][1]))) + 
                                        Y * ll.leaf_variables[leaf][1] + ll.relax_var >= 0))
                    cut_count += 1                        
                end
            end
        # Convex Regressors
        elseif get_param(gm.bbls[i], :gradients) && gm.bbls[i] isa BlackBoxRegressor && gm.bbls[i].convex
            bbr = gm.bbls[i]
            if bbr.feas_gap[end] <= -feas_tol
                rel_vals = var_vals[:, string.(bbr.vars)]
                eval!(bbr, rel_vals)
                Y = bbr.Y[end]
                update_gradients(bbr, [size(bbr.X, 1)])
                cut_grad = bbr.gradients[end, :]
                push!(bbr.mi_constraints[1], 
                    @constraint(gm.model, bbr.dependent_var + bbr.relax_var >= 
                        sum(Array(cut_grad) .* (bbr.vars .- Array(rel_vals)')) + Y)) 
                    cut_count += 1                        
            end
            for ll in bbr.lls
                if ll.feas_gap[end] <= -feas_tol
                    rel_vals = DataFrame(Array(var_vals[:, string.(ll.vars)]), string.(bbr.vars))
                    eval!(bbr, rel_vals)
                    Y = bbr.Y[end]
                    update_gradients(bbr, [size(bbr.X, 1)])
                    cut_grad = bbr.gradients[end, :]
                    push!(ll.mi_constraints[1], 
                        @constraint(gm.model, ll.dependent_var + ll.relax_var >= 
                            sum(Array(cut_grad) .* (ll.vars .- Array(rel_vals)')) + Y)) 
                        cut_count += 1                        
                end
            end
        # elseif get_param(gm.bbls[i], :gradients) && gm.bbls[i] isa BlackBoxRegressor
        #     bbr = gm.bbls[i]
        #     if bbr.feas_gap[end] <= -feas_tol
        #         rel_vals = var_vals[:, string.(bbr.vars)]
        #         eval!(bbr, rel_vals)
        #         Y = bbr.Y[end]
        #         update_gradients(bbr, [size(bbr.X, 1)])
        #         cut_grad = bbr.gradients[end, :]
        #         push!(bbr.mi_constraints[1], 
        #             @constraint(gm.model, bbr.dependent_var + bbr.relax_var >= 
        #                 sum(Array(cut_grad) .* (bbr.vars .- Array(rel_vals)')) + Y)) 
        #             cut_count += 1                        
        #     end
        #     for ll in bbr.lls
        #         if ll.feas_gap[end] <= -feas_tol
        #             rel_vals = DataFrame(Array(var_vals[:, string.(ll.vars)]), string.(bbr.vars))
        #             eval!(bbr, rel_vals)
        #             Y = bbr.Y[end]
        #             update_gradients(bbr, [size(bbr.X, 1)])
        #             cut_grad = bbr.gradients[end, :]
        #             push!(ll.mi_constraints[1], 
        #                 @constraint(gm.model, ll.dependent_var + ll.relax_var >= 
        #                     sum(Array(cut_grad) .* (ll.vars .- Array(rel_vals)')) + Y)) 
        #                 cut_count += 1                        
        #         end
        #     end
        end
        # TODO: add infeasibility cuts for Regressors that are locally convex. 
        # TODO: add infeasibility cuts for equalities as well. 
    end
    @info "$(cut_count) infeasibility cuts added."
    return cut_count
end