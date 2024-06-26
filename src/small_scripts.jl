#=
small_scripts:
- Julia version: 1.5.1
- Author: Berk
- Date: 2020-11-04
=#

"""
    data_to_Dict(data::Union{Dict, DataFrame, DataFrameRow}, model::JuMP.Model)

Turns data into a Dict, with JuMP.VariableRef subs.
"""
function data_to_Dict(data::Union{Dict, DataFrame, DataFrameRow}, model::JuMP.Model)
    if data isa Dict
        return Dict(fetch_variable(model, key) => val for (key, val) in data)
    else
        colnames = names(data)
        if size(data, 1) == 1
            return Dict(fetch_variable(model, name) => data[!, name][1] for name in colnames)
        else
            return Dict(fetch_variable(model, name) => data[!, name] for name in colnames)
        end
    end
end

""" Wrapper around Iterators.flatten for variables. """
flat(arr) = Array{Any, 1}(collect(Iterators.flatten(arr)))

""" For substitution into expressions. IMPORTANT. """
function substitute(e::Expr, pair)
    MacroTools.prewalk(e) do s
        s == pair.first && return pair.second
        s
    end
end

""" power function from gams """
power(var, num) = var^num

""" sqr function from gams. """
sqr(var) = var^2

""" Checks outer-boundedness of values of a Dict. """
function check_bounds(bounds::Dict)
    if any(isinf.(Iterators.flatten(values(bounds))))
        throw(GoMLException("Unbounded variables in model!"))
    else
        return
    end
end

"""
    get_varmap(expr_vars::Array, vars::Array)

Helper function to map vars to flatvars.
Arguments:
    flatvars is a flattened Array{JuMP.VariableRef}
    vars is the unflattened version, usually derived from an Expr.
Returns:
    Dict of ID maps
"""
function get_varmap(expr_vars::Array, vars::Array)
    length(flat(expr_vars)) >= length(vars) || throw(GoMLException(string("Insufficiently many input
        variables declared in ", vars, ".")))
    unique(vars) == vars || throw(GoMLException(string("Nonunique variables among ", vars, ".")))
    if expr_vars == vars
        return collect(1:length(vars))
    end
    varmap = Tuple[(0,0) for i=1:length(vars)]
    for i = 1:length(expr_vars)
        if expr_vars[i] isa JuMP.VariableRef
            try
                varmap[findall(x -> x == expr_vars[i], vars)[1]] = (i,0)
            catch
                throw(GoMLException(string("Scalar variable ", expr_vars[i], " was not properly declared in vars.")))
            end
        else
            for j=1:length(expr_vars[i])
                try
                    varmap[findall(x -> x == expr_vars[i][j], vars)[1]] = (i,j)
                catch
                    continue
                end
            end
        end
    end
    length(varmap) == length(vars) || throw(GoMLException(string("Could not properly map
                                            expr_vars: ", expr_vars,
                                            "to vars: ", vars, ".")))
    return varmap
end

get_varmap(expr_vars::Nothing, vars::Array) = nothing


""" Returns the mapping from flattened expr_vars to vars. """
function get_datamap(expr_vars::Array, vars::Array)
    datamap = []
    flatvars = flat(expr_vars)
    for var in vars
        idx = findall(x -> x == var, flatvars)
        if length(idx) == 1
            push!(datamap, idx[1])
        else
            throw(GoMLException("There was an issue with getting data mapping."))
        end
    end
    return datamap
end


"""Returns the relevant ranges for variables in expr_vars..."""
function get_var_ranges(expr_vars::Array)            
    var_ranges = []
    count = 0
    for varlist in expr_vars
        if varlist isa VariableRef
            count += 1
            push!(var_ranges, count)
        else
            push!(var_ranges, (count + 1 : count + length(varlist)))
            count += length(varlist)
        end
    end
    return var_ranges
end

"""
    zeroarray(varmap::Array)

Creates a template array for deconstruct function.
"""
function zeroarray(var_ranges::Array)
    arr = []
    for i in var_ranges
        if i isa UnitRange || i isa Array
            push!(arr, zeros(length(i)))
        elseif i isa Int64
            push!(arr, 0)
        end
    end
    return arr
end

"""
    deconstruct(data::DataFrame, vars::Array, varmap::Array)

Takes in data for input into a Function, and rips it apart into appropriate arrays.
"""
function deconstruct(data::DataFrame, vars::Array, expr_vars::Array, varmap::Array)
    n_samples, n_vars = size(data)
    zeroarr = zeroarray(get_var_ranges(expr_vars))
    arrs = [];
    stringvars = string.(vars)
    for i = 1:n_samples
        narr = deepcopy(zeroarr)
        for j = 1:length(varmap)
            if varmap[j] isa Tuple && varmap[j][2] != 0
                narr[varmap[j][1]][varmap[j][2]] = data[i, stringvars[j]]
            else
                narr[varmap[j][1]] = data[i, stringvars[j]]
            end
        end
        push!(arrs, narr)
    end
    return arrs
end


"""
Allows for struct inheritance
"""
macro inherit(name, base, fields)
    base_type = Core.eval(@__MODULE__, base)
    base_fieldnames = fieldnames(base_type)
    base_types = [t for t in base_type.types]
    base_fields = [:($f::$T) for (f, T) in zip(base_fieldnames, base_types)]
    res = :(mutable struct $name end)
    push!(res.args[end].args, base_fields...)
    push!(res.args[end].args, fields.args...)
    return res
end