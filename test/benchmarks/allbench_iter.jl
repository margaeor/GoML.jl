
include("../load.jl")


REPAIR = true 
OPT_SAMPLING = false

using Serialization: serialize, deserialize
using DataFrames, Dates 

gm = nothing

"""
Loads and parses gam from file. 
If already loaded once, it will be loaded
from pickle instead of being parsed again.
"""
function load_gam_from_path(path; force_reload=false)

    if !force_reload && isfile("$(path).bin")
        gams = open("$(path).bin", "r") do io
            deserialize(io)
        end
        return gams
    end
    gams = GAMSFiles.parsegams(path)
    GAMSFiles.parseconsts!(gams)

    open("$(path).bin", "w") do io
        serialize(io, gams)
    end

    return gams 
end

"""
Returns a CSV with the stats of different
gam problems located in a subfolder of the fwolderr GoML.GAMS_DIR.
Those sats include:
- Number of constraints 
- Number of variables 
- Number of bounded variables 
- Number of variables of each type (i.e. continuous, integer, binary)

The argument force_reload decides whether we should recreate the stats.
"""
function get_problem_stats(folder_name; force_reload=false, keep_only_selected=true)

    dir = GoML.GAMS_DIR*folder_name*"\\"
    csv_path = dir*"problem_stats.csv"

    if !force_reload && isfile(csv_path)
        df = DataFrame(CSV.File(csv_path))
        if keep_only_selected
            df = filter(:selected => x -> x==1, df)
            if size(df, 1) == 0
                println("You have not selected any problems in the csv.")
                println("Please go to $(csv_path) and set selected=1 in the problems you want to select")
            end
        end
        return df
    end

    # Read all file names/paths in the folder
    all_paths = [d for d in readdir(dir; join=true) if occursin(".gms", d)]
    all_names = [replace(f, ".gms" => "") for f in readdir(dir; join=false) if occursin(".gms", f)]


    df = DataFrame()

    for (name, path) in zip(all_names, all_paths)

        # try
            println("Adding $(name)")

            gams = load_gam_from_path(path; force_reload=false)

            data = DataFrame(
                "name" => name,
                "n_constr" => length(gams["equations"]),
                "n_vars" => 0,
                "n_bounded_vars" => 0,
                "continuous" => 0,
                "integer" => 0,
                "binary" => 0,
                "folder" => folder_name,
                "selected" => 0,
                "all_bounded" => 0,
                "optimal" => ""
            )
            
            vars = GAMSFiles.getvars(gams["variables"])

            for (var, info) in vars
                
                assignments = info.assignments
                type = info.typ # Variable type (free, positive, negative, integer, binary)
                #println(var)
                if var == "objvar" continue end 

                # The types of bounds our variable has 
                bound_types = Set([a.first.text for a in assignments if a.first.text ∈ ["lo","up"]])

                # Do some bookeeping to count the number of variables (and bounded variables)
                if type == "free"

                    data[1, "continuous"] += 1

                elseif type == "positive"
                
                    data[1, "continuous"] += 1
                    push!(bound_types,  "lo")

                elseif type == "negative"

                    data[1, "continuous"] += 1
                    push!(bound_types,  "lo")

                elseif type == "binary"
                    
                    data[1, "binary"] += 1
                    push!(bound_types,  "lo")
                    push!(bound_types,  "up")
                else
                    data[1, "integer"] += 1
                end
                
                if length(bound_types) >= 2
                    data[1, "n_bounded_vars"] += 1
                end
                data[1, "n_vars"] += 1
            end
            data[1, "all_bounded"] = 1*(data[1, "n_bounded_vars"] == data[1, "n_vars"])
            append!(df, data)
        # catch
        #     println("Couldn't add file $(name)")
        # end
    end

    CSV.write(csv_path, df)

    if keep_only_selected
        @warn("You have not selected any problems in the $(folder_name) csv.\n
        Please go to $(csv_path) and set selected=1 in the problems you want to select")
    end
    return df
end

function solve_and_benchmark(folders; alg_list = ["GBM", "SVM"])
    
    function create_gm(name, folder)
        gm = GAMS_to_GlobalModel(GoML.GAMS_DIR*"$(folder)\\", name*".gms"; alg_list = alg_list, regression=false, relax_coeff=0)
        set_optimizer(gm, CPLEX_SILENT)
        set_param(gm, :sample_coeff, 1800)
        return gm 
    end

    function solve_gm(gm, name, folder; relax_coeff=0, ro_factor=0)
        
        set_param(gm, :ro_factor, ro_factor)
        gm.relax_coeff = relax_coeff
        
        bounds = get_bounds(gm)
        bounds = Dict(k => v for (k,v) in bounds if v[1]!= -Inf && v[2]!=Inf)
        bounds = Dict(k => v for (k,v) in bounds if string(k) ∉ ["objvar", "r_rel"])
        llbs, uubs = [v[1] for (k,v) in bounds], [v[2] for (k,v) in bounds]
        lengths = uubs-llbs
        obj_arr = []
        feas_arr = []
        df_algs = nothing

        for iter = 1:3
            try
                globalsolve!(gm; repair=REPAIR, opt_sampling=OPT_SAMPLING)
                df_algs = vcat([bbl.learner_performance for bbl in gm.bbls]...)
                x0 = collect(gm.solution_history[end, string.(keys(bounds))])
                feas_gaps = []
                try
                    feas_gaps = [bbl.feas_gap[end] for bbl in gm.bbls if isa(bbl, BlackBoxClassifier)] 
                catch  
                    println("Feas gap exception")
                end
                #feas_gap(gm)
                # Performance of the different algorithms (e.g. GBM, SVM, OCT)


                push!(obj_arr, gm.cost[end])
                push!(feas_arr, feas_gaps)
                
                gm = create_gm(name, folder)
                # clear_tree_constraints!(gm)
                # clear_data!(gm)

                tmp_bounds = get_bounds(gm)
                tmp_bounds = Dict(k => v for (k,v) in tmp_bounds if v[1]!= -Inf && v[2]!=Inf)
                tmp_bounds = Dict(k => v for (k,v) in tmp_bounds if string(k) ∉ ["objvar", "r_rel"])

                new_lbs, new_ubs = x0-lengths/2^(iter+1), x0+lengths/2^(iter+1) 
                new_lbs, new_ubs = max.(new_lbs, llbs), min.(new_ubs, uubs)
                new_lbs, new_ubs = min.(new_lbs, new_ubs), max.(new_lbs, new_ubs)
                
                println(new_lbs, llbs)
                println(new_ubs, uubs)

                @assert all(new_lbs .>= llbs)
                @assert all(new_ubs .<= uubs)
                for (k, lb, ub) in zip(keys(tmp_bounds), new_lbs, new_ubs)
                    JuMP.set_lower_bound(k, lb)
                    JuMP.set_upper_bound(k, ub)
                end
            catch e 
                showerror(stdout, e)
                break
            end  
        end
        
        

        return df_algs, obj_arr[end], gm, obj_arr, feas_arr
    
    end

    function solve_baron(name, folder)
        m = GAMS_to_baron_model(GoML.GAMS_DIR*"$(folder)\\", name*".gms")
        optimize!(m)

        return JuMP.objective_value(m), DataFrame()
    end

    df_all = DataFrame()
    df_algs_all = DataFrame()

    output_path = "dump/benchmarks/"
    Base.Filesystem.mkpath(output_path)
    suffix = Dates.format(Dates.now(), "YY-mm-dd_HH-MM-SS")
    
    og_alg_list = copy(alg_list)

    for folder in folders 
        df_stats = get_problem_stats(folder;force_reload=false)
        
        for (i, row) in enumerate(eachrow(df_stats))
            
            alg_list = copy(og_alg_list)

            # for ro_factor in [0.0]#[0,0.01,0.1,0.2,0.5,1,2] , 0.05, 0.1, 0.5
            ro_factor = 0.0
            name, folder = row["name"], row["folder"]
            # println(name)
            # if name ∉ ["st_e11","st_e02"]
            #     continue 
            # end
            # if name ∉ ["ex8_3_14", "ex8_3_4", "ex5_2_5", "ex8_3_9", "ex8_3_3", "ex8_3_2"]
            #     continue
            # end
            
            # if name ∉ ["ex8_3_14", "ex8_3_4", "ex5_2_5", "ex8_3_9", "ex8_3_3", "ex8_3_2", "ex5_4_4", "ex8_2_1a", "ex8_2_4a", "ex6_2_7", "ex6_2_5", "ex5_2_5", "ex5_3_3", "ex6_2_9", "ex6_2_10", "ex5_4_4", "ex6_2_13", "ex3_1_1", "ex7_2_3", "ex2_1_1", "ex7_2_4", "ex4_1_1"]
            #     continue
            # end

            # if name ∉  ["st_e17", "ex6_2_8", "ex6_2_12", "sample", "st_e04", "st_bsj4", "ex5_2_2_case3", "ex2_1_6", "ex2_1_8", "st_e30", "st_e16", "alkylation"]#["sample"]#
            #     continue
            # end
            
            # Bad v2
            # if name ∉ ["st_e33","ex2_1_6","ex5_4_3","ex2_1_8","st_e30","ex5_4_4","ex6_2_12","alkyl","st_bsj4","ex6_2_8","ex4_1_1","ex6_2_9","sample","st_e17"]
            #     continue
            # end

            # Infeasible 
            # if name ∉ ["st_e02","st_e11","st_e04","st_e05","ex3_1_1","ex5_4_2","ex7_2_3", "process", "ex5_3_2"]
            #     continue
            # end

            # Infeasible v2
            # if name ∉ ["st_e11","ex7_2_3","process","ex8_2_1b","ex8_2_4b","ex5_3_3","ex8_3_9","ex8_3_14","ex8_3_2","ex8_3_3","ex8_3_4"]
            #     continue
            # end
            # if name ∉ ["st_e11","ex7_2_3","process","ex8_2_1b","ex8_2_4b","ex5_3_3","ex8_3_9","ex8_3_14","ex8_3_2","ex8_3_3","ex8_3_4"]
            #     continue
            # end

            # Non-cvx loose-ends
            # if name ∉ ["ex8_3_4", "ex8_3_9","ex5_2_5","ex8_3_14","ex8_2_1b","ex8_3_2","ex8_2_4b","ex8_3_3","ex5_3_3","ex8_2_4a","ex5_4_4","ex8_2_1a"]
            #     continue
            # end

            # CVX loose-ends
            # if name ∉ ["ex2_1_6","ex5_4_3","ex2_1_1","ex2_1_8","ex5_4_4","st_e30","ex8_4_2","ex7_2_4","ex5_2_5","ex6_2_12","ex8_4_1","st_e04","alkyl"]
            #     continue
            # end

            # if name ∈  ["ex4_1_1","ex4_1_2","ex4_1_3","ex4_1_6","ex4_1_7","prob06","ex4_1_8","ex4_1_9","st_e01","st_e08","st_e09","st_e17","st_e18","st_e19","st_e22","st_e23","st_e24","st_e26","st_ht","ex6_2_8","st_cqpjk2","st_e02","st_e11","ex6_2_12","ex6_2_14","ex6_2_9","sample","st_bpv1","st_bpv2","st_e04","st_e12","st_e41","ex2_1_1","ex3_1_2","st_e05","ex6_2_10","ex6_2_13","ex7_2_2","st_bsj3","st_bsj4","st_e21","ex5_2_4","ex7_2_1","ex3_1_1","ex5_4_2","ex7_2_3","ex7_2_4","st_iqpbk1","st_iqpbk2","ex5_2_2_case1","ex5_2_2_case2","ex5_2_2_case3","ex6_2_5","ex6_2_7","st_e33","alkylation","ex2_1_5"]
            #     continue
            # end

            # if name ∉ ["ex2_1_8","ex2_1_1","ex5_4_4","ex5_4_3","ex8_4_1","ex5_2_5","st_e16","ex5_2_4","ex6_2_8","st_e03","ex5_3_2","ex6_2_7","st_bsj4"]
            #     continue
            # end

            solved = false
            for oct_sampling in [true]

                
                ts = time()
                id = 1
                for ro_factor in [0.0]#[0.0,0.01,0.1,0.5,1]
                    for relax_coeff in [0.0] #[0.0,1e2,1e4]
                        for hessian in [false]
                            for momentum in [0.]
                                global gm = create_gm(name, folder)
                                # if solved 
                                #     continue
                                # end

                                n_bbls = length([bbl for bbl in gm.bbls if (bbl isa BlackBoxClassifier || bbl isa BlackBoxRegressor)])
                                baron_obj = 0.0
                                try
                                    baron_obj = parse(Float32, replace(string(row["optimal"]), r"[^0-9\.-]" => ""))
                                catch
                                    println("Couldn't parse objective for $(name)") 
                                    continue
                                end

                                row_id = size(df_all, 1)

                                df_tmp = DataFrame(
                                    "row_id" => row_id,
                                    "gm" => NaN, 
                                    "baron" => baron_obj,
                                    "diff" => NaN,
                                    "subopt_factor" => NaN,
                                    "gm_time" => NaN,
                                    "ba_time" => NaN,
                                    "algs" => "[\""*join(alg_list, "\",\"")*"\"]",
                                    "feas_gaps" => [[]],
                                    "ro_factor" => ro_factor,
                                    "solved" => NaN,
                                    "relax_coeff" => relax_coeff,
                                    "n_bbls" => n_bbls,
                                    "relax_epsilon" => 0,
                                    "momentum" => NaN,
                                    "hessian" => false,
                                    "oct_sampling" => false,
                                    "cvx_constr" => false,
                                    "obj_arr" => [[]],
                                    "feas_arr" => [[]]
                                )
                                
                                id += 1

                                try
                                    
                                    Random.seed!(50)

                                    # relax_coeff = 0
                                    df_algs = nothing 
                                    gm_obj = nothing
                                    #gm = nothing
                                    # try
                                    #     df_algs, gm_obj, gm = solve_gm(name, folder; ro_factor=ro_factor, relax_coeff=0)
                                    # catch
                                    #     @info("Trying with relax var")
                                    #     use_relax_var = true
                                    #     df_algs, gm_obj, gm = solve_gm(name, folder; ro_factor=ro_factor, relax_coeff=1)
                                    # end
                                    # gm_obj, df_algs = solve_baron(name, folder)
                                    # solved=true
                                    # df_algs, gm_obj, gm = solve_gm(name, folder; ro_factor=ro_factor, relax_coeff=relax_coeff)
                                    
                                    set_param(gm, :momentum, momentum)
                                    set_param(gm, :second_order_repair, hessian)
                                    set_param(gm, :oct_sampling, oct_sampling)

                                    df_algs, gm_obj, gm, obj_arr, feas_arr = solve_gm(gm, name, folder; ro_factor=ro_factor, relax_coeff=relax_coeff)


                                    gm_time = time()-ts
                                    baron_time = gm_time
                                    subopt = abs(baron_obj)<1 ? ((gm_obj+1)/(1+baron_obj)) : gm_obj/baron_obj
                                    subopt = abs(baron_obj)<1 ? ((gm_obj-baron_obj)/(1+abs(baron_obj))) : (gm_obj-baron_obj)/abs(baron_obj)
                                    subopt = 1-subopt
                                    
                                    feas_gaps = []
                                    try
                                        feas_gaps = [bbl.feas_gap[end] for bbl in gm.bbls if isa(bbl, BlackBoxClassifier)] 
                                    catch  
                                        println("Feas gap exception")
                                    end
                                    
                                    if abs(1-subopt) <= 1e-3  
                                        solved = true
                                    end
                                    

                                    df_tmp[!, "gm"] = [gm_obj]
                                    df_tmp[!, "diff"] = [gm_obj-baron_obj]
                                    df_tmp[!, "subopt_factor"] = [subopt]
                                    df_tmp[!, "gm_time"] = [gm_time]
                                    df_tmp[!, "ba_time"] = [gm_time]
                                    df_tmp[!, "feas_gaps"] = [feas_gaps]
                                    df_tmp[!, "solved"] = [1]
                                    df_tmp[!, "relax_coeff"] = [relax_coeff]
                                    df_tmp[!, "relax_epsilon"] = [gm.relax_epsilon]
                                    df_tmp[!, "momentum"] = [get_param(gm, :momentum)]
                                    df_tmp[!, "hessian"] = [get_param(gm, :second_order_repair)]
                                    df_tmp[!, "oct_sampling"] = [get_param(gm, :oct_sampling)]
                                    df_tmp[!, "cvx_constr"] = [get_param(gm, :convex_constrs)]
                                    df_tmp[!, "obj_arr"] = [obj_arr]
                                    df_tmp[!, "feas_arr"] = [feas_arr]
                                    new_row = hcat(df_tmp, DataFrame(row))
                                    
                                    
                                    
                                    df_algs[!, "row_id"] = [row_id for _ = 1:size(df_algs, 1)]

                                    append!(df_all, new_row)
                                    append!(df_algs_all, df_algs)

                                    println(df_all)

                                    # break
                                catch e
                                    df_tmp[!, "solved"] = [0]

                                    new_row = hcat(df_tmp, DataFrame(row))
                                    append!(df_all, new_row)
                                    showerror(stdout, e)
                                    #println("Error solving $(name)")
                                    #println(stacktrace(catch_backtrace()))
                                end

                                try
                                    csv_path = output_path*"benchmark$(suffix).csv"
                                    csv_path_alg = output_path*"benchmark_alg$(suffix).csv"
                                    #println(csv_path)
                                    CSV.write(csv_path, df_all)
                                    CSV.write(csv_path_alg, df_algs_all)
                                catch
                                    println("Couldn't write to CSV")
                                end
                            end
                        end
                    end
                end
            end
        end
    end
    println(df_all)
end

folders = ["global"]

solve_and_benchmark(folders; alg_list = ["GBM", "SVM", "MLP"])
#"GBM", "SVM", "MLP"