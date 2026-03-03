using Dates
using Printf

# Write run log to timestamped file inside `logs/`
function write_run_log(pf, belief, particle_snapshots, particle_weights_snapshots, true_labels, true_graph, found_array, rewards, actions)
    try
        if !isdir("logs")
            mkpath("logs")
        end
        ts = Dates.format(Dates.now(), "yyyy-mm-dd_HHMMSS")
        logdir = joinpath(@__DIR__, "logs")
        mkpath(logdir)
        fname = joinpath(logdir, "run_$(ts).txt")
        open(fname, "w") do io
            # True state header
            println(io, "- True state:")
            labels_dict_s = Dict(j => true_labels[j] for j in 1:length(true_labels))
            println(io, "  Labels: ", labels_dict_s)
            println(io, "  Adjacency matrix:")
            n_s = length(true_labels)
            adj_s = zeros(Int, n_s, n_s)
            for u in 1:n_s
                for v in true_graph[u]
                    adj_s[u, v] = 1
                end
            end
            for u in 1:n_s
                println(io, "  ", join(string.(adj_s[u, :]), " "))
            end
            println(io)

            # Write particle table: rows are particles, columns are iterations (side-by-side)
            println(io, "- Particle Table (rows=particles, columns=iterations):")
            println(io)

            # Get number of particles and iterations
            n_particles = length(particle_snapshots)
            n_iterations = length(particle_snapshots)

            # prepare placeholder for max length per iteration
            col_widths = [0 for _ in 1:n_iterations]

            # scan all particles and iterations to find max line length per column
            for (iter_idx, snap) in enumerate(particle_snapshots)
                iter, particles_list = snap
                for p in particles_list
                    labels_line = "Labels: " * string(Dict(j => p.labels[j] for j in 1:length(p.labels)))
                    col_widths[iter_idx] = max(col_widths[iter_idx], length(labels_line))
                    # adjacency rows
                    try
                        for u in 1:length(p.graph)
                            rowline = "  " * join(string.([ (v in p.graph[u]) ? 1 : 0 for v in 1:length(p.graph) ]), " ")
                            col_widths[iter_idx] = max(col_widths[iter_idx], length(rowline))
                        end
                    catch e
                    end
                    weight_line = "Weight: " * string(0.0)
                    col_widths[iter_idx] = max(col_widths[iter_idx], length(weight_line))
                end
                # also consider header length
                header = "Iteration $(iter):"
                col_widths[iter_idx] = max(col_widths[iter_idx], length(header))
                # also consider action string length if actions were recorded
                if length(actions) >= iter_idx
                    action_line = "Action: " * string(actions[iter_idx])
                    col_widths[iter_idx] = max(col_widths[iter_idx], length(action_line))
                end
                # add padding
                col_widths[iter_idx] += 2
            end

            # Also consider space for averaged topology matrix rows and label dict per iteration
            n_iterations = length(particle_snapshots)
            n_nodes = length(true_labels)
            for (iter_idx, snap) in enumerate(particle_snapshots)
                iter, particles_list = snap
                weights = particle_weights_snapshots[iter_idx][2]
                denom = sum(weights)

                # compute averaged adjacency matrix string lines to measure width
                # build matrix of averaged values
                avgA = zeros(Float64, n_nodes, n_nodes)
                try
                    if denom == 0.0
                        # unweighted average
                        for p in particles_list
                            for i in 1:n_nodes, j in 1:n_nodes
                                avgA[i,j] += (j in p.graph[i]) ? 1.0 : 0.0
                            end
                        end
                        avgA ./= max(length(particles_list), 1)
                    else
                        for (pi, p) in enumerate(particles_list)
                            w = weights[pi]
                            for i in 1:n_nodes, j in 1:n_nodes
                                avgA[i,j] += w * ((j in p.graph[i]) ? 1.0 : 0.0)
                            end
                        end
                        avgA ./= denom
                    end
                catch e
                    # if any error occurs (e.g., empty particles), just use zeros
                    avgA .= 0.0
                end

                # measure width of each matrix row string
                for i in 1:n_nodes
                    rowstr = join([@sprintf("%.4f", avgA[i,j]) for j in 1:n_nodes], " ")
                    col_widths[iter_idx] = max(col_widths[iter_idx], length(rowstr) + 2)
                end

                # measure width of label dict string
                label_probs = [ (denom==0.0) ? mean([p.labels[j] ? 1.0 : 0.0 for p in particles_list]) : sum(weights .* ([p.labels[j] ? 1.0 : 0.0 for p in particles_list]))/denom for j in 1:n_nodes]
                dict_entries = [string(j) * " => " * @sprintf("%.4f", label_probs[j]) for j in 1:n_nodes]
                dictstr = join(dict_entries, ", ") * ")"
                col_widths[iter_idx] = max(col_widths[iter_idx], length(dictstr) + 2)
            end

            # For each particle, print a row with iterations side-by-side using global widths
            for particle_idx in 1:n_particles
                println(io, "Particle $particle_idx:")

                # collect lines for each iteration for this particle
                iter_data = Vector{Vector{String}}()
                max_lines = 0
                for (iter_idx, snap) in enumerate(particle_snapshots)
                    iter, particles_list = snap
                    p = particles_list[particle_idx]
                    w = particle_weights_snapshots[iter_idx][2][particle_idx]

                    lines = String[]
                    push!(lines, "Iteration $iter:")
                    push!(lines, "Labels: " * string(Dict(j => p.labels[j] for j in 1:length(p.labels))))
                    push!(lines, "Adjacency matrix:")
                    try
                        for u in 1:length(p.graph)
                            push!(lines, "  " * join(string.([ (v in p.graph[u]) ? 1 : 0 for v in 1:length(p.graph) ]), " "))
                        end
                    catch e
                        # If adjacency matrix fails, skip those rows but keep the header
                    end
                    push!(lines, "Weight: " * string(w))
                    push!(iter_data, lines)
                    max_lines = max(max_lines, length(lines))
                end

                # print lines side-by-side using col_widths
                for li in 1:max_lines
                    line_str = ""
                    for iter_idx in 1:n_iterations
                        colw = col_widths[iter_idx]
                        lines = iter_data[iter_idx]
                        if li <= length(lines)
                            line_str *= rpad(lines[li], colw)
                        else
                            line_str *= rpad("", colw)
                        end
                    end
                    println(io, "  " * line_str)
                end

                println(io)
            end


            # Compute element-wise weighted average adjacency matrices per iteration
            println(io, "- Element-wise weighted average of particles:")

            n_nodes = length(true_labels)

            # avg_topos[iter] is an n_nodes x n_nodes matrix of averaged adjacency values
            avg_topos = [zeros(Float64, n_nodes, n_nodes) for _ in 1:n_iterations]
            label_mat = zeros(Float64, n_nodes, n_iterations)

            for iter_idx in 1:n_iterations
                particles_list = particle_snapshots[iter_idx][2]
                    # unweighted average
                try
                    for p in particles_list
                        for i in 1:n_nodes, j in 1:n_nodes
                            avg_topos[iter_idx][i,j] += (j in p.graph[i]) ? 1.0 : 0.0
                        end
                    end
                    avg_topos[iter_idx] ./= max(length(particles_list), 1)
                catch e
                end
                for j in 1:n_nodes
                    label_mat[j, iter_idx] = mean([p.labels[j] ? 1.0 : 0.0 for p in particles_list])
                end
            end

            for row_i in 1:n_nodes
                line = ""
                for iter_idx in 1:n_iterations
                    rowvals = [@sprintf("%.4f", avg_topos[iter_idx][row_i, colj]) for colj in 1:n_nodes]
                    rowstr = join(rowvals, " ")
                    line *= rpad(rowstr, col_widths[iter_idx])
                end
                println(io, "  " * line)
            end

            println(io)
            dict_line = ""
            for iter_idx in 1:n_iterations
                iter_num = particle_snapshots[iter_idx][1]
                dict_entries = [string(j) * " => " * @sprintf("%.4f", label_mat[j, iter_idx]) for j in 1:n_nodes]
                dictstr =  join(dict_entries, ", ")
                dict_line *= rpad(dictstr, col_widths[iter_idx])
            end
            println(io, "  " * dict_line)
            println(io)
            # Print actions taken at each iteration (aligned with columns)
            action_line = ""
            for iter_idx in 1:n_iterations
                # guard in case actions shorter (shouldn't happen)
                actstr = if length(actions) >= iter_idx
                    a = actions[iter_idx]
                    if a isa ProbeLabel
                        "Probe $(a.node)"
                    elseif a isa Stop
                        "Stop"
                    else
                        string(a)
                    end
                else
                    ""
                end
                action_line *= rpad(actstr, col_widths[iter_idx])
            end
            println(io, "Action taken:")
            println(io, "  " * action_line)
            println(io)
            actual_steps = length(actions)
            println(io, "Number of iterations: ", actual_steps)
            println(io, "Reward: ", last(rewards))
            println(io, "Probed nodes: ", found_array)
            found_pos = sum(found_array .& true_labels)
            total_pos = max(sum(true_labels), 1)
            pct = round(found_pos / total_pos * 100; digits=2)
            println(io, string(pct, "% ", "found positive-class nodes"))
            # Print configured particle filter postprocess information (if available)
            try
                post_str = ""
                flds = fieldnames(typeof(pf))
                if :postprocess in flds
                    post_str = string(getfield(pf, :postprocess))
                    # fallback to try property access
                    if hasproperty(pf, :postprocess)
                        post_str = string(pf.postprocess)
                    else
                        post_str = "<none>"
                    end
                end
                println(io)
                println(io, "- Particle postprocess:")
                for line in split(post_str, '\n')
                    println(io, "  ", line)
                end
            catch e
                println(io)
                println(io, "- Particle postprocess: <error retrieving postprocess>")
            end
        end
        println("Saved run log to ", fname)
    catch e
        @warn "Failed to write log file" exception=(e, catch_backtrace())
    end

end

function heatmap_hamming(particle_label_history, true_label)
    T = length(particle_label_history)
    Nmax = maximum(length.(particle_label_history))

    heat = fill(-1.0, Nmax, T)

    for t in 1:T
        particles_t = particle_label_history[t]

        for i in 1:length(particles_t)
            # Hamming distance
            heat[i,t] = (count(particles_t[i] .!= true_label)/length(true_label))
        end
    end
    return heat
end 

# Visualization helper: create and save plots for a finished episode
function write_visualizations(labels_list, found_list, rewards, weights_mat, particle_snapshots, sims_topo_mat, sims_label_mat, found_counts, belief)
    # Determine actual episode length (may be shorter if Stop was chosen early)
    actual_steps = length(rewards)
    
    # If no steps were taken, skip plotting
    if actual_steps <= 1
        println("Episode ended immediately (Stop chosen on first step). Skipping plots.")
        return
    end

    cum_rewards = cumsum(rewards)

    # Particle counts
    # N_particles = length(particles(belief))
    # topk = N_particles

    # # At each time step, find the top-K particles and extract their weights/similarities
    # weights_ranked = zeros(Float64, topk, actual_steps)
    # # sims_topo_ranked = zeros(Float64, topk, actual_steps)
    # sims_label_ranked = zeros(Float64, topk, actual_steps)

    # for t in 1:actual_steps
    #     weights_t = weights_mat[t]
    #     # sims_topo_t = sims_topo_mat[t]
    #     sims_label_t = sims_label_mat[t]
    #     top_indices = partialsortperm(weights_t, 1:topk, rev=true)
    #     weights_ranked[:, t] = weights_t[top_indices]
    #     # sims_topo_ranked[:, t] = sims_topo_t[top_indices]
    #     sims_label_ranked[:, t] = sims_label_t[top_indices]
    # end

    p1 = plot(1:length(cum_rewards), cum_rewards, xlabel="Time Step", ylabel="Cumulative Reward", title="Cumulative Reward", legend=false, xticks=1:actual_steps)

    # p2 = heatmap(1:actual_steps, 1:topk, weights_ranked, xlabel="Time Step", ylabel="Rank (1=best)", title="Top-$(topk) Particle Weights Over Time (ranked per-step)", colorbar_title="Weight", xticks=1:actual_steps)
    # p3 = heatmap(1:actual_steps, 1:topk, sims_topo_ranked, xlabel="Time Step", ylabel="Rank (1=best)", title="Topology similarity for top-$(topk) particles (ranked)", colorbar_title="Topo similarity", clim=(0.0,1.0), xticks=1:actual_steps)
    # p4 = heatmap(1:actual_steps, 1:topk, sims_label_ranked, xlabel="Time Step", ylabel="Rank (1=best)", title="Label similarity for top-$(topk) particles (ranked)", colorbar_title="Label similarity", clim=(0.0,1.0), xticks=1:actual_steps)

    found_percent = found_counts[1:actual_steps] ./ max(sum(labels_list), 1) .* 100
    p5 = plot(1:actual_steps, found_percent, xlabel="Time Step", ylabel="Found Positives (%)", title="Percentage of found positive-class nodes over time", legend=false, ylim=(-1, 101), xticks=1:actual_steps)

    p6 = bar(1:length(found_list), Int.(found_list), xlabel = "Node index", ylabel = "Found (1 = probed)", title = "Probed nodes (green=positive, red=negative)", legend = false, ylim = (-0.1, 1.1), xticks = false)
    for i in 1:length(labels_list)
        annotate!(p6, i, -0.15, text(string(i),8,labels_list[i] ? :green : :red, :center))
    end

    # max_topo = vec(mean(sims_topo_ranked; dims=1))
    # max_label = vec(mean(sims_label_ranked; dims=1))
    # p7 = plot(1:actual_steps, hcat(max_topo, max_label), labels=["Avg topo similarity" "Avg label similarity"],
    #           xlabel="Time Step", ylabel="Similarity", title="Average similarity over time",
    #           legend=:topleft, ylim=(-0.1,1.1), xticks=1:actual_steps)

    heat = heatmap_hamming(particle_snapshots, labels_list)
    p8 = heatmap(heat,
        ylabel = "Particle Index",
        xlabel = "Timestep",
        title  = "Particle Distance to True State",
        clim = (-1, 1),
        colorbar_title = "Hamming Distance",
        xticks=1:length(heat)
    )

    fig1 = plot(p1, p5, p6, p8, layout=(4,1), size=(800,1100))
    # fig2 = plot(p2, p3, p4, p7, layout=(4,1), size=(800,1100))
    # fig2 = plot(p2, p4, layout=(2,1), size=(800,1100))

    try
        savefig(fig1, "performance.png")
        # savefig(fig2, "belief_diagnostics.png")
    catch e
        @warn "Failed to save figures" exception=(e, catch_backtrace())
    end
    display(fig1)
    # display(fig2)
end