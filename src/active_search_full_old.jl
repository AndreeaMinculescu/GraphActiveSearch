# active_search_pomcp.jl
# Active Search POMDP + BasicPOMCP example
#
using POMDPs
using POMDPTools.POMDPDistributions: Deterministic, ImplicitDistribution
using BasicPOMCP
using Random
using LinearAlgebra
using LightGraphs
using StatsBase
using ParticleFilters
using Plots  # Add this at the top to enable plotting

########################
# Types / State / Action
########################

struct ASState
    graph::Vector{Vector{Int}}   # adjacency list
    labels::Vector{Bool}         # true label per node
    found::Vector{Bool}          # whether node previously probed (for reward)
end

abstract type ASAction end
struct ProbeLabel <: ASAction
    node::Int
end
# Stop action: end the episode / no-op
struct Stop <: ASAction
end

# Observation type
struct ASObservation
    node::Int                 # the probed node index
    label::Bool
    neighbors::Vector{Int}
end

########################
# Model container
########################

struct ASPOMDP <: POMDP{ASState, ASAction, ASObservation}
    n::Int                           # number of nodes in graph
    graph_prior::Vector{ASState}     # prior as vector of ASState examples
    topo_noise::Float64              # used by local noisy neighbor sensor
    reward_pos::Float64              # reward for finding a new positive label
end

POMDPs.discount(::ASPOMDP) = 0.9

########################
# Prior world sampler (connected ER + clustered positives)
########################


function sample_random_world(n; rng=MersenneTwister())
    # Parameters (fixed, not user-set)
    p_edge = 0.15
    cluster_prob = 0.5
    decay = 0.7
    max_positives = max(1, floor(Int, n/3))

    # 1. Generate an Erdos-Renyi graph
    g = erdos_renyi(n, p_edge)
    graph = [neighbors(g, i) for i in 1:n]

    # 2. Choose cluster centers
    n_centers = max(1, round(Int, sqrt(n)/3))
    centers = rand(rng, 1:n, n_centers)

    # 3. Initialize all labels negative
    labels = falses(n)
    n_pos = 0

    # 4. BFS-based clustering, but strictly cap total positives
    for c in centers
        if n_pos >= max_positives
            break
        end
        if !labels[c] && rand(rng) < cluster_prob
            labels[c] = true
            n_pos += 1
        end
        visited = falses(n)
        queue = [(c, 0)]
        visited[c] = true
        while !isempty(queue)
            if n_pos >= max_positives
                break
            end
            (v, depth) = popfirst!(queue)
            p = cluster_prob * (decay^depth)
            if !labels[v] && rand(rng) < p
                labels[v] = true
                n_pos += 1
                if n_pos >= max_positives
                    break
                end
            end
            for nb in graph[v]
                if !visited[nb]
                    visited[nb] = true
                    push!(queue, (nb, depth + 1))
                end
            end
        end
    end

    found = falses(n)
    return ASState(graph, labels, found)
end


########################
# POMDP interface 
########################

# initial belief: return ParticleCollection of prior ASState objects
function POMDPs.initialstate(m::ASPOMDP)
    #return BootstrapFilter(m, 20)
    #return ParticleCollection(copy(m.graph_prior))
    return copy(m.graph_prior)
end

# actions: probe any node's label, or stop
POMDPs.actions(m::ASPOMDP) = vcat([ProbeLabel(i) for i in 1:m.n], [Stop()])

# transition: graph and labels static; 'found' updated when label probe occurs
function POMDPs.transition(m::ASPOMDP, s::ASState, a::ASAction)
    new_found = copy(s.found)
    if a isa ProbeLabel
        new_found[a.node] = true
    end
    return Deterministic(ASState(s.graph, s.labels, new_found))
end

# local noisy neighbor sensor (false negatives/positives)
# function local_noisy_neighbors(graph::Vector{Vector{Int}}, node::Int, topo_noise::Float64, rng::AbstractRNG)
#     true_neighbors = graph[node]
#     observed = [nb for nb in true_neighbors if rand(rng) > topo_noise]   # drop some true neighbors
#     for other in 1:length(graph)
#         if other != node && !(other in true_neighbors)
#             if rand(rng) < topo_noise * FP_RATE_FACTOR
#                 push!(observed, other)   # add false positives rarely
#             end
#         end
#     end
#     unique!(observed)
#     return observed
# end


# --- 1) include probed node in observation type ---
# replace previous ASObservation with this (put near the struct definitions)
# struct ASObservation
#     label::Bool
#     neighbors::Vector{Int}
# end
# -> change to:

# Make sure POMDPs.observation returns the node too:
const FP_RATE_FACTOR = 1/5   
function POMDPs.observation(m::ASPOMDP, a::ASAction, s::ASState)
    rng = Random.GLOBAL_RNG
    if a isa ProbeLabel
        lbl = s.labels[a.node]
        # Return an ImplicitDistribution that samples noisy observations
        return ImplicitDistribution() do rng
            true_neighbors = s.graph[a.node]
            observed = [nb for nb in true_neighbors if rand(rng) > m.topo_noise]   # drop some true neighbors
            for other in 1:length(s.graph)
                if other != a.node && !(other in true_neighbors)
                    if rand(rng) < m.topo_noise * FP_RATE_FACTOR
                        push!(observed, other)   # add false positives rarely
                    end
                end
            end
            unique!(observed)
            ASObservation(a.node, lbl, observed)
        end
    elseif a isa Stop
        # Return a harmless ASObservation sentinel (node==0) so planners that
        # expect an ASObservation type do not fail when observing a Stop.
        # Treat node==0 as "no observation" in likelihood functions.
        return Deterministic(ASObservation(-1, false, Int[]))
    else
        error("Unsupported action type in observation")
    end
end

# --- 2) observation likelihood: p(o | particle) ---

function neighbors_likelihood(true_neighbors::Vector{Int}, node::Int, observed::Vector{Int}, topo_noise::Float64)
    # We'll compute probability of exactly seeing `observed` given true_neighbors, assuming
    # - each true neighbor is missed with probability topo_noise (false negative)
    # - each non-neighbor (except node itself) is included as false positive with probability topo_noise*FP_RATE_FACTOR
    trset = Set(true_neighbors)
    obsset = Set(observed)

    p = 1.0
    # For true neighbors: must be observed with prob (1 - topo_noise), or missed with topo_noise
    for nb in trset
        if nb in obsset
            p *= (1.0 - topo_noise)
        else
            p *= topo_noise
        end
    end

    # For other possible nodes (excluding the node itself and true neighbors)
    fp_rate = topo_noise * FP_RATE_FACTOR
    for nb in setdiff(obsset, trset)
        p *= fp_rate
    end

    return p
end

function observation_likelihood(p::ASState, o::ASObservation, topo_noise::Float64)
    # If no observation/sentinel (e.g., Stop action) treat likelihood as neutral (1.0)
    if o.node == -1
        return 1.0
    end

    node = o.node
    true_nb = p.graph[node]
    # topology likelihood
    p_neigh = neighbors_likelihood(true_nb, node, o.neighbors, topo_noise)
    # label likelihood (simple sensor model)
    p_label = (p.labels[node] == o.label) ? 1.0 : 0.01

    return p_neigh * p_label
end

# Compute normalized particle weights given a belief and an observation
function compute_particle_weights(belief::Vector{ASState}, o::ASObservation, m::ASPOMDP)
    N = length(belief)
    lik = zeros(Float64, N)
    # If no observation/sentinel (Stop), return uniform weights
    if o.node == -1
        lik .= 1.0 / N
        return lik
    end

    for i in 1:N
        lik[i] = observation_likelihood(belief[i], o::ASObservation, m.topo_noise)
    end
    s = sum(lik)
    if s == 0.0
        lik .= 1.0 / N
    else
        lik ./= s
    end
    return lik
end

# Fraction of labels that match between a particle and the true state
function label_similarity(p::ASState, s::ASState)
    return sum(p.labels .== s.labels) / length(s.labels)
end

# Topology similarity between two states: average per-node Jaccard index of neighbor sets
function topology_similarity(p::ASState, s::ASState)
    n = length(s.graph)
    total = 0.0
    for i in 1:n
        a = Set(p.graph[i])
        b = Set(s.graph[i])
        if isempty(a) && isempty(b)
            total += 1.0
        else
            inter = length(intersect(a,b))
            uni = length(union(a,b))
            total += (uni == 0) ? 0.0 : inter/uni
        end
    end
    return total / n
end

# --- 3) weighted belief updater (resample with replacement) ---
function update_belief_weighted(belief::Vector{ASState}, a::ASAction, o::ASObservation, m::ASPOMDP; nparticles=nothing)
    # belief is ParticleCollection (vector-like)
    particles = [p for p in belief]   # gets vector of ASState
    N = length(particles)
    if N == 0
        error("Belief is empty; cannot update")
    end

    # compute likelihoods for each particle
    lik = zeros(Float64, N)
    for i in 1:N
        lik[i] = observation_likelihood(particles[i], o, m.topo_noise)
    end

    # normalize
    s = sum(lik)
    if s == 0.0
        # lik .= 1.0 / N
        error("Weights are empty")
    else
        lik ./= s
    end

    # choose number of particles in new belief
    M = isnothing(nparticles) ? N : nparticles

    # resample indices with replacement using StatsBase
    new_indices = sample(1:N, Weights(lik), M; replace=true)
    new_particles = [deepcopy(particles[i]) for i in new_indices] 

    # return new ParticleCollection
    return new_particles
end

# reward: + reward_pos when probing a previously-unfound positive
function POMDPs.reward(m::ASPOMDP, s::ASState, a::ASAction, sp::ASState)
    if a isa ProbeLabel
        v = a.node
        if s.found[v]
            return -20
        end
        return (s.labels[v] && !s.found[v]) ? m.reward_pos : -1
    elseif a isa Stop
        if all(sp.found .| .!sp.labels)
            return 100   # bonus for stopping when all positives found
        else
            return -10  # penalty for stopping too early
        end
    else
        error("Unsupported action type in reward")
    end
end


########################
# Build model + prior
########################

function build_model(n::Int; n_prior::Int=8, rng=Random.MersenneTwister(0))
    prior_states = ASState[]
    for _ in 1:n_prior
        push!(prior_states, sample_random_world(n))
    end
    return ASPOMDP(n, prior_states, 0.3, 1.0)
end

########################
# Run POMCP solver (BasicPOMCP)
########################

function run_pomcp_demo(; n=12, tree_queries=600, max_steps=12)
    rng = MersenneTwister(1)
    pomdp = build_model(n; n_prior=500, rng=rng)

    println("Model built: n_nodes=$(pomdp.n), prior size=$(length(pomdp.graph_prior))")

    # Use stepthrough utility to execute an online planning episode:
    belief = initialstate(pomdp)
    s = sample_random_world(n; rng=rng)  # sample true world from prior
    println("Positive labels: $(count(x -> x == true, s.labels))")

    rewards = Float64[]  # Vector to store rewards at each step
    found_counts = zeros(Int, max_steps)  # number of found positive-class nodes per step

    # Create POMCP solver; see BasicPOMCP README for options
    solver = POMCPSolver(tree_queries=tree_queries, c=1.0, rng=rng)

    println("\nSolving (creating planner)... (this may take a moment)")
    planner = solve(solver, pomdp)   # builds the planner object

    println("\nStepping through an episode using POMCP planner:")

    # Prepare containers for belief visualization
    N_particles = length(belief)
    weights_mat = zeros(Float64, N_particles, max_steps)   # rows=particle, cols=time
    sims_topo_mat = zeros(Float64, N_particles, max_steps) # per-particle topology similarity per time
    weighted_sim = zeros(Float64, max_steps)
    max_idx = zeros(Int, max_steps)                        # index of max-weight particle per time
    for step in 1:max_steps
        a = action(planner, belief)
        # handle Stop action: end episode early
        if a isa Stop
            println("Step $step")
            println("  Action: Stop -- ending episode early")
            break
        end
        sp = rand(transition(pomdp, s, a))
        obs_dist = observation(pomdp, a, sp)
        o = rand(obs_dist)
        r = reward(pomdp, s, a, sp)

        # Compute particle weights given this observation (before resampling)
        weights = compute_particle_weights(belief, o, pomdp)
        # Compute similarity of each particle to the true state s (labels and topology)
        sims = [label_similarity(p, s) for p in belief]
        topo_sims = [topology_similarity(p, s) for p in belief]

        # Record visualization data
        if size(weights_mat, 1) == length(weights)
            weights_mat[:, step] = weights
            sims_topo_mat[:, step] = topo_sims
        end
        weighted_sim[step] = sum(weights .* sims)
        max_idx[step] = argmax(weights)

        println("Step $step")
        # println("  Action: $a")
        # println("  Obs: label=$(o.label) neighbors=$(o.neighbors)")
        # println("  Reward: $r")
        # println("  Top particle index: $(max_idx[step]) with weight $(round(weights[max_idx[step]], digits=4))")
        # println("True state: $s")
        

        # Update belief 
        belief = update_belief_weighted(belief, a, o, pomdp)

        # Store reward
        push!(rewards, r)

        # Count how many positive-class nodes have been found in the true state after this probe
        found_counts[step] = sum(sp.found .& sp.labels)

        s = sp
    end

    println("Episode finished.")
    # println("Final belief: $belief")

    # Plot cumulative rewards and belief diagnostics
    println("Plotting cumulative rewards and belief diagnostics...")
    
    # Determine actual episode length (may be shorter if Stop was chosen early)
    actual_steps = length(rewards)
    
    # If no steps were taken, skip plotting
    if actual_steps == 0
        println("Episode ended immediately (Stop chosen on first step). Skipping plots.")
        return
    end
    
    cum_rewards = cumsum(rewards)

    # Show only a top-K subset of particles for readability when prior is large
    topk = min(10, N_particles)
    # compute mean weight per particle across time (avoid extra imports)
    mean_weights = vec(sum(weights_mat[:, 1:actual_steps], dims=2)) ./ max(actual_steps, 1)
    order = sortperm(mean_weights, rev=true)
    top_idx = order[1:topk]
    weights_top = weights_mat[top_idx, 1:actual_steps]
    sims_topo = sims_topo_mat[top_idx, 1:actual_steps]

    p1 = plot(1:actual_steps, cum_rewards, xlabel="Time Step", ylabel="Cumulative Reward", title="Cumulative Reward", legend=false)

    # Reorder rows per time-step so row 1 = best particle (highest weight) at that timestep
    weights_ranked = similar(weights_top)
    sims_topo_ranked = similar(sims_topo)
    for t in 1:actual_steps
        order_t = sortperm(weights_top[:, t], rev=true)
        weights_ranked[:, t] = weights_top[order_t, t]
        sims_topo_ranked[:, t] = sims_topo[order_t, t]
    end

    # Rows now correspond to rank: 1=best, topk=worst (within top-K) at each timestep
    p2 = heatmap(1:actual_steps, 1:topk, weights_ranked, xlabel="Time Step", ylabel="Rank (1=best)", title="Top-$(topk) Particle Weights Over Time (ranked per-step)", colorbar_title="Weight")
    # For the top-K particles, show topology similarity to the ground truth (heatmap 0..1), ranked per-step
    p3 = heatmap(1:actual_steps, 1:topk, sims_topo_ranked, xlabel="Time Step", ylabel="Rank (1=best)", title="Topology similarity (Jaccard per-node) for top-$(topk) particles (ranked)", colorbar_title="Topo similarity", clim=(0.0,1.0))
    # convert to percentage of positive-class nodes found (0..100)
    found_percent = found_counts[1:actual_steps] ./ max(sum(s.labels), 1) .* 100
    p4 = plot(1:actual_steps, found_percent, xlabel="Time Step", ylabel="Found Positives (%)", title="Percentage of found positive-class nodes over time", legend=false, ylim=(-1, 100.0))

    fig = plot(p1, p4, p2, p3, layout=(4,1), size=(800,1100))
    savefig(fig, "belief_diagnostics.png")
    display(fig)

end

# Run 
println("Running code")
run_pomcp_demo(n=10, tree_queries=20000, max_steps=100)

# TODO:
# - check that topo-noise works properly (seems like changing values doesn't affect performance)
# - compute average over multiple runs/seeds
