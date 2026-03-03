# active_search_pomcp.jl
# Active Search POMDP + BasicPOMCP 
#
using POMDPs
using POMDPTools.POMDPDistributions: Deterministic, ImplicitDistribution
using POMDPTools.BeliefUpdaters: DiscreteUpdater
using BasicPOMCP
using Random
using LinearAlgebra
using LightGraphs
using StatsBase
using ParticleFilters
using Plots  

include("likelihoods.jl")

########################
# POMDP constructor
########################

struct ASPOMDP <: POMDP{ASState, ASAction, ASObservation}
    n_nodes::Int                     # number of nodes in graph
    graph_prior::Vector{ASState}     # prior as vector of ASState examples
    topo_noise::Float64              # used by local noisy neighbor sensor
    reward_pos::Float64              # reward for finding a new positive label
end

function build_model(n_nodes::Int, s::ASState; n_prior::Int=8, rng=Random.MersenneTwister(0))
    prior_states = ASState[]
    for _ in 1:n_prior
        push!(prior_states, sample_random_world(n_nodes))
    end
    # push!(prior_states, s)  # ensure true state is in prior
    return ASPOMDP(n_nodes, prior_states, 0.3, 1.0)
end

########################
#  POMDP components
########################

# state: adjacency list, true labels, found labels
struct ASState
    graph::Vector{Vector{Int}}   # adjacency list
    labels::Vector{Bool}         # true label per node
    found::Vector{Bool}          # whether node previously probed (for reward)
end

# action: probe label of a node, or stop
abstract type ASAction end
struct ProbeLabel <: ASAction
    node::Int                    # node chosen for probing
end
# Stop action: end the episode 
struct Stop <: ASAction
end

# observation: label and noisy neighbors
struct ASObservation
    node::Int                    # the probed node index
    label::Bool                  # observed label
    neighbors::Vector{Int}       # (noisy) list of observed neighbors
end

# initial belief: return ParticleCollection of prior ASState objects
function POMDPs.initialstate(m::ASPOMDP)
    #return BootstrapFilter(m, 20)
    #return ParticleCollection(copy(m.graph_prior))
    return sample_random_world(m.n_nodes)
end

# actions: probe any node's label, or stop
POMDPs.actions(m::ASPOMDP) = vcat([ProbeLabel(i) for i in 1:m.n_nodes], [Stop()])

# transition: graph and labels static; 'found' updated when label probe occurs
function POMDPs.transition(m::ASPOMDP, s::ASState, a::ASAction)
    new_found = copy(s.found)
    if a isa ProbeLabel
        new_found[a.node] = true
    end
    return Deterministic(ASState(s.graph, s.labels, new_found))
end

# observation: label and noisy neighbors
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
        # Return a harmless ASObservation sentinel (node=-1) so planners that
        # expect an ASObservation type do not fail when observing a Stop.
        return Deterministic(ASObservation(-1, false, Int[]))
    else
        error("Unsupported action type in observation")
    end
end

# reward: +reward_pos when finding a previously-unfound positive; -1 per probe; -10 for stopping too early; +100 bonus for stopping when all positives found
function POMDPs.reward(m::ASPOMDP, s::ASState, a::ASAction, sp::ASState)
    if a isa ProbeLabel
        v = a.node
        # if s.found[v]
        #     return -20
        # end
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

# discount factor
POMDPs.discount(::ASPOMDP) = 0.9

########################
# Particle filter
########################

# update belief: compute weights and resample
# function POMDPs.update(m::ASPOMDP, belief::BootstrapFilter, a::ASAction, o::ASObservation)
#     particles_list = [p for p in particles(belief)]   # gets vector of ASState
#     N = length(particles_list)
#     if N == 0
#         error("Belief is empty; cannot update")
#     end

#     # compute likelihoods for each particle
#     lik = zeros(Float64, N)
#     for i in 1:N
#         lik[i] = observation_likelihood(particles_list[i], o, m.topo_noise)
#     end

#     # normalize
#     s = sum(lik)
#     if s == 0.0
#         # lik .= 1.0 / N
#         error("Weights are empty")
#     else
#         lik ./= s
#     end

#     # resample indices with replacement using StatsBase
#     new_indices = sample(1:N, Weights(lik), N; replace=true)
#     new_particles = [deepcopy(particles_list[i]) for i in new_indices] 

#     # return new ParticleCollection
#     return ParticleCollection(new_particles)
# end

# initialize belief: return ParticleCollection of ASState objects
# ParticleFilters.initialize_belief(m::ASPOMDP, dist) = ParticleCollection(copy(m.graph_prior))
# ParticleFilters.initialize_belief(m::ASPOMDP, rng) = BootstrapFilter(m, 50)

########################
# Prior world sampler (connected ER + clustered positives)
########################

function sample_random_world(n_nodes; rng=MersenneTwister())
    # Parameters (fixed, not user-set)
    p_edge = 0.15                               # controls graph density
    cluster_prob = 0.9                          # probability of positive label near cluster center
    decay = 0.7                                 # decay of positive label prob with distance  
    max_positives = max(1, floor(Int, n_nodes/3))     # max number of positive-class nodes

    # 1. Generate a random graph
    g = erdos_renyi(n_nodes, p_edge)
    graph = [neighbors(g, i) for i in 1:n_nodes]

    # 2. Initialize all labels negative
    labels = falses(n_nodes)

    # 2. Choose cluster centers
    n_centers = max(1, floor(Int, max_positives/3))     # a third of nodes are cluster centers
    centers = rand(rng, 1:n_nodes, n_centers)                 # randomly choose centers
    labels[centers] .= true                             # cluster centers are positive
    n_pos = length(centers)                             # count initial positives

    # 4. BFS-based clustering, but strictly cap total positives
    for c in centers
        if n_pos >= max_positives
            break
        end

        visited = falses(n_nodes)
        queue = [(c, 0)]
        visited[c] = true
        while !isempty(queue)
            if n_pos >= max_positives
                break
            end
            (v, depth) = popfirst!(queue)
            # the closer a node is to the cluster center, the higher the probability of becoming positive
            p = cluster_prob * (decay^depth)    
            # a node becomes positive if it is negative and it passes the probability test
            if !labels[v] && rand(rng) < p
                labels[v] = true
                n_pos += 1
                if n_pos >= max_positives
                    break
                end
            end
            # add neighbors of the new positive nodes to queue
            for nb in graph[v]
                if !visited[nb]
                    visited[nb] = true
                    push!(queue, (nb, depth + 1))
                end
            end
        end
    end

    found = falses(n_nodes)
    return ASState(graph, labels, found)
end

########################
# Run POMCP solver (BasicPOMCP)
########################

function run_pomcp_demo(; n_nodes=12, tree_queries=600, max_steps=12, verbose=1)
    rng = MersenneTwister(1)
    # Sample true world 
    s = sample_random_world(n_nodes; rng=rng)  
    # Initialize POMDP model
    pomdp = build_model(n_nodes, s; n_prior=50, rng=rng)
    # Initialize belief
    # belief = initialize_belief(pomdp, rng)  
    belief = ParticleCollection(copy(pomdp.graph_prior))
    pf = BootstrapFilter(pomdp, 50; rng=rng)
    # Initialize POMCP solver
    solver = POMCPSolver(tree_queries=tree_queries, c=1.0, rng=rng)
    
    # Track rewards at each step
    rewards = Float64[]  
    # Track number of found positive-class nodes per step
    found_counts = zeros(Int, max_steps)  

    # Prepare containers for belief visualization
    # N_particles = length(particles(belief))                # number of particles
    # weights_mat = zeros(Float64, N_particles, max_steps)   # rows=particle, cols=time
    # sims_topo_mat = zeros(Float64, N_particles, max_steps) # per-particle topology similarity per time
    # weighted_sim = zeros(Float64, max_steps)               # label similarity per time
    # max_idx = zeros(Int, max_steps)                        # index of max-weight particle per time
    

    if verbose >= 1
        println("Model built: n_nodes=$(pomdp.n_nodes), prior size=$(length(pomdp.graph_prior))")
        println("Positive labels: $(count(x -> x == true, s.labels))")
        println("\nSolving (creating planner)... (this may take a moment)")
        println("\nStepping through an episode using POMCP planner:")
    end

    # Build the planner object
    planner = solve(solver, pomdp)   

    for step in 1:max_steps
        a = action(planner, belief)
        sp = rand(transition(pomdp, s, a))
        obs_dist = observation(pomdp, a, sp)
        o = rand(obs_dist)
        r = reward(pomdp, s, a, sp)

        # handle Stop action: end episode early
        if a isa Stop
            println("Step $step")
            println("  Action: Stop -- ending episode early")
            break
        end

        if verbose >= 1
            println("Step $step")
        elseif verbose >= 2
            println("  True state: $s")
            println("  Action: $a")
            println("  Obs: label=$(o.label) neighbors=$(o.neighbors)")
            println("  Reward: $r")
            println("  Top particle index: $(max_idx[step]) with weight $(round(weights[max_idx[step]], digits=4))")
            
        end
        
        # Store reward
        push!(rewards, r)

        # Update belief 
        # belief = update(pomdp, belief, a, o)
        belief = update(pf, belief, a, o)
        # Count how many positive-class nodes have been found in the true state after this probe
        s = sp

        ####### Visualization data collection #######
        # Compute particle weights given this observation (before resampling)
        found_counts[step] = sum(sp.found .& sp.labels)
        weights = compute_particle_weights(belief, o, pomdp)
        # Compute similarity of each particle to the true state s (labels and topology)
        sims = [label_similarity(p, s) for p in particles(belief)]
        topo_sims = [topology_similarity(p, s) for p in particles(belief)]

        if size(weights_mat, 1) == length(weights)
            weights_mat[:, step] = weights
            sims_topo_mat[:, step] = topo_sims
        end
        weighted_sim[step] = sum(weights .* sims)
        max_idx[step] = argmax(weights)
    end

    if verbose >= 1
        println("Episode finished.")
        println("Plotting cumulative rewards and belief diagnostics...")
    end
    
    
    ####### Plotting #######
    # Determine actual episode length (may be shorter if Stop was chosen early)
    actual_steps = length(rewards)-1

    # If no steps were taken, skip plotting
    if actual_steps == 0
        println("Episode ended immediately (Stop chosen on first step). Skipping plots.")
        return
    end
    
    cum_rewards = cumsum(rewards)

    # Show only a top-K subset of particles for readability when prior is large
    topk = min(10, N_particles)
    # compute mean weight per particle across time
    mean_weights = vec(sum(weights_mat[:, 1:actual_steps], dims=2)) ./ max(actual_steps, 1)
    order = sortperm(mean_weights, rev=true)
    top_idx = order[1:topk]
    weights_top = weights_mat[top_idx, 1:actual_steps]
    sims_topo = sims_topo_mat[top_idx, 1:actual_steps]

    p1 = plot(1:length(cum_rewards), cum_rewards, xlabel="Time Step", ylabel="Cumulative Reward", title="Cumulative Reward", legend=false)

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
run_pomcp_demo(n_nodes=25, tree_queries=20000, max_steps=100)

# TODO:
# - check that topo-noise works properly (seems like changing values doesn't affect performance)
# - compute average over multiple runs/seeds
# - node in ASObservation needed?