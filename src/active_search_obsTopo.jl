# active_search_pomcp.jl
# Active Search POMDP + BasicPOMCP 
#
using POMDPs
using POMDPTools.POMDPDistributions: Deterministic, ImplicitDistribution
using POMDPTools.BeliefUpdaters: DiscreteUpdater
using POMDPTools
using Distributions
using BasicPOMCP
using Random
using LinearAlgebra
using LightGraphs
using StatsBase
using ParticleFilters
using Plots  
using D3Trees

include("likelihoods.jl")
include("utilities.jl")

########################
#  POMDP components
########################

# state: adjacency list, true labels, found labels
struct ASState
    labels::Vector{Bool}         # true label per node
    found::Vector{Bool}          # whether node previously probed (for reward)
    terminal::Bool               # whether episode has ended
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
    label::Bool        # observed label
    positive_neighbors::Int  # observed neighbors (noisy)
end

########################
# POMDP constructor
########################

struct ASPOMDP <: POMDP{ASState, ASAction, ASObservation}
    n_nodes::Int                     # number of nodes in graph
    graph::Vector{Vector{Int}}       # topology 
    reward_pos::Float64              # reward for finding a new positive label
    n_particles::Int
end

function build_model(n_nodes::Int, true_topo::Vector{Vector{Int}}; n_particles::Int=50, rng=Random.MersenneTwister(0))
    # prior_labels = Vector{ASState}()
    # for _ in 1:n_prior
    #     # push!(prior_labels, sample_random_labels(true_topo))
    #     push!(prior_labels, true_state)  # for testing, use true state as prior (no uncertainty)
    #     # push!(prior_labels, ASState(1 .- true_state.labels, falses(n_nodes), false))  # for testing, use true state as prior (no uncertainty)
    # end
    # # push!(prior_states, s)  # ensure true state is in prior
    return ASPOMDP(n_nodes, true_topo, 3.0, n_particles)
end

########################
#  POMDP components
########################
# POMDPs.initialstate(m::ASPOMDP) = ImplicitDistribution(rng -> begin
#     sampled = rand(rng, m.label_prior)   # sampled is an ASState containing labels
#     return ASState(deepcopy(sampled.labels), falses(m.n_nodes), false)
# end)

POMDPs.initialstate(m::ASPOMDP) = ImplicitDistribution(rng -> begin
    sampled = sample_random_labels(m.graph; rng)   # sampled is an ASState containing labels
    return sampled
end)

# POMDPs.initialstate(m::ASPOMDP) = Deterministic(sample_random_labels(m.graph; rng=MersenneTwister(0)))  # deterministic initial state for testing

POMDPs.isterminal(m::ASPOMDP, s::ASState) = s.terminal

# actions: probe any node's label, or stop
POMDPs.actions(m::ASPOMDP) = vcat([ProbeLabel(i) for i in 1:m.n_nodes], [Stop()])

# transition: graph and labels static; 'found' updated when label probe occurs
function POMDPs.transition(m::ASPOMDP, s::ASState, a::ASAction)
    if s.terminal
        return Deterministic(s)  # no change if already terminal
    end

    new_found = copy(s.found)
    if a isa ProbeLabel
        new_found[a.node] = true
        return Deterministic(ASState(s.labels, new_found, false))
    elseif a isa Stop
        return Deterministic(ASState(s.labels, new_found, true))
    end
    
end


function POMDPs.observation(m::ASPOMDP, a::ProbeLabel, s::ASState)
    true_label = s.labels[a.node]
    no_pos_neighbors = count(s.labels[j] for j in m.graph[a.node])
    return Deterministic(ASObservation(a.node, true_label, no_pos_neighbors))
end

Base.:(==)(o1::ASObservation, o2::ASObservation) = o1.node == o2.node && o1.label == o2.label && o1.positive_neighbors == o2.positive_neighbors

function Base.hash(o::ASObservation, h::UInt)
    h = hash(o.node, h)
    h = hash(o.label, h)
    h = hash(o.positive_neighbors, h)
    return h
end

const STOP_OBS = ASObservation(-1, false, 0)
function POMDPs.observation(m::ASPOMDP, a::Stop, s::ASState)
    return Deterministic(STOP_OBS)
end

# reward: +reward_pos when finding a previously-unfound positive; -1 per probe; -10 for stopping too early; +100 bonus for stopping when all positives found
function POMDPs.reward(m::ASPOMDP, s::ASState, a::ASAction, sp::ASState)
    # println("   Calculating reward for action: ", s.labels)
    if a isa ProbeLabel
        v = a.node
        # if s.found[v]
        #     return -20
        # end
        return (s.labels[v] && !s.found[v]) ? m.reward_pos : -1
    elseif a isa Stop
        # Check if all nodes that are TRULY positive have been found
        true_positives = findall(s.labels)
        all_positives_found = all(s.found[i] for i in true_positives)
        if all_positives_found
            return 20  # bonus for stopping when all positives found
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


function metropolis_hastings_postprocess(m::ASPOMDP; steps=1)
    function postjuprocess(bp, a, o, b, bb, rng)
        if weight_sum(bp) < 0.5 * length(particles(bp))
            println("Running MH postprocess...")
            a isa ProbeLabel || return bp

            ps = copy(bp._particles)
            ws = bp._weights

            for i in eachindex(ps)
                s = ps[i]
                for _ in 1:steps
                    s_prop = deepcopy(s)
                    propose_local_move!(s_prop, rng)

                    log_mh =
                        (log_prior(s_prop) - log_prior(s)) +
                        (log_likelihood(m, s_prop, a, o) -
                        log_likelihood(m, s, a, o))

                    if log(rand(rng)) < log_mh
                        s = s_prop
                    end
                end
                ps[i] = s
            end
            
            normalize!(ws)
            return WeightedParticleBelief(ps, ws)
        else 
            return bp
        end
    end
end



########################
# Prior world sampler (connected ER + clustered positives)
########################

function sample_random_labels(graph::Vector{Vector{Int}}; rng=MersenneTwister())
    n_nodes = length(graph) 
    # Parameters (fixed, not user-set)                             # controls graph density
    cluster_prob = 0.9                          # probability of positive label near cluster center
    decay = 0.7                                 # decay of positive label prob with distance  
    max_positives = max(1, floor(Int, n_nodes/2))     # max number of positive-class nodes

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

    return ASState(labels, falses(n_nodes), false)
end

########################
# Run POMCP solver (BasicPOMCP)
########################

struct DebugUnweighted{M,RNG} <: Updater
    model::M
    rng::RNG
end

function POMDPs.update(up::DebugUnweighted, b, a, o)

    println("\n==========================")
    println("DEBUG FILTER UPDATE")
    println("Action: ", a)
    println("Observation: ", o)
    println("Current particles: ", length(particles(b)))
    println("==========================")

    new_particles = ASState[]
    i = 1

    for s in particles(b)
        println("\nParticle $i BEFORE transition:")
        println("  s = ", s)

        # transition
        sp = rand(up.rng, transition(up.model, s, a))
        println("  sp (after transition) = ", sp)

        # predicted observation
        o_pred = rand(up.rng, observation(up.model, a, sp))
        println("  predicted observation = ", o_pred)

        print(typeof(o_pred), typeof(o))
        println("  matches actual? → ", o_pred == o)
        println("  node equal? ", o_pred.node == o.node)
        println("  label equal? ", o_pred.label == o.label)
        println("  struct equal? ", o_pred == o)
        println("  same object? ", o_pred === o)

        if o_pred == o
            push!(new_particles, sp)
        end

        i += 1
    end

    println("\nParticles surviving: ", length(new_particles))

    if isempty(new_particles)
        println("!!! PARTICLE DEPLETION !!!")
    end

    return ParticleCollection(new_particles)
end


function run_pomcp_demo(; n_nodes=12, tree_queries=600, max_steps=12, verbose=1)
    rng = MersenneTwister(1)
    # Sample true topology
    true_topo = [neighbors(erdos_renyi(n_nodes, 0.15), i) for i in 1:n_nodes]
    # Initialize POMDP model
    pomdp = build_model(n_nodes, true_topo; n_particles=10000, rng=rng)
    # Initialize belief
    # belief = initialize_belief(pomdp, rng)  
    particles = [rand(rng, initialstate(pomdp)) for _ in 1:pomdp.n_particles]  # 200 particles from initial state distribution
    belief = WeightedParticleBelief(particles, ones(length(particles)) ./ length(particles)) # belief = WeightedParticleBelief([init_s], [1.0]) # single particle with true state # belief = DiscreteUpdater(pomdp, belief, rng=rng) # use DiscreteUpdater for belief updates # Initialize particle filter # pf = BootstrapFilter(pomdp, length(particles(belief)); rng=rng, postprocess = metropolis_hastings_postprocess(pomdp; steps=1))
    # pf = BootstrapFilter(pomdp, length(particles(belief)); rng=rng, postprocess = ParticleFilters.PostprocessChain(metropolis_hastings_postprocess(pomdp; steps=1), importance_sampling_postprocess(pomdp; resample=true)))
    # pf = BootstrapFilter(pomdp, length(particles(belief)); rng=rng, postprocess = metropolis_hastings_postprocess(pomdp; steps=1))
    pf = BootstrapFilter(pomdp, pomdp.n_particles)


    # Initialize POMCP solver
    solver = POMCPSolver(tree_queries=tree_queries, c=10, max_depth=50, rng=rng)
    # solver = POMCPOWSolver(tree_queries = tree_queries, max_depth = 50, k_observation = 1.0, alpha_observation = 0.5, rng = rng)
    
    # Track rewards at each step
    rewards = Float64[]  
    # Track number of found positive-class nodes per step
    found_counts = zeros(Int, max_steps)  

    # Prepare containers for belief visualization
    # N_particles = length(particles(belief))                # number of particles
    weights_mat = Vector{Float64}[]   # vector of weight vectors (one per time step)
    sims_topo_mat = Vector{Float64}[] # vector of topology similarity vectors (one per time step)
    sims_label_mat = Vector{Float64}[] # vector of label similarity vectors (one per time step)
    max_idx = zeros(Int, max_steps)                        # index of max-weight particle per time
    # Record actions taken at each timestep (for logging)
    actions = ASAction[]
    
    # Storage for particle states and weights across iterations (for table logging)
    particle_snapshots = Vector{Vector{Vector{Bool}}}()  # Vector of (iteration_num, [ASState, ASState, ...]) tuples
    particle_weights_snapshots = []  # Vector of (iteration_num, [w1, w2, ...]) tuples
    

    if verbose >= 1
        println("Model built: n_nodes=$(pomdp.n_nodes), particle filter size=$(length(pomdp.n_particles))")
        # println("Positive labels: $(findall(init_s.labels)); n=$(sum(init_s.labels))")
        println("\nSolving (creating planner)... (this may take a moment)")
        println("\nStepping through an episode using POMCP planner:")
    end

    # Build the planner object
    planner = solve(solver, pomdp) 
    vis_info = nothing  
    bel = nothing
    final_s = nothing

    debug_updater = DebugUnweighted(pomdp, rng)

    
    for (s,a,r,sp,o,b,step) in stepthrough(pomdp, planner, "s,a,r,sp,o,b,t"; max_steps=max_steps)

        # belief = b   # stepthrough already updated belief
        bel = b
        push!(rewards, r)

        ####### Visualization data collection #######
        found_counts[step] = sum(sp.found .& sp.labels)

        # weights = belief.weights
        # label_sims = [label_similarity(p, sp) for p in belief.particles]

        # push!(weights_mat, weights)
        # push!(sims_label_mat, label_sims)
        push!(actions, a)

        # max_idx[step] = argmax(weights)
        temp_particles = []
        for p in bel._particles
            push!(temp_particles, copy(p.labels))
        end
        push!(particle_snapshots, temp_particles)
        # push!(particle_snapshots, [copy(p.labels) for p in bel.particles])
        # push!(particle_weights_snapshots, (step, deepcopy(belief.weights)))

        final_s = sp  # update final state
        # if step >= 1
            # _, vis_info = action_info(planner, b, tree_in_info=true)
            # print(b)
            # inchrome(D3Tree(vis_info[:tree], init_expand=5))
        # end

        if verbose >= 1
            println("Step $step")
        end

        if verbose >= 2
            println("  True state: $sp")
            println("  Action: $a")
            println("  Obs: label=$(o.label)")
            println("  Reward: $r")
            println("  Belief: $(length(bel._particles)) particles")
        end

        if a isa Stop
            println("  Action: Stop -- ending episode early")
            break
        end

        
    end
    if verbose >= 1 
        println("Episode finished.") 
        println("Plotting cumulative rewards and belief diagnostics...") 
    end 
    
    ####### Plotting ####### 
    println(rewards)

    _, info = action_info(planner, initialstate(pomdp), tree_in_info=true)
    # inchrome(D3Tree(info[:tree], init_expand=1))
    # Save HTML for host
    html_file = "/app/output/tree.html"
    open(html_file, "w") do f
        write(f, sprint(show, MIME"text/html"(), D3Tree(info[:tree], init_expand=1)))
    end
    println("Saved D3Tree visualization to $html_file")

    write_visualizations(final_s.labels, final_s.found, rewards, weights_mat, particle_snapshots, sims_topo_mat, sims_label_mat, found_counts, belief) 
    
    # Write run log using helper function write_run_log(pf, belief, particle_snapshots, particle_weights_snapshots, s, rewards, actions)
    # write_run_log(pf, belief, particle_snapshots, particle_weights_snapshots, final_s.labels, true_topo, final_s.found, rewards, actions)

end


# Run 
println("Running code")
run_pomcp_demo(n_nodes=5, tree_queries=10000, max_steps=100, verbose=2)

# TODO:
# - init_s not passed as the true state?
# - check that topo-noise works properly (seems like changing values doesn't affect performance)
# - compute average over multiple runs/seeds
# - node in ASObservation needed?

