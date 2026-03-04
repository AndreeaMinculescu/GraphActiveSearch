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
using Dates

include("likelihoods.jl")
include("utilities.jl")
include("pomcp_debug.jl")

########################
#  POMDP components
########################

# state: adjacency list, true labels, found labels
struct ASState
    graph::Vector{Vector{Int}}   # adjacency list
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
    label::Bool                  # observed label
    neighbors::Vector{Int}       # (noisy) list of observed neighbors
end

########################
# POMDP constructor
########################

struct ASPOMDP <: POMDP{ASState, ASAction, ASObservation}
    n_nodes::Int                     # number of nodes in graph
    topo_noise::Float64              # used by local noisy neighbor sensor
    reward_pos::Float64              # reward for finding a new positive label
    n_particles::Int
end

function build_model(n_nodes::Int; n_particles::Int=1000, rng=MersenneTwister(0))
    return ASPOMDP(n_nodes, 0.3, 2.0, n_particles)
end

########################
#  POMDP components
########################
POMDPs.initialstate(m::ASPOMDP) = ImplicitDistribution(rng -> begin
    sampled = sample_random_world(m.n_nodes; rng=rng)   # sampled is an ASState containing labels
    return sampled
end)

# POMDPs.initialstate(m::ASPOMDP) = Deterministic(sample_random_world(m.n_nodes; seed=0, rng=MersenneTwister(0)))  # deterministic initial state for testing

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
        return Deterministic(ASState(s.graph, s.labels, new_found, false))
    elseif a isa Stop
        return Deterministic(ASState(s.graph, s.labels, new_found, true))
    end
    
end

struct NoisyNeighborDist
    graph::Vector{Vector{Int}}
    labels::Vector{Bool}
    node::Int
    topo_noise::Float64
end

# observation sampling
const FP_RATE_FACTOR = 1/5  # false positive rate factor
function Base.rand(rng::AbstractRNG, d::NoisyNeighborDist)
    true_neighbors = d.graph[d.node]

    observed = Int[]
    for nb in true_neighbors
        rand(rng) < (1-d.topo_noise) && push!(observed, nb)
    end

    fp = d.topo_noise * FP_RATE_FACTOR
    for u in 1:length(d.graph)
        if u != d.node && !(u in true_neighbors)
            rand(rng) < fp && push!(observed, u)
        end
    end

    unique!(observed)
    return ASObservation(d.node, d.labels[d.node], observed)
end

# observation likelihood: might want to make more efficient
function Distributions.pdf(d::NoisyNeighborDist, o::ASObservation)
    # label likelihood
    label_p = 0.95
    p = o.label == d.labels[d.node] ? label_p : 1 - label_p

    true_neighbors = Set(d.graph[d.node])
    observed = Set(o.neighbors)

    fp = d.topo_noise * FP_RATE_FACTOR

    # ----- True neighbors -----
    for u in true_neighbors
        if u in observed
            p *= (1 - d.topo_noise)   # correctly observed
        else
            p *= d.topo_noise         # FALSE NEGATIVE (was missing before)
        end
    end

    # ----- Non-neighbors -----
    for u in 1:length(d.graph)
        if u == d.node || u in true_neighbors
            continue
        end

        if u in observed
            p *= fp                  # false positive
        else
            p *= (1 - fp)            # correctly absent
        end
    end

    return p
end

function POMDPs.observation(m::ASPOMDP, a::ProbeLabel, s::ASState)
    return NoisyNeighborDist(s.graph, s.labels, a.node, m.topo_noise)
end

Base.:(==)(o1::ASObservation, o2::ASObservation) = o1.node == o2.node && o1.label == o2.label && o1.neighbors == o2.neighbors

function Base.hash(o::ASObservation, h::UInt)
    h = hash(o.node, h)
    h = hash(o.label, h)
    h = hash(o.neighbors, h)
    return h
end

const STOP_OBS = ASObservation(-1, false, Int[])
function POMDPs.observation(m::ASPOMDP, a::Stop, s::ASState)
    return Deterministic(STOP_OBS)
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
        true_positives = findall(s.labels)
        all_positives_found = all(s.found[i] for i in true_positives)
        if all_positives_found
            return 20   # bonus for stopping when all positives found
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

function resample_uniform(bp, a, o, b, bb, rng)
    if weight_sum(bp) == 0.0
        ps = particles(bp)
        N = length(ps)
        new_ps = [deepcopy(ps[rand(rng, 1:N)]) for _ in 1:N]
        return WeightedParticleBelief(new_ps)
    else
        return bp
    end
end

function importance_sampling_postprocess(m::ASPOMDP;
        threshold = 0.5, 
        resample = true
    )

    function postprocess(bp, a, o, b, bb, rng)
        a isa ProbeLabel || return bp   # no update on Stop

        println("Running importance sampling postprocess...")
        ps = bp._particles
        ws = copy(bp._weights)
        N  = length(ps)

        # importance reweighting
        for i in 1:N
            ws[i] *= exp(log_likelihood(m, ps[i], a, o))
        end

        normalize!(ws)
        new_bp = WeightedParticleBelief(ps, ws)

        if resample 
            idx = sample(rng, 1:N, Weights(ws), N, replace=true)
            new_ps = [deepcopy(ps[i]) for i in idx]
            new_ws = [deepcopy(ws[i]) for i in idx]
            normalize!(new_ws)
            return WeightedParticleBelief(new_ps, new_ws)
        end

        return new_bp
    end
end


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

function sample_random_world(n_nodes; rng=MersenneTwister(), seed=nothing)
    # Parameters (fixed, not user-set)
    p_edge = 0.15                               # controls graph density
    cluster_prob = 0.9                          # probability of positive label near cluster center
    decay = 0.7                                 # decay of positive label prob with distance  
    max_positives = max(1, floor(Int, n_nodes/3))     # max number of positive-class nodes

    # 1. Generate a random graph
    if seed !== nothing
        g = erdos_renyi(n_nodes, p_edge; seed=seed)
    else
        g = erdos_renyi(n_nodes, p_edge)  
    end
    
    graph = [neighbors(g, i) for i in 1:n_nodes]

    # 2. Initialize all labels negative
    labels = falses(n_nodes)

    # 3. Choose cluster centers
    n_centers = max(1, floor(Int, max_positives/3))     # a third of nodes are cluster centers
    centers = rand(rng, 1:n_nodes, n_centers)
    labels[centers] .= true
    n_pos = length(centers)

    # 4. BFS-order nodes
    queue = collect(centers)
    visited = falses(n_nodes)
    visited[centers] .= true

    while n_pos < max_positives && !isempty(queue)
        v = popfirst!(queue)
        for nb in graph[v]
            if !visited[nb]
                visited[nb] = true
                push!(queue, nb)
                if n_pos < max_positives
                    labels[nb] = true
                    n_pos += 1
                end
            end
        end
    end

    found = falses(n_nodes)
    return ASState(graph, labels, found, false)
end

########################
# Run POMCP solver (BasicPOMCP)
########################


function run_pomcp_demo(; n_nodes=12, tree_queries=600, max_steps=12, verbose=1)
    rng = MersenneTwister(1)
    # Initialize POMDP model
    pomdp = build_model(n_nodes; n_particles=1000, rng=rng)
    # Initialize belief
    # belief = initialize_belief(pomdp, rng)  
    particles = [rand(rng, initialstate(pomdp)) for _ in 1:pomdp.n_particles]  # 200 particles from initial state distribution
    belief = WeightedParticleBelief(particles, ones(length(particles)) ./ length(particles)) # belief = WeightedParticleBelief([init_s], [1.0]) # single particle with true state # belief = DiscreteUpdater(pomdp, belief, rng=rng) # use DiscreteUpdater for belief updates # Initialize particle filter # pf = BootstrapFilter(pomdp, length(particles(belief)); rng=rng, postprocess = metropolis_hastings_postprocess(pomdp; steps=1))
    # pf = BootstrapFilter(pomdp, length(particles(belief)); rng=rng, postprocess = ParticleFilters.PostprocessChain(metropolis_hastings_postprocess(pomdp; steps=1), importance_sampling_postprocess(pomdp; resample=true)))
    # pf = BootstrapFilter(pomdp, length(particles(belief)); rng=rng, postprocess = metropolis_hastings_postprocess(pomdp; steps=1))
    pf = BootstrapFilter(pomdp, pomdp.n_particles)


    # Initialize POMCP solver
    solver = POMCPSolver(tree_queries=tree_queries, c=10, max_depth=50, rng=rng)
    # solver = DebugPOMCPSolver(tree_queries, 10, 10.0, MersenneTwister(1), Inf, false)
    # solver = POMCPOWSolver(tree_queries = tree_queries, max_depth = 50, k_observation = 1.0, alpha_observation = 0.5, rng = rng)
    
    # Track rewards at each step
    rewards = Float64[]  
    # Track number of found positive-class nodes per step
    found_counts = zeros(Int, max_steps)  

    # Prepare containers for belief visualization
    weights_mat = Vector{Float64}[]   # vector of weight vectors (one per time step)
    sims_topo_mat = Vector{Float64}[] # vector of topology similarity vectors (one per time step)
    sims_label_mat = Vector{Float64}[] # vector of label similarity vectors (one per time step)
    max_idx = zeros(Int, max_steps)                        # index of max-weight particle per time
    # Record actions taken at each timestep (for logging)
    actions = ASAction[]
    
    # Storage for particle states and weights across iterations (for table logging)
    particle_snapshots = []  # Vector of (iteration_num, [ASState, ASState, ...]) tuples
    particle_weights_snapshots = []  # Vector of (iteration_num, [w1, w2, ...]) tuples
    

    if verbose >= 1
        println("Model built: n_nodes=$(pomdp.n_nodes), particle filter size=$(length(pomdp.n_particles))")
        println("\nSolving (creating planner)... (this may take a moment)")
        println("\nStepping through an episode using POMCP planner:")
    end

    # Build the planner object
    planner = solve(solver, pomdp) 

    vis_info = nothing  
    bel = nothing
    final_s = nothing

    for (s,a,r,sp,o,b,step) in stepthrough(pomdp, planner, "s,a,r,sp,o,b,t"; max_steps=max_steps)

        bel = b   # stepthrough already updated belief

        push!(rewards, r)

        ####### Visualization data collection #######
        found_counts[step] = sum(sp.found .& sp.labels)

        # weights = belief._weights
        # label_sims = [label_similarity(p, sp) for p in belief._particles]
        # topo_sims  = [topology_similarity(p, sp) for p in belief._particles]

        # push!(weights_mat, weights)
        # push!(sims_topo_mat, topo_sims)
        # push!(sims_label_mat, label_sims)
        push!(actions, a)

        # max_idx[step] = argmax(weights)

        temp_particles = []
        for p in bel._particles
            push!(temp_particles, copy(p.labels))
        end
        push!(particle_snapshots, temp_particles)
        # push!(particle_weights_snapshots, (step, deepcopy(belief._weights)))

        final_s = sp  # update final state

        if verbose >= 1
            println("Step $step")
        end
        println("Number positives: ", sum(sp.labels))

        if verbose >= 2
            println("  True state: $sp")
            println("  Action: $a")
            println("  Obs: label=$(o.label) neighbors=$(o.neighbors)")
            println("  Reward: $r")
            println("  Belief: $(length(bel._particles)) particles")
        end

        if verbose >= 3
            for i in 1:length(bel._particles)
                println("    Particle $i: labels=$(bel._particles[i]), likelihood=$(exp(log_likelihood(pomdp, bel._particles[i], a, o)))")
            end
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
    _, info = action_info(planner, initialstate(pomdp), tree_in_info=true)
    
    date_time = Dates.format(Dates.now(), "yyyy-mm-dd_HHMMSS")
    output_path = joinpath(pwd(), "output")
    mkpath(output_path)
    open(joinpath(output_path, "$(date_time)_tree.html"), "w") do f
        write(f, sprint(show, MIME"text/html"(), D3Tree(info[:tree], init_expand=1)))
    end
    println("Saved D3Tree visualization to $output_path")
    write_visualizations(final_s.labels, final_s.found, rewards, weights_mat, particle_snapshots, sims_topo_mat, sims_label_mat, found_counts, belief, output_path, date_time) 
    println("Saved performance visualizations to $output_path")
end


# Run 
println("Running code")
run_pomcp_demo(n_nodes=5, tree_queries=1000, max_steps=50, verbose=2)

