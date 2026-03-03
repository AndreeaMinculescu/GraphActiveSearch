using POMDPs
using POMDPTools.POMDPDistributions: Deterministic, ImplicitDistribution
using POMDPTools.BeliefUpdaters: DiscreteUpdater
using POMDPTools
using Distributions
using BasicPOMCP: POMCPTree, POMCPObsNode, stepthrough, insert_obs_node!

struct DebugPOMCPSolver <: Solver
    tree_queries::Int
    max_depth::Int
    c::Float64
    rng::AbstractRNG
    max_time::Float64
    tree_in_info::Bool
end


mutable struct DebugPOMCPPlanner{S,P,RNG} <: Policy
    solver::S
    problem::P
    rng::RNG
    _tree
    _best_node_mem::Vector{Int}
    time
end

function DebugPOMCPPlanner(solver::DebugPOMCPSolver, pomdp)
    return DebugPOMCPPlanner(
        solver,
        pomdp,
        solver.rng,
        nothing,
        Int[],
        time
    )
end

POMDPs.solve(solver::DebugPOMCPSolver, pomdp::POMDP) = DebugPOMCPPlanner(solver, pomdp)

function search(p::DebugPOMCPPlanner, b, t::POMCPTree, info::Dict)
    println("=========== NEW SEARCH ===========")
    try
        println("Belief has $(length(b.particles)) particles")
        for i in 1:length(b.particles)
                println("    Particle $i: labels=$(b.particles[i])")
            end
    catch ex
    end

    # If weighted belief
    if hasproperty(b, :weights)
        ws = b.weights
        idx = sortperm(ws, rev=true)
        println("Top 5 particles by weight:")
        for i in 1:min(5,length(ws))
            j = idx[i]
            println("  idx=$j weight=$(ws[j]) labels=$(b.particles[j].labels)")
        end
    end

    nquery = 0
    for i in 1:p.solver.tree_queries
        nquery += 1

        s = rand(p.rng, b)

        println("\n--- Simulation $i ---")
        println("Sampled particle labels: ", s.labels)
        println("Sampled particle graph: ", s.graph)
        println("Found: ", s.found)

        if !isterminal(p.problem, s)
            simulate(p, s, POMCPObsNode(t, 1), p.solver.max_depth)
        end
    end

    # identical action selection logic as before
    h = 1
    best_node = first(t.children[h])
    best_v = t.v[best_node]
    for node in t.children[h][2:end]
        if t.v[node] >= best_v
            best_v = t.v[node]
            best_node = node
        end
    end

    return t.a_labels[best_node]
end

function simulate(p::DebugPOMCPPlanner, s, hnode::POMCPObsNode, steps::Int)

    if steps == 0 || isterminal(p.problem, s)
        return 0.0
    end

    t = hnode.tree
    h = hnode.node

    println("Simulating at depth $steps")
    println("Current state labels: ", s.labels)
    println("Current state graph: ", s.graph)

    # ===== UCB ACTION SELECTION =====
    ltn = log(t.total_n[h] + 1e-10)
    best_nodes = empty!(p._best_node_mem)
    best_val = -Inf

    for node in t.children[h]
        n = t.n[node]
        if n == 0
            criterion = Inf
        else
            criterion = t.v[node] + p.solver.c*sqrt(ltn/n)
        end

        println("UCB")
        println("   Node $(t.a_labels[node]): criterion=$criterion")
        println("   t.v[node] = ", t.v[node], ", ltn = ", ltn, ", n = ", n)
        if criterion > best_val
            best_val = criterion
            empty!(best_nodes)
            push!(best_nodes, node)
        elseif criterion == best_val
            push!(best_nodes, node)
        end
    end

    ha = rand(p.rng, best_nodes)
    a = t.a_labels[ha]

    println("   Selected action: ", a)

    # ===== GENERATIVE STEP =====
    sp, o, r = @gen(:sp, :o, :r)(p.problem, s, a, p.rng)

    println("Generated observation: ", o)
    println("Reward: ", r)

    # Check consistency
    d = observation(p.problem, a, s)
    likelihood = pdf(d, o)
    println("Likelihood under state: ", likelihood)

    if likelihood == 0.0
        println("⚠️  INCONSISTENT OBSERVATION")
    end

    # ===== TREE UPDATE =====
    hao = get(t.o_lookup, (ha, o), 0)

    if hao == 0
        println("New observation node created.")
        hao = insert_obs_node!(t, p.problem, ha, sp, o)
        v = 0.0
        R = r + discount(p.problem)*v
    else
        R = r + discount(p.problem)*simulate(p, sp, POMCPObsNode(t, hao), steps-1)
    end

    t.total_n[h] += 1
    t.n[ha] += 1
    t.v[ha] += (R - t.v[ha]) / t.n[ha]

    return R
end

function POMDPTools.action_info(p::DebugPOMCPPlanner, b; tree_in_info=false)
    local a::actiontype(p.problem)
    info = Dict{Symbol, Any}()
    try
        tree = POMCPTree(p.problem, b, p.solver.tree_queries)
        a = search(p, b, tree, info)
        p._tree = tree
        if p.solver.tree_in_info || tree_in_info
            info[:tree] = tree
        end
    catch ex
        # Note: this might not be type stable, but it shouldn't matter too much here
        println("!!!!!!!  Exception during action selection: ", ex)

        # a = convert(actiontype(p.problem), default_action(p.solver.default_action, p.problem, b, ex))
        a = nothing
        info[:exception] = ex
    end
    return a, info
end

POMDPs.action(p::DebugPOMCPPlanner, b) = first(action_info(p, b))

function POMDPs.updater(p::DebugPOMCPPlanner)
    P = typeof(p.problem)
    S = statetype(P)
    A = actiontype(P)
    O = obstype(P)
    return UnweightedParticleFilter(p.problem, p.solver.tree_queries, rng=p.rng)
    # XXX It would be better to automatically use an SIRParticleFilter if possible
    # if !@implemented ParticleFilters.obs_weight(::P, ::S, ::A, ::S, ::O)
    #     return UnweightedParticleFilter(p.problem, p.solver.tree_queries, rng=p.rng)
    # end
    # return SIRParticleFilter(p.problem, p.solver.tree_queries, rng=p.rng)
end
