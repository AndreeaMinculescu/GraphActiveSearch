include("active_search_obsTopo.jl")

function test_optimality(max_steps=10; verbose=true)
    rng = MersenneTwister(42)

    # Fixed graph & labels for reproducibility
    true_topo = [[2,3], [1,4], [1,5], [2], [3]]  # small graph
    true_labels = [true, false, true, false, false]
    n_nodes = length(true_labels)
    true_state = ASState(true_labels, falses(n_nodes), false)

    # Build model
    pomdp = build_model(n_nodes, true_topo, true_state; n_prior=10, rng=rng)

    # Run POMCP
    rewards = Float64[]
    final_s = nothing
    run_pomcp_demo(n_nodes=n_nodes, tree_queries=500, max_steps=max_steps, verbose=0)
    # For rigorous test, store rewards from run_pomcp_demo and final_s

    # Compute optimal possible reward
    reward_pos = pomdp.reward_pos
    max_reward = sum(reward_pos for l in true_labels if l)  # sum of positives
    stop_bonus = 10  # if all positives found
    optimal_reward = max_reward + stop_bonus

    # Evaluate outcome
    found_count = sum(final_s.found[i] && true_labels[i] for i in 1:n_nodes)
    success = (found_count == sum(true_labels))

    if verbose
        println("Optimal reward possible: $optimal_reward")
        println("Cumulative reward obtained: $(sum(rewards))")
        println("All positives found? $success")
    end

    return (sum(rewards), optimal_reward, success)
end

test_optimality()