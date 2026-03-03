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

function observation_likelihood(p, o, topo_noise::Float64)
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
function compute_particle_weights(belief, o, m)
    N = length(particles(belief))
    lik = zeros(Float64, N)
    # If no observation/sentinel (Stop), return uniform weights
    if o.node == -1
        lik .= 1.0 / N
        return lik
    end

    for i in 1:N
        lik[i] = observation_likelihood(particles(belief)[i], o, m.topo_noise)
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
function label_similarity(p, s)
    return sum(p.labels .== s.labels) / length(s.labels)
end

# Topology similarity between two states: normalized graph edit distance
# GED is computed as the number of edge edits (additions/deletions) needed to transform one graph to another
# Normalized by dividing by the maximum possible edits (n*(n-1))
function topology_similarity(p, s)
    n = length(s.graph)
    
    # Convert adjacency lists to edge sets
    edges_p = Set()
    edges_s = Set()
    
    for i in 1:n
        for j in p.graph[i]
            # Store edges as sorted tuples to avoid (i,j) and (j,i) being different
            edge = i < j ? (i, j) : (j, i)
            push!(edges_p, edge)
        end
    end
    
    for i in 1:n
        for j in s.graph[i]
            edge = i < j ? (i, j) : (j, i)
            push!(edges_s, edge)
        end
    end
    
    # Compute graph edit distance as symmetric difference of edges
    edits = length(symdiff(edges_p, edges_s))
    
    # Normalize by maximum possible edits: n*(n-1)/2 (complete graph)
    max_edits = n * (n - 1) / 2
    
    # Return similarity (1 - normalized distance)
    similarity = 1.0 - (edits / max(max_edits, 1.0))
    return max(0.0, similarity)
end


##### Metropolis-Hastings #######
function propose_local_move!(s, rng)
    n = length(s.labels)

    if rand(rng) < 5
        # Flip one unobserved label
        i = (c = findall(x -> !x, s.found); isempty(c) && return; rand(rng, c))
        s.labels[i] = !s.labels[i]
    end
    if rand(rng) < 5
        # Toggle one edge (graph is undirected)
        i, j = rand(rng, 1:n, 2)
        i == j && return

        if j in s.graph[i]
            deleteat!(s.graph[i], findfirst(==(j), s.graph[i]))
            deleteat!(s.graph[j], findfirst(==(i), s.graph[j]))
        else
            push!(s.graph[i], j)
            push!(s.graph[j], i)
        end
    end
end

function log_prior(s)
    # Penalize too many positives
    prob = -0.5 * sum(s.labels)

    # clustering reward
    for i in 1:length(s.graph)
        for j in s.graph[i]
            prob += (s.labels[i] == s.labels[j]) ? 0.1 : -0.1
        end
    end

    return prob
end

function log_likelihood(m, s, a, o)
    obs_dist = observation(m, a, s)
    return log(pdf(obs_dist, o))

    # v = a.node
    # ll = 0.0
    # fp = m.topo_noise / 5 # false positive rate

    # # label likelihood (assumed noise-free here)
    # ll += (s.labels[v] == o.label) ? 0.0 : -Inf

    # # true and observed neighbors
    # true_nb = Set(s.graph[v])
    # obs_nb  = Set(o.neighbors)

    # # false negatives
    # for u in true_nb
    #     if !(u in obs_nb)
    #         ll += log(m.topo_noise)
    #     end
    # end

    # # false positives
    # for u in obs_nb
    #     if !(u in true_nb)
    #         ll += log(fp)
    #     end
    # end

    # return ll
end




