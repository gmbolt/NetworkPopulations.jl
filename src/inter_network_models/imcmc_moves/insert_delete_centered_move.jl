using Distributions

export InsertDeleteCenteredMove

struct InsertDeleteCenteredMove <: InvMcmcMove
    ν::Int
    counts::Vector{Int}
    function InsertDeleteCenteredMove(; ν::Int=1)
        new(ν, [0, 0])
    end
end


Base.show(io::IO, x::InsertDeleteCenteredMove) = print(io, "InsertDeleteCenteredMove(ν=$(x.ν))")


function multi_delete_prop_sample!(
    S_curr::InteractionSequence{Int},
    S_prop::InteractionSequence{Int},
    move::InsertDeleteCenteredMove,
    pointers::InteractionSequence{Int},
    ε::Int,
    V::UnitRange
)

    log_ratio = 0.0
    # ind = move.ind_del # Store which entries were deleted 

    n = length(S_curr)
    k = ε
    i = 0
    # j = 0
    live_index = 0
    m_del = Int[] # Lengths of deleted paths 
    while k > 0
        u = rand()
        q = (n - k) / n
        while q > u  # skip
            i += 1
            n -= 1
            q *= (n - k) / n
        end
        i += 1
        @inbounds tmp = popat!(S_prop, i - live_index)
        pushfirst!(pointers, tmp)
        m = length(tmp)
        push!(m_del, m)
        log_ratio += (-m * log(length(V)))
        live_index += 1
        n -= 1
        k -= 1
    end
    # We can now get the insertion distribution centered on remaining paths.
    len_dist = Poisson(mean(length(path) for path in S_prop))
    log_ratio += loglikelihood(len_dist, m_del)

    if length(S_curr) - ε < 1
        log_ratio += -Inf
    end
    return log_ratio

end

function multi_insert_prop_sample!(
    S_curr::InteractionSequence{Int},
    S_prop::InteractionSequence{Int},
    move::InsertDeleteCenteredMove,
    pointers::InteractionSequence{Int},
    ε::Int,
    V::UnitRange
)

    # For insertion method, we start by finding length distribution
    len_dist = Poisson(mean(length(path) for path in S_prop))

    log_ratio = 0.0
    # ind = move.ind_add

    n = length(S_prop) + ε
    k = ε
    i = 0
    # j = 0
    while k > 0
        u = rand()
        q = (n - k) / n
        while q > u  # skip
            i += 1
            n -= 1
            q *= (n - k) / n
        end
        i += 1
        # j += 1
        # Insert path at index i
        # i is now index to insert
        # ind[j] = i
        tmp = popfirst!(pointers) # Get storage 
        m = rand(len_dist)  # Sample length 
        resize!(tmp, m) # Resize 
        sample!(V, tmp) # Sample new entries uniformly
        @inbounds insert!(S_prop, i, tmp) # Insert path into S_prop
        log_ratio += -logpdf(len_dist, m) + m * log(length(V))
        n -= 1
        k -= 1
    end
    return log_ratio

end


function prop_sample!(
    S_curr::InteractionSequence{Int},
    S_prop::InteractionSequence{Int},
    move::InsertDeleteCenteredMove,
    pointers::InteractionSequence{Int},
    V::UnitRange
)

    N = length(S_curr)
    ν = move.ν
    ε = rand(1:ν)
    d = rand(0:min(ε, N))
    a = ε - d
    # move.num_add_del[1] = a
    # move.num_add_del[2] = d

    log_ratio = 0.0
    # @show d, a, S_curr, S_prop
    if d > 0
        log_ratio += multi_delete_prop_sample!(
            S_curr, S_prop,
            move, pointers,
            d, V
        )
    end
    if a > 0
        log_ratio += multi_insert_prop_sample!(
            S_curr, S_prop,
            move, pointers,
            a, V
        )
    end

    M = N - d + a
    log_ratio += log(min(ε, N) + 1) - log(min(ε, M) + 1)

    return log_ratio
end
