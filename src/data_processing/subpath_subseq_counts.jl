"""
Functions to summarise the distribution of subsequences/subpaths appearing in interaction 
networks or samples of interaction sequences, e.g. to get the average number of times 
a given subsequence appears in an observation, or the proportion of interaction sequences in which 
a given subsequence appears (at least once).
"""

using IterTools

export get_subseqs!, get_subseqs, get_subpaths!, get_subpaths, get_subpath_counts, get_subseq_counts
export get_unique_subseqs, get_unique_subseq_counts, get_unique_subseq_proportions
export get_unique_subpaths!, get_unique_subpaths, get_unique_subpath_counts!, get_unique_subpath_counts!, get_unique_subpath_proportions
export subseq_isin, subpath_isin
export dict_rank

# subsequences
# ------------

"""
    get_subseqs!(out::Dict{Vector{T},Int}, S::InteractionSequence{T}, r::Int) 

Augment `out` with counts of length `r` subsequences appearing in `S`.
"""
function get_subseqs!(
    out::Dict{Vector{T},Int},
    S::InteractionSequence{T}, 
    r::Int    
    ) where {T<:Union{Int,String}}
    for path in S 
        n = length(path)
        for ind in subsets(1:n, r)
            out[path[ind]] = get(out, path[ind], 0) + 1
        end 
    end 
end 

"""
    get_subseqs(S, r; drop_zero, vertex_set) 

Get dictionary with counts of length `r` subsequences appearing in `S`.

# Arguments

Where `T <: Union{Int,String}` we have 

- `S::InteractionSequence{T}`: an interaction sequence 
- `r::Int`: length of subsequences to count 
- `drop_zero::Bool=true`: whether to drop subsequences with zero count
- `vertex_set::Vector{T}=unique(vcat(S...))`: underlying vertex set, will determine the possible subsequences. Defaults to unique values seen in `S`.
"""
function get_subseqs(
    S::InteractionSequence{T}, 
    r::Int;
    drop_zero::Bool=true, 
    vertex_set::Vector{T}=unique(vcat(S...))
    ) where {T<:Union{Int,String}}

    out = Dict{Vector{T},Int}()

    # If drop_zero=false, we initialise all possible subsequences with zero count
    if !drop_zero 
        # Use Base.Iterators.product - returns a tuple in each iteration 
        for path_tuple in Base.Iterators.product([vertex_set for i in 1:r]...)
            out[collect(path_tuple)] = 0
        end 
    end 
    get_subseqs!(out, S, r)
    return out
end  



""" 
    get_subseq_counts(data, r; vertex_set) 

Given sample of interaction sequences, obtain subseq counts in each and return these in a dictionary.

The returned dictionary maps paths to vectors of counts, with the latter having length equal to the number of interaction sequences in `data`. 

# Arguments 

With `T <: Union{Int,String}` we have

- `data::InteractionSequenceSample{T}`: sample of interaction sequences 
- `r::Int`: length of subsequences to count 
- `vertex_set::Vector{T}=unique(vcat([vcat(S_i...) for S_i in data]...))`: underlying vertex set, will determine the keys of output dict. Defaults to unique values seen in `data`.
"""
function get_subseq_counts(
    data::InterSeqSample{T}, 
    r::Int;
    vertex_set::Vector{T}=unique(vcat([vcat(S_i...) for S_i in data]...))
    ) where {T<:Union{Int,String}}

    # For output 
    out = Dict{Vector{T},Vector{Int}}()
    # For storing counts in each datapoint 
    out_i = Dict{Vector{T},Int}()

    # Use Base.Iterators.product - returns a tuple in each iteration 
    for path_tuple in Base.Iterators.product([vertex_set for i in 1:r]...)
        out_i[collect(path_tuple)] = 0
        out[collect(path_tuple)] = Int[]
    end 

    for S_i in data
        
        # Count subpaths in S_i
        get_subseqs!(out_i, S_i, r)
        
        # Augment out_vec with these 
        for (path, count) in out_i
            push!(out[path], count)
        end 

        # Set all values to zero 
        map!(x->0, values(out_i))

    end 

    return out

end 


"""
    get_unique_subseqs(S::InteractionSequence{T}, r::Int) 

Get the `Set` of unique length `r` subsequences appearing in `S`.
"""
function get_unique_subseqs(S::InteractionSequence{T}, r::Int) where {T<:Union{Int,String}}
    # Get subseq counts (excl. zero count subseqs)
    dict_subseq_counts = get_subseqs(S, r)
    
    # Keys of this dict will be uniq subseqs appearing in S 
    return Set(keys(dict_subseq_counts))
end     


"""
    get_unique_subseq_counts(data, r; vertex_set)

Given a sample of interaction sequences, return a dictionary detailing, for a given subsequence, the number of interaction sequences it appears in at least once.

# Arguments

Where `T<:Union{Int,String}` we have 

- `data::InteractionSequenceSample{T}`: sample of interaction sequences 
- `r::Int`: length of subsequences to count 
- `vertex_set::Vector{T}=unique(vcat([vcat(S_i...) for S_i in data]...))`: underlying vertex set, will determine the keys of output dict. Defaults to unique values seen in `data`.
"""
function get_unique_subseq_counts(
    data::InteractionSequenceSample{T}, 
    r::Int; 
    vertex_set::Vector{T}=unique(vcat([vcat(S_i...) for S_i in data]...))
    ) where {T<:Union{Int,String}}
    
    # Get subseq counts for each interaction sequence in data 
    dict_subseq_counts = get_subseq_counts(data, r, vertex_set=vertex_set)

    # Now we loop over paths and obtain number of networks wherein it appears 
    # at least once...
    out = Dict{Vector{T}, Int}()  # For output 
    for (path, counts_vec) in dict_subseq_counts
        out[path] = sum(counts_vec .> 0) 
    end 

    return out
end 


"""
    get_unique_subseq_proportions(data, r; vertex_set)

Given a sample of interaction sequences, return a dictionary detailing, for a given subsequence, the proportion of interaction sequences it appears in at least once. 

# Arguments

Where `T<:Union{Int,String}` we have 

- `data::InteractionSequenceSample{T}`: sample of interaction sequences 
- `r::Int`: length of subsequences to count 
- `vertex_set::Vector{T}=unique(vcat([vcat(S_i...) for S_i in data]...))`: underlying vertex set, will determine the keys of output dict. Defaults to unique values seen in `data`.
"""
function get_unique_subseq_proportions(
    data::InteractionSequenceSample{T}, 
    r::Int; 
    vertex_set::Vector{T}=unique(vcat([vcat(S_i...) for S_i in data]...))
    ) where {T<:Union{Int,String}}

    # Get subseq counts for each interaction sequence in data 
    dict_subseq_counts = get_subseq_counts(data, r, vertex_set=vertex_set)

    # Sample size (number of networks) 
    n = length(data)

    # Now we loop over paths and obtain proportion of networks wherein it appears 
    # at least once...
    out = Dict{Vector{T}, Float64}()  # For output 
    for (path, counts_vec) in dict_subseq_counts
        out[path] = sum(counts_vec .> 0) / n
    end 

    return out
end 


"""
    subseq_isin(S::InteractionSequence{T}, x::Vector{T}) where {T<:Union{Int,String}}

Return boolean resulting from testing if subsequence `x` appears in interaction sequence `S`.
"""
function subseq_isin(S::InteractionSequence{T}, x::Vector{T}) where {T<:Union{Int,String}}
    r = length(x)
    for path in S 
        n = length(path)
        for ind in subsets(1:n, r)
            if x == path[ind]
                return true 
            end 
        end 
    end 
    return false
end 


"""
    subseq_isin(data::InteractionSequenceSample{T}, x::Vector{T}) where {T<:Union{Int,String}}

Return boolean resulting from testing if subsequence `x` appears in sample of interaction sequences `data`.
"""
function subseq_isin(data::InteractionSequenceSample{T}, x::Vector{T}) where {T<:Union{Int,String}}
    out = Vector{Bool}()
    for S in data
        push!(out, subseq_isin(S, x))
    end 
    return out
end 

# Subpaths 
# --------

"""
    get_subpaths!(out::Dict{Vector{T},Int}, S::InteractionSequence{T}, r::Int) 

Augment dictionary `out` with counts of length `r` subpaths appearing in `S`.
"""
function get_subpaths!(
    out::Dict{Vector{T},Int},
    S::InteractionSequence{T}, 
    r::Int    
    ) where {T<:Union{Int,String}}
    for path in S 
        ind = 1:r  # Index for path (will be like a sliding window)
        n = length(path)
        for i in 1:(n-r+1)
            out[path[ind]] = get(out, path[ind], 0) + 1
            ind = ind .+ 1 # Slide indices of path
        end 
    end 
end 

"""
    get_subpaths(S, r; drop_zero, vertex_set) 

Get dictionary with counts of length `r` subpaths appearing in `S`.

# Arguments

Where `T<:Union{Int,String}` we have 

- `S::InteractionSequence{T}`: an interaction sequence 
- `r::Int`: length of subpaths to count
- `drop_zero::Bool=true`: whether to drop subpaths with zero count
- `vertex_set::Vector{T}=unique(vcat(S...))`: underlying vertex set, will determine the possible subsequences. Defaults to unique values seen in `S`.
"""
function get_subpaths(
    S::InteractionSequence{T}, 
    r::Int;
    drop_zero::Bool=true, 
    vertex_set::Vector{T}=unique(vcat(S...))
    ) where {T<:Union{Int,String}}

    out = Dict{Vector{T},Int}()

    # If drop_zero=false, we initialise all possible subsequences with zero count
    if !drop_zero 
        # Use Base.Iterators.product - returns a tuple in each iteration 
        for path_tuple in Base.Iterators.product([vertex_set for i in 1:r]...)
            out[collect(path_tuple)] = 0
        end 
    end 
    get_subpaths!(out, S, r)
    return out
end  

""" 
get_subseq_counts(data, r; vertex_set) 

Given sample of interaction sequences, obtain subpath counts in each and return these in a dictionary.

The returned dictionary maps paths to vectors of counts, with the latter having length equal to the number of interaction sequences in `data`. 

# Arguments 

With `T <: Union{Int,String}` we have

- `data::InteractionSequenceSample{T}`: sample of interaction sequences 
- `r::Int`: length of subpaths to count 
- `vertex_set::Vector{T}=unique(vcat([vcat(S_i...) for S_i in data]...))`: underlying vertex set, will determine the keys of output dict. Defaults to unique values seen in `data`.
"""
function get_subpath_counts(
    data::InterSeqSample{T}, 
    r::Int;
    vertex_set::Vector{T}=unique(vcat([vcat(S_i...) for S_i in data]...))
    ) where {T<:Union{Int,String}}

    # For output 
    out = Dict{Vector{T},Vector{Int}}()
    # For storing counts in each datapoint 
    out_i = Dict{Vector{T},Int}()

    # Use Base.Iterators.product - returns a tuple in each iteration 
    for path_tuple in Base.Iterators.product([vertex_set for i in 1:r]...)
        out_i[collect(path_tuple)] = 0
        out[collect(path_tuple)] = Int[]
    end 

    for S_i in data
        
        # Count subpaths in S_i
        get_subpaths!(out_i, S_i, r)
        
        # Augment out_vec with these 
        for (path, count) in out_i
            push!(out[path], count)
        end 

        # Set all values to zero 
        map!(x->0, values(out_i))

    end 

    return out

end 

"""
    get_unique_subpaths(S::InteractionSequence{T}, r::Int) 

Get the `Set` of unique length `r` subpaths appearing in `S`.
"""
function get_unique_subpaths(
    S::InteractionSequence{T}, 
    r::Int
    ) where {T<:Union{Int,String}}

    # Get subpath counts (excl. zero counts)
    dict_subpath_counts = get_subpaths(S, r)
    
    # Keys of this dict will be unique subseqs appearing in S 
    return Set(keys(dict_subpath_counts))
end


"""
    get_unique_subpath_counts(data, r; vertex_set)

Given a sample of interaction sequences, return a dictionary detailing, for a given subpath, the number of interaction sequences it appears in at least once.

# Arguments

Where `T<:Union{Int,String}` we have 

- `data::InteractionSequenceSample{T}`: sample of interaction sequences 
- `r::Int`: length of supaths to count 
- `vertex_set::Vector{T}=unique(vcat([vcat(S_i...) for S_i in data]...))`: underlying vertex set, will determine the keys of output dict. Defaults to unique values seen in `data`.
"""
function get_unique_supath_counts(
    data::InteractionSequenceSample{T}, 
    r::Int; 
    vertex_set::Vector{T}=unique(vcat([vcat(S_i...) for S_i in data]...))
    ) where {T<:Union{Int,String}}
    
    # Get subpath counts for each interaction sequence in data 
    dict_subpath_counts = get_subpath_counts(data, r, vertex_set=vertex_set)

    # Now we loop over paths and obtain number of networks wherein it appears 
    # at least once...
    out = Dict{Vector{T}, Int}()  # For output 
    for (path, counts_vec) in dict_subpath_counts
        out[path] = sum(counts_vec .> 0) 
    end 

    return out
end 


"""
    get_unique_subpath_proportions(data, r; vertex_set)

Given a sample of interaction sequences, return a dictionary detailing, for a given subpath, the proportion of interaction sequences it appears in at least once. 

# Arguments

Where `T<:Union{Int,String}` we have 

- `data::InteractionSequenceSample{T}`: sample of interaction sequences 
- `r::Int`: length of subpaths to count 
- `vertex_set::Vector{T}=unique(vcat([vcat(S_i...) for S_i in data]...))`: underlying vertex set, will determine the keys of output dict. Defaults to unique values seen in `data`.
"""
function get_unique_subpath_proportions(
    data::InteractionSequenceSample{T}, 
    r::Int; 
    vertex_set::Vector{T}=unique(vcat([vcat(S_i...) for S_i in data]...))
    ) where {T<:Union{Int,String}}

    # Get subpath counts for each interaction sequence in data 
    dict_subpath_counts = get_subpath_counts(data, r, vertex_set=vertex_set)

    # Sample size (number of networks) 
    n = length(data)

    # Now we loop over paths and obtain proportion of networks wherein it appears 
    # at least once...
    out = Dict{Vector{T}, Float64}()  # For output 
    for (path, counts_vec) in dict_subpath_counts
        out[path] = sum(counts_vec .> 0) / n
    end 

    return out
end 

"""
    subpath_isin(S::InteractionSequence{T}, x::Vector{T}) where {T<:Union{Int,String}}

Return `true` if supath `x` appears in interaction sequence `S`.
"""
function subpath_isin(S::InteractionSequence{T}, x::Vector{T}) where {T<:Union{Int,String}}
    r = length(x)
    for path in S 
        n = length(path)
        ind = 1:r 
        for i in 1:(n-r+1)
            if path[ind] == x
                return true
            end 
            ind = ind .+ 1
        end 
    end 
    return false
end 

"""
    subpath_isin(S::InteractionSequence{T}, x::Vector{T}) where {T<:Union{Int,String}}

Return `true` if subpath `x` appears in sample of interaction sequences `data`.
"""
function subpath_isin(data::InteractionSequenceSample{T}, x::Vector{T}) where {T<:Union{Int,String}}
    out = Vector{Bool}()
    for S in data
        push!(out, subpath_isin(S, x))
    end 
    return out
end 
