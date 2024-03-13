"""
Functions to get summarise the distribution of subsequences appearing in interaction 
networks or samples of interaction networks, e.g. to get the average number of times 
this subsequence appears in an observation, or the proportion of interaction networks in which 
a given subsequence appears (at least once).
"""

using IterTools

export get_subseqs!, get_subseqs, get_subpaths!, get_subpaths, get_subpath_counts, get_subseq_counts
export get_unique_subseqs, get_unique_subseq_counts, get_unique_subseq_proportions
export get_unique_subpaths!, get_unique_subpaths, get_unique_subpath_counts!, get_unique_subpath_counts!, get_unique_subpath_proportions
export subseq_isin, subpath_isin
export dict_rank

# TODO - make functions to get the count of subsequences in an interaction network, and then in a sample of networks
#      - Try and make this such that each subsequence will have a vector of counts, that can then be averaged to get 
#        the average count of this path in observations 
#      - Should actually be able to recover the old function by doing a count>0 on the vectors 
#      - Could possibly pass in a collection of subsequences to consider, or the vertex set so we can infer all possible 
#        subsequences and then have these make up the keys of a dictionary. This will mean some subsequences will have 
#        zero count. Could then have an option to drop zero count subsequences
#      - Do the same for subpaths  

"""
    get_subseqs!(out::Dict{Vector{T},Int}, S::InteractionSequence{T}, r::Int) 

Augment dictionary `out` with counts of length `r` subsequences appearing in `S`.
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
    get_subseqs(S::InteractionSequence{T}, r::Int; kwargs...) 

Get dict with counts of length `r` subsequences appearing in `S`.

This has following key-word arguments:

* `drop_zero::Bool` (default: `true`) = whether to drop subsequences with zero count
* `vertex_set::Vector{T} where {T<:Union{Int,String}}` (default: unique values in `S`) = underlying vertex set. Will determine the possible subsequences.
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
    get_subseq_counts(
        data::InterSeqSample{T}, r::Int; 
        vertex_set::Vector{T}=unique(vcat([vcat(S_i...) for S_i in data]...))
        ) where {T<:Union{Int,String}}

Given sample of interaction networks, obtains subseq counts in each. Returns dict mapping paths to vectors of counts, with the latter having length equal to the number of interaction networks. 

Keyword arguments:

* `vertex_set` (default = all vertices seen `data`) = underlying vertex set, will determine the keys of output dict.
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

Get the set of unique length `r` subsequences appearing in `S` (of type `Set`)
"""
function get_unique_subseqs(S::InteractionSequence{T}, r::Int) where {T<:Union{Int,String}}
    # Get subseq counts (excl. zero count subseqs)
    dict_subseq_counts = get_subseqs(S, r)
    
    # Keys of this dict will be uniq subseqs appearing in S 
    return Set(keys(dict_subseq_counts))
end     


"""
    get_unique_subseq_counts(
        data::InteractionSequenceSample{T}, 
        r::Int; 
        vertex_set::Vector{T}=unique(vcat([vcat(S_i...) for S_i in data]...))
        ) where {T<:Union{Int,String}}

Given a sample of interaction networks returns a dict detailing, for a given path, the number of interaction networks it appears in at least once. 
"""
function get_unique_subseq_counts(
    data::InteractionSequenceSample{T}, 
    r::Int; 
    vertex_set::Vector{T}=unique(vcat([vcat(S_i...) for S_i in data]...))
    ) where {T<:Union{Int,String}}
    
    # Get subseq counts for each interaction network in data 
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
    get_unique_subseq_proportions(
        data::InteractionSequenceSample{T}, 
        r::Int; 
        vertex_set::Vector{T}=unique(vcat([vcat(S_i...) for S_i in data]...))
        ) where {T<:Union{Int,String}}

Given a sample of interaction networks returns a dict detailing, for a given path, the proportion of interaction networks it appears in at least once. 
"""
function get_unique_subseq_proportions(
    data::InteractionSequenceSample{T}, 
    r::Int; 
    vertex_set::Vector{T}=unique(vcat([vcat(S_i...) for S_i in data]...))
    ) where {T<:Union{Int,String}}

    # Get subseq counts for each interaction network in data 
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

Test if subsequence appears in interaction network `S`. Returns `Bool`.
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
    subseq_isin(S::InteractionSequence{T}, x::Vector{T}) where {T<:Union{Int,String}}

Test if subsequence appears in sample of interaction networks. Returns vector of `Bool`.
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
    get_subpaths(S::InteractionSequence{T}, r::Int; kwargs...) 

Get dict with counts of length `r` subpaths appearing in `S`.

This has following key-word arguments:

* `drop_zero::Bool` (default: `true`) = whether to drop subpaths with zero count
* `vertex_set::Vector{T} where {T<:Union{Int,String}}` (default: unique values in `S`) = underlying vertex set. Will determine the possible subpaths.
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
    get_subpath_counts(
        data::InterSeqSample{T}, r::Int; 
        vertex_set::Vector{T}=unique(vcat([vcat(S_i...) for S_i in data]...))
        ) where {T<:Union{Int,String}}

Given sample of interaction networks, obtains subpath counts in each. Returns dict mapping paths to vectors of counts, with the latter having length equal to the number of interaction networks. 

Keyword arguments:

* `vertex_set` (default = all vertices seen `data`) = underlying vertex set, will determine the keys of output dict.
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

function get_unique_subpaths!(
    out::Set{Vector{T}}, 
    S::InteractionSequence{T}, 
    r::Int
    ) where {T<:Union{Int,String}}
    for path in S 
        ind = 1:r 
        n = length(path)
        for i in 1:(n-r+1)
            push!(out, path[ind])
            ind = ind .+ 1
        end 
    end 
end 

function get_unique_subpaths(
    S::InteractionSequence{T}, 
    r::Int
    ) where {T<:Union{Int,String}}
    out = Set{Vector{T}}()
    get_unique_subpaths!(out, S, r)
    return out
end



function get_unique_subpath_counts!(
    out::Dict{Vector{T},S}, 
    data::InteractionSequenceSample{T},
    r::Int
    ) where {T<:Union{Int,String},S<:Real}

    uniq_paths = Set{Vector{T}}()
    for x in data 
        empty!(uniq_paths)
        get_unique_subpaths!(uniq_paths, x, r)
        for path in uniq_paths
            out[path] = get(out, path, 0) + 1
        end 
    end 
end 

function get_unique_subpath_counts(data::InteractionSequenceSample{T}, r::Int) where {T<:Union{Int,String}}
    out = Dict{Vector{T},Int}()
    get_unique_subpath_counts!(out, data, r)
    return out
end 

function get_unique_subpath_proportions(data::InteractionSequenceSample{T}, r::Int) where {T<:Union{Int,String}}
    out = Dict{Vector{T},Float64}()
    n = length(data)
    get_unique_subpath_counts!(out, data, r)
    for key in keys(out)
        out[key] /= n 
    end 
    return out
end 

"""
    subpath_isin(S::InteractionSequence{T}, x::Vector{T}) where {T<:Union{Int,String}}

Test if subpath appears in interaction network `S`. Returns `Bool`.
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

Test if subpath appears in sample of interaction networks. Returns vector of `Bool`.
"""
function subpath_isin(data::InteractionSequenceSample{T}, x::Vector{T}) where {T<:Union{Int,String}}
    out = Vector{Bool}()
    for S in data
        push!(out, subpath_isin(S, x))
    end 
    return out
end 
