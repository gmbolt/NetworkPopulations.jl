
"""
    dict_rank(d::Dict{T, S}) where {T<:Any, S<:Real}

Given dict with keys mapping to real values, returns a vector of (key, value) pairs sorted according to their 
values (descending).
"""
function dict_rank(d::Dict{T, S}) where {T<:Any, S<:Real}
    pairs = collect(d) # Makes vector of pairs
    vals = collect(values(d))
    ind = sortperm(vals, rev=true)
    return pairs[ind]
end 