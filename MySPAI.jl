module MySPAI

using LinearAlgebra
using SparseArrays
using LinearMaps: FunctionMap

export frob_inner_prod, frob_norm, frob_norm_squared, res_frob_norm
export mr, sd, cg, cr
export pmr, psd, pcg, pcr, lopmr, lopcr
export pmr_spai, pcr_spai, lopmr_spai
export lopmr_spai_split # pcr_spai_split
export pcg

include("frob-inner-prod.jl")
include("global-descent-algorithms.jl")
include("dropping-strategies.jl")
include("pcg.jl")

end