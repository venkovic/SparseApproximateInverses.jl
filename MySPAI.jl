module MySPAI

using LinearAlgebra
#using LinearAlgebra: mul!, axpy!, axpby!, dot, norm2
using SparseArrays
using LinearMaps: FunctionMap

export frob_inner_prod, frob_norm, frob_norm_squared, res_frob_norm
export mr, sd, cg, csd, pmr, lopmr, psd, pcg, pcr, lopcr
export pmr_spai, lopmr_spai, pcr_spai
export lopmr_spai_split
export pcg

include("frob-inner-prod.jl")
include("global-descent-algorithms.jl")
include("dropping-strategies.jl")
include("pcg.jl")

end