#push!(LOAD_PATH, ".")
using Pkg
Pkg.activate(".")
using SparseApproximateInverses: pcg

using Random: seed!
using LinearAlgebra
using SparseArrays
using MatrixMarket: mmread, mmwrite
using NPZ


matrix_source = "data/mtcs/spd/"


matrix = "4bw100eigs20k2"


if matrix == "4bw100eigs20k2" # Custom matrix
  A = mmread(matrix_source * "4bw100eigs20k2.mtx")
  n = A.n # n = 20,000 | nnz = 179,980
end

tol = 1e-6
itmax = 2000
nreals = 20

M = spdiagm(diag(A).^-1)
M_pmr = mmread("data/Experiment03_" * matrix * "_M_pmr_spd.mtx")
M_pmr .+= M_pmr'; M_pmr .*= .5
M_pcg = mmread("data/Experiment03_" * matrix * "_M_pcg.mtx")
M_pcg .+= M_pcg'; M_pcg .*= .5
M_lopmr = mmread("data/Experiment03_" * matrix * "_M_lopmr_spd.mtx")
M_lopmr .+= M_lopmr'; M_lopmr .*= .5

seed!(1)
for ireal in 1:nreals
  b = rand(n)

  x, it, res_norm = pcg(A, b, zeros(n), M, tol, itmax)
  npzwrite("data/Experiment04_" * matrix * "_cg-res_real$ireal.npz", res_norm)

  x, it, res_norm_pmr = pcg(A, b, zeros(n), M_pmr, tol, itmax)
  npzwrite("data/Experiment04_" * matrix * "_pcg-res_pmr_real$ireal.npz", res_norm_pmr)

  x, it, res_norm_pcg = pcg(A, b, zeros(n), M_pcg, tol, itmax)
  npzwrite("data/Experiment04_" * matrix * "_pcg-res_pcg_real$ireal.npz", res_norm_pcg)

  x, it, res_norm_lopmr = pcg(A, b, zeros(n), M_lopmr, tol, itmax)
  npzwrite("data/Experiment04_" * matrix * "_pcg-res_lopmr_real$ireal.npz", res_norm_lopmr)
end