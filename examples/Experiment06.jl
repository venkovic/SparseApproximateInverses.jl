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


matrix = "bundle1"        # \in {"bundle1", 
                          #      "4bw100eigs20k", 
                          #      "rand20k"
                          #      "rand20k2"
                          #      "wathen100", 
                          #      "Poisson32k"}


if matrix == "bundle1" # Computer graphics problem
  A = mmread(matrix_source * "bundle1.mtx")
  n = A.n # n = 10,581 | nnz = 770,811
  itmax = 170
elseif matrix == "4bw100eigs20k" # Custom matrix
  A = mmread(matrix_source * "4bw100eigs20k.mtx")
  n = A.n # n = 20,000 | nnz = 179,980
  itmax = 500
elseif matrix == "rand20k"
  A = mmread(matrix_source * "rand20k.mtx")
  n = A.n # n = 20,000 | nnz = 99,772
  itmax = 20
elseif matrix == "rand20k2"
  A = mmread(matrix_source * "rand20k2.mtx")
  n = A.n # n = 20,000 | nnz = 99,772
  itmax = 300
elseif matrix == "wathen100" # Random 2D/3D problem
  A = mmread(matrix_source * "wathen100.mtx")
  n = A.n # n = 30,401 | nnz = 471,601
  itmax = 100
elseif matrix == "Poisson32k" # Random Poisson PDE
  A = mmread(matrix_source * "Poisson32k.mtx")
  n = A.n # n = 31,839 | nnz = 221,375 
  itmax = 200
end


tol = 1e-6
nreals = 20

M = spdiagm(diag(A).^-1)
M_pmr = mmread("data/Experiment05_" * matrix * "_M_pmr_spai_it100.mtx")
M_pmr .+= M_pmr'; M_pmr .*= .5
M_pcg = mmread("data/Experiment05_" * matrix * "_M_pcg_spai_it100.mtx")
M_pcg .+= M_pcg'; M_pcg .*= .5
M_lopmr = mmread("data/Experiment05_" * matrix * "_M_lopmr_spai_it100.mtx")
M_lopmr .+= M_lopmr'; M_lopmr .*= .5


seed!(1)
for ireal in 1:nreals
  b = rand(n)

  x, it, res_norm = pcg(A, b, zeros(n), M, tol, itmax)
  npzwrite("data/Experiment06_" * matrix * "_cg-res_real$ireal.npz", res_norm)

  x, it, res_norm_pmr = pcg(A, b, zeros(n), M_pmr, tol, itmax)
  npzwrite("data/Experiment06_" * matrix * "_pcg-res_pmr_spai_real$ireal.npz", res_norm_pmr)

  x, it, res_norm_pcg = pcg(A, b, zeros(n), M_pcg, tol, itmax)
  npzwrite("data/Experiment06_" * matrix * "_pcg-res_pcg_spai_real$ireal.npz", res_norm_pcg)

  x, it, res_norm_lopmr = pcg(A, b, zeros(n), M_lopmr, tol, itmax)
  npzwrite("data/Experiment06_" * matrix * "_pcg-res_lopmr_spai_real$ireal.npz", res_norm_lopmr)
end