push!(LOAD_PATH, ".")
using MySPAI: pcg

using Random: seed!
using LinearAlgebra
using MatrixMarket: mmread, mmwrite
using NPZ


matrix_source = "../matrix-market/"


matrix = "4bw100eigs20k2"


if matrix == "4bw100eigs20k2" # Custom matrix
  A = mmread(matrix_source * "4bw100eigs20k2.mtx")
  n = A.n # n = 20,000 | nnz = 179,980
end


tol = 1e-6
itmax = 2_000

seed!(1)
b = A * rand(n)

x, it, res_norm = pcg(A, b, zeros(n), I, tol, itmax)
npzwrite("data/Experiment04_" * matrix * "_cg-res.npz", res_norm)

M = mmread("data/Experiment03_" * matrix * "_M_pmr.mtx")
x, it, res_norm_pmr = pcg(A, b, zeros(n), M, tol, itmax)
npzwrite("data/Experiment04_" * matrix * "_pcg-res_pmr.npz", res_norm_pmr)

M = mmread("data/Experiment03_" * matrix * "_M_pcg.mtx")
x, it, res_norm_pcg = pcg(A, b, zeros(n), M, tol, itmax)
npzwrite("data/Experiment04_" * matrix * "_pcg-res_pcg.npz", res_norm_pcg)

M = mmread("data/Experiment03_" * matrix * "_M_lopmr.mtx")
x, it, res_norm_lopmr = pcg(A, b, zeros(n), M, tol, itmax)
npzwrite("data/Experiment04_" * matrix * "_pcg-res_lopmr.npz", res_norm_lopmr)