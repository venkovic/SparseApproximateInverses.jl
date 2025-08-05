push!(LOAD_PATH, ".")
using MySPAI: mr, sd, cg, cr, pmr, psd, pcg, pcr, lopmr, lopcr

using Random: seed!
using SparseArrays
using LinearAlgebra
using Arpack
using MatrixMarket: mmread, mmwrite
using NPZ

matrix_source = "../matrix-market/"

seed!(1)

matrix = "bcsstk15"

if matrix == "bcsstk21" # structural mechanics, Harwell-Boeing
  A = mmread(matrix_source * "bcsstk21.mtx") # SPD
  n = A.n # n = 3,600 | nnz = 26,600
elseif matrix == "Poisson4k" # FEM of random Poisson equation, TUM
  A = mmread(matrix_source * "Poisson_SExp_sig21.0_L0.1_DoF4000_K.mtx") # SPD
  n = A.n # n = 3,922 | nnz = 26,942
elseif matrix == "bcsstk15" # structural mechanics, Harwell-Boeing
  A = mmread(matrix_source * "bcsstk15.mtx") # SPD
  n = A.n # n = 3,948 | nnz = 117,816
end

function spectral_condition_number(A::SparseMatrixCSC)
  λmax = real(eigs(A, nev=1, which=:LM)[1][1])
  λmin = real(eigs(A, nev=1, which=:LM, sigma=0.0)[1][1])
  return λmax / λmin
end

tol = 1e-15
itmax = 1_000

#n = 4_000
#k = 100
#A = spdiagm(0=>vcat(rand(k), ones(n-k)), -1=>rand(n-1))
#A = A * A'
#println("cond(A) = $(spectral_condition_number(A))")
#n = 4_000; k = 100
#A = spdiagm(0=>vcat(rand(k), ones(n-k)), -1=>rand(n-1)); A = A * A'
#println("cond(A) = $(spectral_condition_number(A))")

Pr = spdiagm(diag(A).^-1)
M0 = spdiagm(ones(n))

#dt_mr = @elapsed, R_norm = mr(A, copy(M0), itmax, tol)
#npzwrite("data/Experience01_" * matrix * "_R_norm_mr.npz", R_norm)
#mmwrite("data/Experience01_" * matrix * "_M_mr.mtx", M)

#dt_sd = @elapsed M, R_norm = mr(A, copy(M0), itmax, tol)
#npzwrite("data/Experience01_" * matrix * "_R_norm_sd.npz", R_norm)
#mmwrite("data/Experience01_" * matrix * "_M_sd.mtx", M)

#dt_cg = @elapsed M, R_norm = cg(A, copy(M0), itmax, tol)
#npzwrite("data/Experience01_" * matrix * "_R_norm_cg.npz", R_norm)
#mmwrite("data/Experience01_" * matrix * "_M_cg.mtx", M)

#dt_cr = @elapsed M, R_norm = csd(A, copy(M0), itmax, tol)
#npzwrite("data/Experience01_" * matrix * "_R_norm_cr.npz", R_norm)
#mmwrite("data/Experience01_" * matrix * "_M_cr.mtx", M)


dt_pmr = @elapsed M, R_norm = pmr(A, Pr, copy(M0), itmax, tol)
npzwrite("data/Experience01_" * matrix * "_R_norm_pmr.npz", R_norm)
mmwrite("data/Experience01_" * matrix * "_M_pmr.mtx", M)

dt_psd = @elapsed M, R_norm = psd(A, Pr, copy(M0), itmax, tol)
npzwrite("data/Experience01_" * matrix * "_R_norm_psd.npz", R_norm)
mmwrite("data/Experience01_" * matrix * "_M_psd.mtx", M)

dt_pcg = @elapsed M, R_norm = pcg(A, Pr, copy(M0), itmax, tol)
npzwrite("data/Experience01_" * matrix * "_R_norm_pcg.npz", R_norm)
mmwrite("data/Experience01_" * matrix * "_M_pcg.mtx", M)

dt_pcr = @elapsed M, R_norm = pcr(A, Pr, copy(M0), itmax, tol)
npzwrite("data/Experience01_" * matrix * "_R_norm_pcr.npz", R_norm)
mmwrite("data/Experience01_" * matrix * "_M_pcr.mtx", M)

dt_lopmr = @elapsed M, R_norm = lopmr(A, Pr, copy(M0), itmax, tol)
npzwrite("data/Experience01_" * matrix * "_R_norm_lopmr.npz", R_norm)
mmwrite("data/Experience01_" * matrix * "_M_lopmr.mtx", M)

# LOPCR is equivalent to PCR, just more expensive to compute.
#dt_lopcr = @elapsed M, R_norm = lopcr(A, Pr, copy(M0), itmax, tol)
#npzwrite("data/Experience01_" * matrix * "_R_norm_lopcr.npz", R_norm)
#mmwrite("data/Experience01_" * matrix * "_M_lopcr.mtx", M)