#push!(LOAD_PATH, ".")
using Pkg
Pkg.activate(".")
using SparseApproximateInverses: pmr_spd_spai, pcg_spai, lopmr_spd_spai

using Random: seed!
using SparseArrays
using LinearAlgebra
using Arpack
using MatrixMarket: mmread, mmwrite
using NPZ


matrix_source = "data/mtcs/spd/"


matrix = "bundle1" # \in {"bundle1", 
                   #      "4bw100eigs20k", 
                   #      "4bw100eigs20k2", 
                   #      "rand20k",
                   #      "rand20k2",
                   #      "wathen100", 
                   #      "Poisson32k"}


if matrix == "bundle1" # Computer graphics problem
  A = mmread(matrix_source * "bundle1.mtx")
  n = A.n # n = 10,581 | nnz = 770,811
elseif matrix == "4bw100eigs20k" # Custom matrix
  A = mmread(matrix_source * "4bw100eigs20k.mtx")
  n = A.n # n = 20,000 | nnz = 179,980
elseif matrix == "4bw100eigs20k2" # Custom matrix
  A = mmread(matrix_source * "4bw100eigs20k2.mtx")
  n = A.n # n = 20,000 | nnz = 179,980
elseif matrix == "rand20k"
  A = mmread(matrix_source * "rand20k.mtx")
  n = A.n # n = 20,000 | nnz = 99,772
elseif matrix == "rand20k2"
  A = mmread(matrix_source * "rand20k2.mtx")
  n = A.n # n = 20,000 | nnz = 99,772
elseif matrix == "wathen100" # Random 2D/3D problem
  A = mmread(matrix_source * "wathen100.mtx")
  n = A.n # n = 30,401 | nnz = 471,601
  # pmr and pcr make no progress after 100 iterations
elseif matrix == "Poisson32k" # Random Poisson PDE
  A = mmread(matrix_source * "Poisson32k.mtx")
  n = A.n # n = 31,839 | nnz = 221,375 end
end

itmax = [5, 20, 100]
tol= 1e-6 * sqrt(n)

s = .03

Pr = spdiagm(diag(A).^-1)
M0 = spdiagm(ones(n))


for it in itmax

  dt = @elapsed M, R_norm = pmr_spd_spai(A, Pr, copy(M0), it, tol, s)
  npzwrite("data/Experiment05_" * matrix * "_R_norm_pmr_spai_it$it.npz", R_norm)
  mmwrite("data/Experiment05_" * matrix * "_M_pmr_spai_it$it.mtx", M)
  val_LR = real(eigs(M, nev=1, which=:LR, tol=1e-3, maxiter=2_000)[1][1]); println("val_LR = $val_LR")
  val_SR = try
    real(eigs(M, nev=1, which=:SR, tol=1e-3, maxiter=3_000)[1][1])
  catch
    real(eigs(M, nev=1, sigma=-100.0, which=:LM, tol=1e-3, maxiter=3000)[1][1])
  end
  println("val_SR = $val_SR")
  npzwrite("data/Experiment05_" * matrix * "_metadata_pmr_spai_it$it.npz", 
         [val_LR, val_SR])

  dt = @elapsed M, R_norm = pcg_spai(A, Pr, copy(M0), it, tol, s)
  npzwrite("data/Experiment05_" * matrix * "_R_norm_pcg_spai_it$it.npz", R_norm)
  mmwrite("data/Experiment05_" * matrix * "_M_pcg_spai_it$it.mtx", M)
  val_LR = real(eigs(M, nev=1, which=:LR, tol=1e-3, maxiter=2_000)[1][1]); println("val_LR = $val_LR")
  val_SR = try
    real(eigs(M, nev=1, which=:SR, tol=1e-3, maxiter=3_000)[1][1])
  catch
    real(eigs(M, nev=1, sigma=-100.0, which=:LM, tol=1e-3, maxiter=3000)[1][1])
  end
  println("val_SR = $val_SR")
  npzwrite("data/Experiment05_" * matrix * "_metadata_pcg_spai_it$it.npz", 
           [val_LR, val_SR])

  dt = @elapsed M, R_norm = lopmr_spd_spai(A, Pr, copy(M0), it, tol, s)
  npzwrite("data/Experiment05_" * matrix * "_R_norm_lopmr_spai_it$it.npz", R_norm)
  mmwrite("data/Experiment05_" * matrix * "_M_lopmr_spai_it$it.mtx", M)
  val_LR = real(eigs(M, nev=1, which=:LR, tol=1e-3, maxiter=2_000)[1][1]); println("val_LR = $val_LR")
  val_SR = try
    real(eigs(M, nev=1, which=:SR, tol=1e-3, maxiter=3_000)[1][1])
  catch
    real(eigs(M, nev=1, sigma=-100.0, which=:LM, tol=1e-3, maxiter=3000)[1][1])
  end
  println("val_SR = $val_SR")
  npzwrite("data/Experiment05_" * matrix * "_metadata_lopmr_spai_it$it.npz", 
           [val_LR, val_SR])

end