#push!(LOAD_PATH, ".")
using Pkg
Pkg.activate(".")
using SparseApproximateInverses: frob_norm, frob_inner_prod

using Random: seed!
using SparseArrays
using LinearAlgebra
using Arpack
using MatrixMarket: mmread, mmwrite
using NPZ


matrix_source = "data/mtcs/spd/"


matrices = ("bundle1", 
            "4bw100eigs20k", 
            "rand20k", 
            "rand20k2", 
            "wathen100", 
            "Poisson32k")

function get_condF_upper_bound(A)
  # Wolkowicz and Styan (1980),
  # Bounds for eigenvalues using traces, 
  # Linear Algebra and its Applications.
  n = A.n
  vals = [n, 0., 0.]
  tr_A = tr(A)
  norm_A = frob_norm(A)
  val[2] = sqrt(n) / (tr_A / norm_A / sqrt(n))^2
  p = sqrt(n - 1.)
  m = tr_A / n
  s = sqrt(norm_A^2/n - (tr_A/n)^2)
  val[3] = 1 + 2 * s / (m - s / p)
  return maximum(vals)
end

function get_rho(M)
  n = M.n
  tr_M = tr(M)
  norm_M = frob_norm(M)
  rho = tr_M / n
  rho -= sqrt(norm_M^2/n - tr_M^2/n^2) * sqrt(n - 1)
  return rho
end

function check_AM_matvec(A, M)
  AM = A * M
  n = A.n
  av_dev = 0.
  k = 5
  for _ in 1:k
    x = rand(n)
    y = AM * x
    av_dev += norm(x - y) / norm(x)
  end
  return av_dev / k
end

function check_diagonal_non_dominance(AM)
  abs_diag = Vector(abs.(diag(AM)))
  AM_diag0 = copy(AM)
  AM_diag0[diagind(AM_diag0)] .= 0.
  max_abs_offdiag = maximum(abs.(AM_diag0), dims=2)
  return maximum(abs_diag ./ max_abs_offdiag)
end

for matrix in matrices
  println(matrix)
  if matrix == "Poisson32k"
    A = mmread(matrix_source * "Poisson_SExp_sig21.0_L0.1_DoF32000_K.mtx")
  else 
    A = mmread(matrix_source * matrix * ".mtx")
  end
  n = A.n
  A_norm = frob_norm(A)

  R_norm = npzread("data/Experiment05_" * matrix * "_R_norm_pmr_spai.npz")[end]
  M = mmread("data/Experiment05_" * matrix * "_M_pmr_spai.mtx")
  M_norm = frob_norm(M)
  println("backward-error of PMR = ", R_norm / (A_norm * M_norm + sqrt(n)))
  println("cos(A, M) = ", frob_inner_prod(A, M) / (A_norm * M_norm))
  println(get_rho(M))
  println(minimum(diag(M)))
  AM = A * M
  println(minimum(diag(AM)))
  println(tr(AM) / (frob_norm(AM) * sqrt(n)))
  println(check_AM_matvec(A, M))
  println(maximum(diag(I-AM)))
  AM[diagind(AM)] .= 0.
  println(maximum(abs.(AM)))
  #println(check_diagonal_non_dominance(AM))

  R_norm = npzread("data/Experiment05_" * matrix * "_R_norm_pcr_spai.npz")[end]
  M = mmread("data/Experiment05_" * matrix * "_M_pcr_spai.mtx")
  M_norm = frob_norm(M)
  println("backward-error of PCR = ", R_norm / (A_norm * M_norm + sqrt(n)))
  println("cos(A, M) = ", frob_inner_prod(A, M) / (A_norm * M_norm))
  println(get_rho(M))
  println(minimum(diag(M)))
  AM = A * M
  println(minimum(diag(AM)))
  println(tr(AM) / (frob_norm(AM) * sqrt(n)))
  println(check_AM_matvec(A, M))
  println(maximum(diag(I-AM)))
  AM[diagind(AM)] .= 0.
  println(maximum(abs.(AM)))
  #println(check_diagonal_non_dominance(AM))

  R_norm = npzread("data/Experiment05_" * matrix * "_R_norm_lopmr_spai.npz")[end]
  M = mmread("data/Experiment05_" * matrix * "_M_lopmr_spai.mtx")
  M_norm = frob_norm(M)
  println("backward-error of LOPMR = ", R_norm / (A_norm * M_norm + sqrt(n)))
  println("cos(A, M) = ", frob_inner_prod(A, M) / (A_norm * M_norm))
  println(get_rho(M))
  println(minimum(diag(M)))
  AM = A * M
  println(minimum(diag(AM)))
  println(tr(AM) / (frob_norm(AM) * sqrt(n)))
  println(check_AM_matvec(A, M))
  println(maximum(diag(I-AM)))
  AM[diagind(AM)] .= 0.
  println(maximum(abs.(AM)))
  #println(check_diagonal_non_dominance(AM))
end