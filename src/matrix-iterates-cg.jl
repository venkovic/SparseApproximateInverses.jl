"""
cg(A, M, itmax, tol; stopping_criterion=:res, smax=1., ExplicitResidualUpdate=false)

Conjugate gradient method for approximate matrix inverses.

"""
function cg(A, M, itmax, tol; stopping_criterion=:res, smax=1., ExplicitResidualUpdate=false)
  n = A.n
  R = spzeros(n, n)
  P = spzeros(n, n)
  AP = spzeros(n, n)
  R_norm = zeros(itmax + 1)
  densities = zeros(itmax + 1)
  if stopping_criterion == :backward_error
    A_norm = frob_norm(A)
    backward_error = zeros(itmax + 1)
  end
  R .= I - A * M
  R_frob_norm_squared = frob_norm_squared(R)
  P .= R
  AP .= A * P
  i = 0
  R_norm[i + 1] = sqrt(R_frob_norm_squared)
  densities[i + 1] = nnz(M) / n^2
  if stopping_criterion == :backward_error
    M_norm = frob_norm(M)
    backward_error[i + 1] = R_norm[i + 1] / (A_norm * M_norm + sqrt(n))
  end
  while (i < itmax)
    P_AP_frob = frob_inner_prod(P, AP)
    alpha = R_frob_norm_squared / P_AP_frob
    beta = 1. / R_frob_norm_squared
    M .+= alpha .* P
    if ExplicitResidualUpdate
      R .= I - A * M
    else
      R .-= alpha .* AP
    end
    R_frob_norm_squared = frob_norm_squared(R)
    i += 1
    R_norm[i + 1] = sqrt(R_frob_norm_squared)
    densities[i + 1] = nnz(M) / n^2
    err = R_norm[i + 1]
    if stopping_criterion == :backward_error
      M_norm = frob_norm(M)
      backward_error[i + 1] = R_norm[i + 1] / (A_norm * M_norm + sqrt(n))
      err = backward_error[i + 1]
    end
    s = nnz(M) / n^2
    println("cg it = $i, err = $err, 
             nnz(M)/n^2 = $s, 
             nnz(P)/n^2 = $(nnz(P)/n^2)")
    if (err < tol) || (s > smax)
      break
    end
    beta *= R_frob_norm_squared
    P .*= beta
    P .+= R
    AP .= A * P
  end
  if stopping_criterion == :backward_error
    return M, R_norm[1:i+1], backward_error[1:i+1]
  end
  return M, R_norm[1:i+1], densities
end


"""
pcg(A, Pr, M, itmax, tol; stopping_criterion=:res, smax=1., ExplicitResidualUpdate=false)

Preconditioned conjugate gradient method for approximate matrix inverses with
symmetric positive definite (SPD) preconditioner.

"""
function pcg(A, Pr, M, itmax, tol; stopping_criterion=:res, smax=1., ExplicitResidualUpdate=false)
  n = A.n
  R = spzeros(n, n)
  P = spzeros(n, n)
  Z = spzeros(n, n)
  AP = spzeros(n, n)
  R_norm = zeros(itmax + 1)
  densities = zeros(itmax + 1)
  if stopping_criterion == :backward_error
    A_norm = frob_norm(A)
    backward_error = zeros(itmax + 1)
  end
  R .= I - A * M
  Z .= Pr * R
  R_Z_frob = frob_inner_prod(R, Z)
  P .= Z
  AP .= A * P
  i = 0
  R_norm[i + 1] = frob_norm(R)
  densities[i + 1] = nnz(M) / n^2
  if stopping_criterion == :backward_error
    M_norm = frob_norm(M)
    backward_error[i + 1] = R_norm[i + 1] / (A_norm * M_norm + sqrt(n))
  end
  while (i < itmax)
    P_AP_frob = frob_inner_prod(P, AP)
    alpha = R_Z_frob / P_AP_frob
    beta = 1. / R_Z_frob
    M .+= alpha .* P
    if ExplicitResidualUpdate
      R .-= alpha .* AP
    else
      R .= I - A * M
    end
    Z .= Pr * R
    R_Z_frob = frob_inner_prod(R, Z)
    i += 1
    R_norm[i + 1] = frob_norm(R)
    densities[i + 1] = nnz(M) / n^2
    err = R_norm[i + 1]
    if stopping_criterion == :backward_error
      M_norm = frob_norm(M)
      backward_error[i + 1] = R_norm[i + 1] / (A_norm * M_norm + sqrt(n))
      err = backward_error[i + 1]
    end
    s = nnz(M) / n^2
    println("pcg it = $i, err = $err, 
             nnz(M)/n^2 = $s, 
             nnz(P)/n^2 = $(nnz(P)/n^2)")
    if (err < tol) || (s > smax)
      break
    end
    beta *= R_Z_frob
    P .*= beta
    P .+= Z
    AP .= A * P
  end
  if stopping_criterion == :backward_error
    return M, R_norm[1:i+1], backward_error[1:i+1]
  end
  return M, R_norm[1:i+1], densities
end


"""
pcg_spai(A, Pr, M, itmax, tol, s; stopping_criterion=:res, eval_pcg=false)

Preconditioned conjugate gradient method for sparse approximate matrix 
inverses with SPD preconditioner.

"""
function pcg_spai(A, Pr, M, itmax, tol, s; stopping_criterion=:res, eval_pcg=false)
  n = A.n
  m = round(Int, s * n * n)
  R = spzeros(n, n)
  P = spzeros(n, n)
  Z = spzeros(n, n)
  AR = spzeros(n, n)
  AP = spzeros(n, n)
  W = AP
  A_cols_dot_prods = Diagonal(vec(sum(abs2, A, dims=1)))
  R_norm = zeros(itmax + 1)
  densities = zeros(itmax + 1)
  if stopping_criterion == :backward_error
    A_norm = frob_norm(A)
    backward_error = zeros(itmax + 1)
  end
  if eval_pcg
    seed!(1)
    pcg_tol = 1e-6
    b = A * rand(n)
    pcg_iters = zeros(Int, itmax + 1)
    _, pcg_iters[1], _ = pcg(A, b, zeros(n), M, 1e-6, n)
    println("pcg iters = $(pcg_iters[1])")
  end
  dropping_M!(M, W, R, AR, A_cols_dot_prods, A, m)
  Z .= Pr * R
  R_Z_frob = frob_inner_prod(R, Z)
  P .= Z
  dropping_P!(P, m)
  AP .= A * P
  i = 0
  R_norm[i + 1] = frob_norm(R)
  densities[i + 1] = nnz(M) / n^2
  if stopping_criterion == :backward_error
    M_norm = frob_norm(M)
    backward_error[i + 1] = R_norm[i + 1] / (A_norm * M_norm + sqrt(n))
  end
  while (i < itmax)
    P_AP_frob = frob_inner_prod(P, AP)
    alpha = R_Z_frob / P_AP_frob
    beta = 1. / R_Z_frob
    M .+= alpha .* P
    dropping_M!(M, W, R, AR, A_cols_dot_prods, A, m)
    Z .= Pr * R
    R_Z_frob = frob_inner_prod(R, Z)
    i += 1
    R_norm[i + 1] = frob_norm(R)
    densities[i + 1] = nnz(M) / n^2
    err = R_norm[i + 1]
    if eval_pcg
      _, pcg_iters[i+1], _ = pcg(A, b, zeros(n), M, pcg_tol, pcg_iters[1])
      println("pcg iters = $(pcg_iters[i+1])")
    end
    if stopping_criterion == :backward_error
      M_norm = frob_norm(M)
      backward_error[i + 1] = R_norm[i + 1] / (A_norm * M_norm + sqrt(n))
      err = backward_error[i + 1]
    end
    println("pcg-spai it = $i, err = $err, 
             nnz(M)/n^2 = $(nnz(M)/n^2), 
             nnz(P)/n^2 = $(nnz(P)/n^2)")
    if (err < tol)
      break
    end
    beta *= R_Z_frob
    P .*= beta
    P .+= Z
    dropping_P!(P, m)
    AP .= A * P
  end
  if stopping_criterion == :backward_error
    return M, R_norm[1:i+1], backward_error[1:i+1]
  end
  if eval_pcg
    return M, R_norm[1:i+1], pcg_iters[1:i+1]
  end
  return M, R_norm[1:i+1], densities
end


"""
lopcg(A, Pr, M, itmax, tol; stopping_criterion=:res, smax=1.)

Locally optimal preconditioned conjugate gradient method for approximate matrix inverses.

"""
function lopcg(A, Pr, M, itmax, tol; stopping_criterion=:res, smax=1.)
  n = A.n
  R = spzeros(n, n)
  Z = spzeros(n, n)
  P = spzeros(n, n)
  AZ = spzeros(n, n)
  AP = spzeros(n, n)
  R_norm = zeros(itmax + 1)
  densities = zeros(itmax + 1)
  if stopping_criterion == :backward_error
    A_norm = frob_norm(A)
    backward_error = zeros(itmax + 1)
  end
  R .= I - A * M
  Z .= Pr * R
  P = spzeros(n, n)
  i = 0
  R_norm[i + 1] = frob_norm(R)
  densities[i + 1] = nnz(M) / n^2
  if stopping_criterion == :backward_error
    M_norm = frob_norm(M)
    backward_error[i + 1] = R_norm[i + 1] / (A_norm * M_norm + sqrt(n))
  end
  while (i < itmax)
    AP .= A * P
    AZ .= A * Z
    if i == 0
      Z_AZ_frob = frob_inner_prod(Z, AZ)
      R_Z_frob = frob_inner_prod(R, Z)
      delta = R_Z_frob / Z_AZ_frob
      gamma = 1.
    else
      Z_AZ_frob = frob_inner_prod(Z, AZ)
      P_AP_frob = frob_inner_prod(P, AP)
      Z_AP_frob = frob_inner_prod(Z, AP) 
      R_Z_frob = frob_inner_prod(R, Z)
      R_P_frob = frob_inner_prod(R, P)
      c = Z_AZ_frob * P_AP_frob - Z_AP_frob ^ 2
      delta = (P_AP_frob * R_Z_frob - Z_AP_frob * R_P_frob) / c
      gamma = (Z_AZ_frob * R_P_frob - Z_AP_frob * R_Z_frob) / c
    end
    M .+= delta .* Z
    M .+= gamma .* P
    R .-= delta .* AZ
    R .-= gamma .* AP
    i += 1
    R_norm[i + 1] = frob_norm(R)
    densities[i + 1] = nnz(M) / n^2
    err = R_norm[i + 1]
    if stopping_criterion == :backward_error
      M_norm = frob_norm(M)
      backward_error[i + 1] = R_norm[i + 1] / (A_norm * M_norm + sqrt(n))
      err = backward_error[i + 1]
    end
    s = nnz(M) / n^2
    println("lopcg it = $i, err = $err, 
             nnz(M)/n^2 = $s, 
             nnz(P)/n^2 = $(nnz(P)/n^2)")
    if (err < tol) || (s > smax)
      break
    end
    P .*= gamma / delta
    P .+= Z
    Z .= Pr * R
  end
  if stopping_criterion == :backward_error
    return M, R_norm[1:i+1], backward_error[1:i+1]
  end
  return M, R_norm[1:i+1], densities
end