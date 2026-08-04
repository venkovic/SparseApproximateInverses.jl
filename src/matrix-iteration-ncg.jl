"""
ncg(A, M, itmax, tol; stopping_criterion=:res, smax=1., ExplicitResidualUpdate=false)

Nonlinear conjugate gradient method for approximate matrix inverses.

"""
function ncg(A, M, itmax, tol; stopping_criterion=:res, smax=1., ExplicitResidualUpdate=false)
  n = A.n
  R = spzeros(n, n)
  G = spzeros(n, n)
  P = spzeros(n, n)
  AP = spzeros(n, n)
  R_norm = zeros(itmax + 1)
  densities = zeros(itmax + 1)
  if stopping_criterion == :backward_error
    A_norm = frob_norm(A)
    backward_error = zeros(itmax + 1)
  end
  R .= I - A * M
  G .= - A * R
  R_G_frob = frob_inner_prod(R, G)
  P .= - G
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
    alpha = - R_G_frob / P_AP_frob
    beta = 1. / R_G_frob
    M .+= alpha .* P
    if ExplicitResidualUpdate
      R .= I - A * M
    else
      R .-= alpha .* AP
    end
    G .-= A * R
    R_G_frob = frob_inner_prod(R, G)
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
    println("ncg it = $i, err = $err, 
             nnz(M)/n^2 = $s,
             nnz(P)/n^2 = $(nnz(P)/n^2)")
    if (err < tol) || (s > smax)
      break
    end
    beta *= R_G_frob
    P .*= beta
    P .-= G
    AP .= A * P
  end
  if stopping_criterion == :backward_error
    return M, R_norm[1:i+1], backward_error[1:i+1]
  end
  return M, R_norm[1:i+1], densities
end


"""
npcg(A, Pr, M, itmax, tol; stopping_criterion=:res, smax=1., ExplicitResidualUpdate=false)

Nonlinear preconditioned conjugate gradient method for approximate matrix inverses.

"""
function npcg(A, Pr, M, itmax, tol; stopping_criterion=:res, smax=1., ExplicitResidualUpdate=false)
  n = A.n
  R = spzeros(n, n)
  Z = spzeros(n, n)
  G = spzeros(n, n)
  P = spzeros(n, n)
  AP = spzeros(n, n)
  R_norm = zeros(itmax + 1)
  densities = zeros(itmax + 1)
  if stopping_criterion == :backward_error
    A_norm = frob_norm(A)
    backward_error = zeros(itmax + 1)
  end
  R .= I - A * M
  Z .= Pr * R
  G .= - A * Z
  G .= Pr * G 
  R_G_frob = frob_inner_prod(R, G)
  P .= - G
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
    alpha = - R_G_frob / P_AP_frob
    beta = 1. / R_G_frob
    M .+= alpha .* P
    if ExplicitResidualUpdate
      R .= I - A * M
    else
      R .-= alpha .* AP
    end
    Z .= Pr * R
    G .= - A * Z
    G .= Pr * G
    R_G_frob = frob_inner_prod(R, G)
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
    println("npcg it = $i, err = $err, 
             nnz(M)/n^2 = $s,
             nnz(P)/n^2 = $(nnz(P)/n^2)")
    if (err < tol) || (s > smax)
      break
    end
    beta *= R_G_frob
    P .*= beta
    P .-= G
    AP .= A * P
  end
  if stopping_criterion == :backward_error
    return M, R_norm[1:i+1], backward_error[1:i+1]
  end
  return M, R_norm[1:i+1], densities
end