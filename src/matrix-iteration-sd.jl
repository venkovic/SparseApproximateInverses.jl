"""
sd(A, M, itmax, tol; stopping_criterion=:res, smax=1., ExplicitResidualUpdate=false)

Steepest descent method for approximate matrix inverses.

"""
function sd(A, M, itmax, tol; stopping_criterion=:res, smax=1., ExplicitResidualUpdate=false)
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
  P .= A * R
  AP .= A * P
  i = 0
  R_norm[i + 1] = frob_norm(R)
  densities[i + 1] = nnz(M) / n^2
  if stopping_criterion == :backward_error
    M_norm = frob_norm(M)
    backward_error[i + 1] = R_norm[i + 1] / (A_norm * M_norm + sqrt(n))
  end
  while (i < itmax)
    R_AP_frob = frob_inner_prod(R, AP)
    AP_norm_square = frob_norm_squared(AP)
    alpha = R_AP_frob / AP_norm_square
    M .+= alpha .* P
    if ExplicitResidualUpdate
      R .= I - A * M
    else
      R .-= alpha .* AP
    end
    P .= A * R
    AP .= A * P
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
    println("sd it = $i, err = $err, 
             nnz(M)/n^2 = $s,
             nnz(P)/n^2 = $(nnz(P)/n^2)")
    if (err < tol) || (s > smax)
      break
    end
  end
  if stopping_criterion == :backward_error
    return M, R_norm[1:i+1], backward_error[1:i+1]
  end
  return M, R_norm[1:i+1], densities
end


"""
psd_spd(A, Pr, M, itmax, tol; stopping_criterion=:res, smax=1., ExplicitResidualUpdate=false)

Right-preconditioned steepest descent method for approximate matrix inverses.

"""
function psd_spd(A, Pr, M, itmax, tol; stopping_criterion=:res, smax=1., ExplicitResidualUpdate=false)
  n = A.n
  R = spzeros(n, n)
  Z = spzeros(n, n)
  P = spzeros(n, n)
  AP = spzeros(n, n)
  PrAP = spzeros(n, n)
  R_norm = zeros(itmax + 1)
  densities = zeros(itmax + 1)
  if stopping_criterion == :backward_error
    A_norm = frob_norm(A)
    backward_error = zeros(itmax + 1)
  end
  R .= I - A * M
  Z .= Pr * R
  P .= A'R
  AP .= A * P
  PrAP .= Pr * AP
  i = 0
  R_norm[i + 1] = frob_norm(R)
  densities[i + 1] = nnz(M) / n^2
  if stopping_criterion == :backward_error
    M_norm = frob_norm(M)
    backward_error[i + 1] = R_norm[i + 1] / (A_norm * M_norm + sqrt(n))
  end
  while (i < itmax)
    Z_AP_frob = frob_inner_prod(Z, AP)
    AP_PrAP_frob = frob_inner_prod(AP, PrAP)
    alpha = Z_AP_frob / AP_PrAP_frob
    M .+= alpha .* P
    if ExplicitResidualUpdate
      R .= I - A * M
    else
      R .-= alpha * AP
    end
    Z .= Pr * R
    P .= A'R
    AP .= A * P
    PrAP = Pr * AP
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
    println("psd it = $i, err = $err, 
             nnz(M)/n^2 = $s,
             nnz(P)/n^2 = $(nnz(P)/n^2)")
    if (err < tol) || (s > smax)
      break
    end
  end
  if stopping_criterion == :backward_error
    return M, R_norm[1:i+1], backward_error[1:i+1]
  end
  return M, R_norm[1:i+1], densities
end


"""
psd_r(A, Pr, M, itmax, tol; stopping_criterion=:res, smax=1., ExplicitResidualUpdate=false)

Right-preconditioned steepest descent method for approximate matrix inverses.

"""
function psd_r(A, Pr, M, itmax, tol; stopping_criterion=:res, smax=1., ExplicitResidualUpdate=false)
  n = A.n
  R = spzeros(n, n)
  P = spzeros(n, n)
  Q = spzeros(n, n)
  AQ = spzeros(n, n)
  R_norm = zeros(itmax + 1)
  densities = zeros(itmax + 1)
  if stopping_criterion == :backward_error
    A_norm = frob_norm(A)
    backward_error = zeros(itmax + 1)
  end
  R .= I - A * M
  P .= A'R
  Q = Pr * P
  AQ = A * Q
  i = 0
  R_norm[i + 1] = frob_norm(R)
  densities[i + 1] = nnz(M) / n^2
  if stopping_criterion == :backward_error
    M_norm = frob_norm(M)
    backward_error[i + 1] = R_norm[i + 1] / (A_norm * M_norm + sqrt(n))
  end
  while (i < itmax)
    R_AQ_frob = frob_inner_prod(R, Q)
    AQ_norm_square = frob_norm_squared(AQ)
    alpha = R_AQ_frob / AQ_norm_square
    M .+= alpha .* Q
    if ExplicitResidualUpdate
      R .= I - A * M
    else
      R .-= alpha * AQ
    end
    P .= A'R
    Q .= Pr * P
    AQ .= A * Q
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
    println("psd it = $i, err = $err, 
             nnz(M)/n^2 = $s,
             nnz(P)/n^2 = $(nnz(P)/n^2)")
    if (err < tol) || (s > smax)
      break
    end
  end
  if stopping_criterion == :backward_error
    return M, R_norm[1:i+1], backward_error[1:i+1]
  end
  return M, R_norm[1:i+1], densities
end


"""
psd_l(A, Pr, M, itmax, tol; stopping_criterion=:res, smax=1.)

Left-preconditioned steepest descent method for approximate matrix inverses.

"""
function psd_l(A, Pr, M, itmax, tol; stopping_criterion=:res, smax=1.)
  n = A.n
  R = spzeros(n, n)
  Z = spzeros(n, n)
  P = spzeros(n, n)
  AP = spzeros(n, n)
  PrA = spzeros(n, n)
  PrAP = spzeros(n, n)
  R_norm = zeros(itmax + 1)
  densities = zeros(itmax + 1)
  if stopping_criterion == :backward_error
    A_norm = frob_norm(A)
    backward_error = zeros(itmax + 1)
  end
  PrA .= Pr * A
  R .= I - A * M
  Z .= Pr * R
  P .= PrA * Z
  PrAP .= PrA * P
  i = 0
  R_norm[i + 1] = frob_norm(R)
  densities[i + 1] = nnz(M) / n^2
  if stopping_criterion == :backward_error
    M_norm = frob_norm(M)
    backward_error[i + 1] = R_norm[i + 1] / (A_norm * M_norm + sqrt(n))
  end
  while (i < itmax)
    Z_PrAP_frob = frob_inner_prod(Z, PrAP)
    PrAP_norm_square = frob_norm_squared(PrAP)
    alpha = Z_PrAP_frob / PrAP_norm_square
    M .+= alpha .* P
    R .= I - A * M
    Z .= Pr * R
    P .= PrA * Z
    PrAP .= PrA * P
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
    println("psd it = $i, err = $err, 
             nnz(M)/n^2 = $s,
             nnz(P)/n^2 = $(nnz(P)/n^2)")
    if (err < tol) || (s > smax)
      break
    end
  end
  if stopping_criterion == :backward_error
    return M, R_norm[1:i+1], backward_error[1:i+1]
  end
  return M, R_norm[1:i+1], densities
end