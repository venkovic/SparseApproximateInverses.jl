function frob_inner_prod(X, Y)
  return sum(X .* Y)
end

function frob_norm(X)
  return sqrt(sum(X .^ 2))
end

function frob_norm_squared(X)
  return sum(X .^ 2)
end

function res_frob_norm(A, M)
  return frob_norm(I - A * M)
end

function mr(A, M, itmax, tol; stopping_criterion=:res)
  n = A.n
  R_norm = zeros(itmax + 1)
  if stopping_criterion == :backward_error
    A_norm = frob_norm(A)
    backward_error = zeros(itmax + 1)
  end
  nnz = zeros(Int, itmax + 1)
  R = I - A * M
  AR = A * R
  i = 0
  R_norm[i + 1] = frob_norm(R)
  nnz[i + 1] = length(M.nzval)
  if stopping_criterion == :backward_error
    M_norm = frob_norm(M)
    backward_error[i + 1] = R_norm[i + 1] / (A_norm * M_norm + sqrt(n))
  end
  while (i < itmax)
    R_AR_frob = frob_inner_prod(R, AR)
    AR_norm_square = frob_norm_squared(AR)
    alpha = R_AR_frob / AR_norm_square
    M += alpha .* R
    R -= alpha .* AR
    AR .= A * R
    i += 1
    R_norm[i + 1] = frob_norm(R)
    nnz[i + 1] = length(M.nzval)
    err = R_norm[i + 1]
    if stopping_criterion == :backward_error
      M_norm = frob_norm(M)
      backward_error[i + 1] = R_norm[i + 1] / (A_norm * M_norm + sqrt(n))
      err = backward_error[i + 1]
    end
    println("mr it = $i, err = $err, nnz(M)/n^2 = $(nnz[i + 1]/n^2)")
    if (err < tol)
      break
    end
  end
  if stopping_criterion == :backward_error
    return M, R_norm[1:i+1], backward_error[1:i+1], nnz[1:i+1]
  end
  return M, R_norm[1:i+1], nnz[1:i+1]
end

function pmr(A, Pr, M, itmax, tol; stopping_criterion=:res)
  n = A.n
  R_norm = zeros(itmax + 1)
  if stopping_criterion == :backward_error
    A_norm = frob_norm(A)
    backward_error = zeros(itmax + 1)
  end
  nnz = zeros(Int, itmax + 1)
  PrA = Pr * A
  R = I - A * M
  Z = Pr * R
  PrAZ = PrA * Z
  i = 0
  R_norm[i + 1] = frob_norm(R)
  nnz[i + 1] = length(M.nzval)
  if stopping_criterion == :backward_error
    M_norm = frob_norm(M)
    backward_error[i + 1] = R_norm[i + 1] / (A_norm * M_norm + sqrt(n))
  end
  while (i < itmax)
    Z_PrAZ_frob = frob_inner_prod(Z, PrAZ)
    PrAZ_norm_square = frob_norm_squared(PrAZ)
    alpha = Z_PrAZ_frob / PrAZ_norm_square
    M += alpha .* Z
    R .= I - A * M
    Z .= Pr * R
    PrAZ .= PrA * Z
    i += 1
    R_norm[i + 1] = frob_norm(R)
    nnz[i + 1] = length(M.nzval)
    err = R_norm[i + 1]
    if stopping_criterion == :backward_error
      M_norm = frob_norm(M)
      backward_error[i + 1] = R_norm[i + 1] / (A_norm * M_norm + sqrt(n))
      err = backward_error[i + 1]
    end
    println("pmr it = $i, err = $err, nnz(M)/n^2 = $(nnz[i + 1]/n^2)")
    if (err < tol)
      break
    end
  end
  if stopping_criterion == :backward_error
    return M, R_norm[1:i+1], backward_error[1:i+1], nnz[1:i+1]
  end
  return M, R_norm[1:i+1], nnz[1:i+1]
end

function sd(A, M, itmax, tol; stopping_criterion=:res)
  n = A.n
  R_norm = zeros(itmax + 1)
  if stopping_criterion == :backward_error
    A_norm = frob_norm(A)
    backward_error = zeros(itmax + 1)
  end
  nnz = zeros(Int, itmax + 1)
  R = I - A * M
  P = A * R
  AP = A * P
  i = 0
  R_norm[i + 1] = frob_norm(R)
  nnz[i + 1] = length(M.nzval)
  if stopping_criterion == :backward_error
    M_norm = frob_norm(M)
    backward_error[i + 1] = R_norm[i + 1] / (A_norm * M_norm + sqrt(n))
  end
  while (i < itmax)
    R_AP_frob = frob_inner_prod(R, AP)
    AP_norm_square = frob_norm_squared(AP)
    alpha = R_AP_frob / AP_norm_square
    M += alpha .* P
    R -= alpha .* AP
    P .= A * R
    AP .= A * P
    i += 1
    R_norm[i + 1] = frob_norm(R)
    nnz[i + 1] = length(M.nzval)
    err = R_norm[i + 1]
    if stopping_criterion == :backward_error
      M_norm = frob_norm(M)
      backward_error[i + 1] = R_norm[i + 1] / (A_norm * M_norm + sqrt(n))
      err = backward_error[i + 1]
    end
    println("mr it = $i, err = $err, nnz(M)/n^2 = $(nnz[i + 1]/n^2)")
    if (err < tol)
      break
    end
  end
  if stopping_criterion == :backward_error
    return M, R_norm[1:i+1], backward_error[1:i+1], nnz[1:i+1]
  end
  return M, R_norm[1:i+1], nnz[1:i+1]
end

function cg(A, M, itmax, tol; stopping_criterion=:res)
  n = A.n
  R_norm = zeros(itmax + 1)
  if stopping_criterion == :backward_error
    A_norm = frob_norm(A)
    backward_error = zeros(itmax + 1)
  end
  nnz = zeros(Int, itmax + 1)
  R = I - A * M
  G = - A * R
  R_G_frob = frob_inner_prod(R, G)
  P = - G
  AP = A * P
  i = 0
  R_norm[i + 1] = frob_norm(R)
  nnz[i + 1] = length(M.nzval)
  if stopping_criterion == :backward_error
    M_norm = frob_norm(M)
    backward_error[i + 1] = R_norm[i + 1] / (A_norm * M_norm + sqrt(n))
  end
  while (i < itmax)
    P_AP_frob = frob_inner_prod(P, AP)
    alpha = - R_G_frob / P_AP_frob
    beta = 1. / R_G_frob
    M += alpha .* P
    R -= alpha .* AP
    G .= - A * R
    R_G_frob = frob_inner_prod(R, G)
    i += 1
    R_norm[i + 1] = frob_norm(R)
    nnz[i + 1] = length(M.nzval)
    err = R_norm[i + 1]
    if stopping_criterion == :backward_error
      M_norm = frob_norm(M)
      backward_error[i + 1] = R_norm[i + 1] / (A_norm * M_norm + sqrt(n))
      err = backward_error[i + 1]
    end
    println("mr it = $i, err = $err, nnz(M)/n^2 = $(nnz[i + 1]/n^2)")
    if (err < tol)
      break
    end
    beta *= R_G_frob
    P .*= beta
    P -= G
    AP .= A * P
  end
  if stopping_criterion == :backward_error
    return M, R_norm[1:i+1], backward_error[1:i+1], nnz[1:i+1]
  end
  return M, R_norm[1:i+1], nnz[1:i+1]
end

function pcg(A, Pr, M, itmax, tol; stopping_criterion=:res)
  n = A.n
  R_norm = zeros(itmax + 1)
  if stopping_criterion == :backward_error
    A_norm = frob_norm(A)
    backward_error = zeros(itmax + 1)
  end
  nnz = zeros(Int, itmax + 1)
  R = I - A * M
  Z = Pr * R
  G = - A * Z
  G .= Pr * G 
  R_G_frob = frob_inner_prod(R, G)
  P = - G
  AP = A * P
  i = 0
  R_norm[i + 1] = frob_norm(R)
  nnz[i + 1] = length(M.nzval)
  if stopping_criterion == :backward_error
    M_norm = frob_norm(M)
    backward_error[i + 1] = R_norm[i + 1] / (A_norm * M_norm + sqrt(n))
  end
  while (i < itmax)
    P_AP_frob = frob_inner_prod(P, AP)
    alpha = - R_G_frob / P_AP_frob
    beta = 1. / R_G_frob
    M += alpha .* P
    R -= alpha .* AP
    Z .= Pr * R
    G .= - A * Z
    G .= Pr * G
    R_G_frob = frob_inner_prod(R, G)
    i += 1
    R_norm[i + 1] = frob_norm(R)
    nnz[i + 1] = length(M.nzval)
    err = R_norm[i + 1]
    if stopping_criterion == :backward_error
      M_norm = frob_norm(M)
      backward_error[i + 1] = R_norm[i + 1] / (A_norm * M_norm + sqrt(n))
      err = backward_error[i + 1]
    end
    println("mr it = $i, err = $err, nnz(M)/n^2 = $(nnz[i + 1]/n^2)")
    if (err < tol)
      break
    end
    beta *= R_G_frob
    P .*= beta
    P -= G
    AP .= A * P
  end
  if stopping_criterion == :backward_error
    return M, R_norm[1:i+1], backward_error[1:i+1], nnz[1:i+1]
  end
  return M, R_norm[1:i+1], nnz[1:i+1]
end

function cr(A, M, itmax, tol; stopping_criterion=:res)
  n = A.n
  R_norm = zeros(itmax + 1)
  if stopping_criterion == :backward_error
    A_norm = frob_norm(A)
    backward_error = zeros(itmax + 1)
  end
  nnz = zeros(Int, itmax + 1)
  R = I - A * M
  R_frob_norm_squared = frob_norm_squared(R)
  P = copy(R)
  AP = A * P
  i = 0
  R_norm[i + 1] = sqrt(R_frob_norm_squared)
  nnz[i + 1] = length(M.nzval)
  if stopping_criterion == :backward_error
    M_norm = frob_norm(M)
    backward_error[i + 1] = R_norm[i + 1] / (A_norm * M_norm + sqrt(n))
  end
  while (i < itmax)
    P_AP_frob = frob_inner_prod(P, AP)
    alpha = R_frob_norm_squared / P_AP_frob
    beta = 1. / R_frob_norm_squared
    M += alpha .* P
    R -= alpha .* AP
    R_frob_norm_squared = frob_norm_squared(R)
    i += 1
    R_norm[i + 1] = sqrt(R_frob_norm_squared)
    nnz[i + 1] = length(M.nzval)
    err = R_norm[i + 1]
    if stopping_criterion == :backward_error
      M_norm = frob_norm(M)
      backward_error[i + 1] = R_norm[i + 1] / (A_norm * M_norm + sqrt(n))
      err = backward_error[i + 1]
    end
    println("mr it = $i, err = $err, nnz(M)/n^2 = $(nnz[i + 1]/n^2)")
    if (err < tol)
      break
    end
    beta *= R_frob_norm_squared
    P .*= beta
    P += R
    AP .= A * P
  end
  if stopping_criterion == :backward_error
    return M, R_norm[1:i+1], backward_error[1:i+1], nnz[1:i+1]
  end
  return M, R_norm[1:i+1], nnz[1:i+1]
end

function pcr(A, Pr, M, itmax, tol; stopping_criterion=:res)
  n = A.n
  R_norm = zeros(itmax + 1)
  if stopping_criterion == :backward_error
    A_norm = frob_norm(A)
    backward_error = zeros(itmax + 1)
  end
  nnz = zeros(Int, itmax + 1)
  R = I - A * M
  Z = Pr * R
  R_Z_frob = frob_inner_prod(R, Z)
  P = copy(Z)
  AP = A * P
  i = 0
  R_norm[i + 1] = frob_norm(R)
  nnz[i + 1] = length(M.nzval)
  if stopping_criterion == :backward_error
    M_norm = frob_norm(M)
    backward_error[i + 1] = R_norm[i + 1] / (A_norm * M_norm + sqrt(n))
  end
  while (i < itmax)
    P_AP_frob = frob_inner_prod(P, AP)
    alpha = R_Z_frob / P_AP_frob
    beta = 1. / R_Z_frob
    M += alpha .* P
    R -= alpha .* AP
    Z .= Pr * R
    R_Z_frob = frob_inner_prod(R, Z)
    i += 1
    R_norm[i + 1] = frob_norm(R)
    nnz[i + 1] = length(M.nzval)
    err = R_norm[i + 1]
    if stopping_criterion == :backward_error
      M_norm = frob_norm(M)
      backward_error[i + 1] = R_norm[i + 1] / (A_norm * M_norm + sqrt(n))
      err = backward_error[i + 1]
    end
    println("mr it = $i, err = $err, nnz(M)/n^2 = $(nnz[i + 1]/n^2)")
    if (err < tol)
      break
    end
    beta *= R_Z_frob
    P .*= beta
    P += Z
    AP .= A * P
  end
  if stopping_criterion == :backward_error
    return M, R_norm[1:i+1], backward_error[1:i+1], nnz[1:i+1]
  end
  return M, R_norm[1:i+1], nnz[1:i+1]
end