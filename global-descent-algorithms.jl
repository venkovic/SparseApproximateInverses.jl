function mr(A, M, itmax, tol; stopping_criterion=:res)
  n = A.n
  R = spzeros(n, n)
  AR = spzeros(n, n)
  R_norm = zeros(itmax + 1)
  if stopping_criterion == :backward_error
    A_norm = frob_norm(A)
    backward_error = zeros(itmax + 1)
  end
  R .= I - A * M
  AR .= A * R
  i = 0
  R_norm[i + 1] = frob_norm(R)
  if stopping_criterion == :backward_error
    M_norm = frob_norm(M)
    backward_error[i + 1] = R_norm[i + 1] / (A_norm * M_norm + sqrt(n))
  end
  while (i < itmax)
    R_AR_frob = frob_inner_prod(R, AR)
    AR_norm_square = frob_norm_squared(AR)
    alpha = R_AR_frob / AR_norm_square
    M .+= alpha .* R
    R .-= alpha .* AR
    AR .= A * R
    i += 1
    R_norm[i + 1] = frob_norm(R)
    err = R_norm[i + 1]
    if stopping_criterion == :backward_error
      M_norm = frob_norm(M)
      backward_error[i + 1] = R_norm[i + 1] / (A_norm * M_norm + sqrt(n))
      err = backward_error[i + 1]
    end
    println("mr it = $i, err = $err, 
             nnz(M)/n^2 = $(nnz(M)/n^2)")
    if (err < tol)
      break
    end
  end
  if stopping_criterion == :backward_error
    return M, R_norm[1:i+1], backward_error[1:i+1]
  end
  return M, R_norm[1:i+1]
end

function pmr(A, Pr, M, itmax, tol; stopping_criterion=:res)
  n = A.n
  R = spzeros(n, n)
  Z = spzeros(n, n)
  PrA = spzeros(n, n)
  PrAZ = spzeros(n, n)
  R_norm = zeros(itmax + 1)
  if stopping_criterion == :backward_error
    A_norm = frob_norm(A)
    backward_error = zeros(itmax + 1)
  end
  PrA .= Pr * A
  R .= I - A * M
  Z .= Pr * R
  PrAZ .= PrA * Z
  i = 0
  R_norm[i + 1] = frob_norm(R)
  if stopping_criterion == :backward_error
    M_norm = frob_norm(M)
    backward_error[i + 1] = R_norm[i + 1] / (A_norm * M_norm + sqrt(n))
  end
  while (i < itmax)
    Z_PrAZ_frob = frob_inner_prod(Z, PrAZ)
    PrAZ_norm_square = frob_norm_squared(PrAZ)
    alpha = Z_PrAZ_frob / PrAZ_norm_square
    M .+= alpha .* Z
    R .= I - A * M
    Z .= Pr * R
    PrAZ .= PrA * Z
    i += 1
    R_norm[i + 1] = frob_norm(R)
    err = R_norm[i + 1]
    if stopping_criterion == :backward_error
      M_norm = frob_norm(M)
      backward_error[i + 1] = R_norm[i + 1] / (A_norm * M_norm + sqrt(n))
      err = backward_error[i + 1]
    end
    println("pmr it = $i, err = $err, 
             nnz(M)/n^2 = $(nnz(M)/n^2)")
    if (err < tol)
      break
    end
  end
  if stopping_criterion == :backward_error
    return M, R_norm[1:i+1], backward_error[1:i+1]
  end
  return M, R_norm[1:i+1]
end

function pmr_spai(A, Pr, M, itmax, tol, s; stopping_criterion=:res)
  n = A.n
  m = round(Int, s * n * n)
  R = spzeros(n, n)
  Z = spzeros(n, n)
  PrA = spzeros(n, n)
  PrAZ = spzeros(n, n)
  W = Z
  A_cols_dot_prods = Diagonal(vec(sum(abs2, A, dims=1)))
  R_norm = zeros(itmax + 1)
  if stopping_criterion == :backward_error
    A_norm = frob_norm(A)
    backward_error = zeros(itmax + 1)
  end
  PrA .= Pr * A
  dropping_M!(M, W, R, AR, A_cols_dot_prods, A, m)
  Z .= Pr * R
  PrAZ .= PrA * Z
  i = 0
  R_norm[i + 1] = frob_norm(R)
  if stopping_criterion == :backward_error
    M_norm = frob_norm(M)
    backward_error[i + 1] = R_norm[i + 1] / (A_norm * M_norm + sqrt(n))
  end
  while (i < itmax)
    Z_PrAZ_frob = frob_inner_prod(Z, PrAZ)
    PrAZ_norm_square = frob_norm_squared(PrAZ)
    alpha = Z_PrAZ_frob / PrAZ_norm_square
    M .+= alpha .* Z
    dropping_M!(M, W, AR, R, A_cols_dot_prods, A, m)
    Z .= Pr * R
    PrAZ .= PrA * Z
    i += 1
    R_norm[i + 1] = frob_norm(R)
    err = R_norm[i + 1]
    if stopping_criterion == :backward_error
      M_norm = frob_norm(M)
      backward_error[i + 1] = R_norm[i + 1] / (A_norm * M_norm + sqrt(n))
      err = backward_error[i + 1]
    end
    println("pmr-spai it = $i, err = $err, 
             nnz(M)/n^2 = $(nnz(M)/n^2)")
    if (err < tol)
      break
    end
  end
  if stopping_criterion == :backward_error
    return M, R_norm[1:i+1], backward_error[1:i+1]
  end
  return M, R_norm[1:i+1]
end

function lopmr(A, Pr, M, itmax, tol; stopping_criterion=:res)
  n = A.n
  R = spzeros(n, n)
  Z = spzeros(n, n)
  P = spzeros(n, n)
  AZ = spzeros(n, n)
  AP = spzeros(n, n)
  PrAP = spzeros(n, n)
  PrAZ = spzeros(n, n)
  R_norm = zeros(itmax + 1)
  if stopping_criterion == :backward_error
    A_norm = frob_norm(A)
    backward_error = zeros(itmax + 1)
  end
  R .= I - A * M
  Z .= Pr * R  
  i = 0
  R_norm[i + 1] = frob_norm(R)
  if stopping_criterion == :backward_error
    M_norm = frob_norm(M)
    backward_error[i + 1] = R_norm[i + 1] / (A_norm * M_norm + sqrt(n))
  end
  while (i < itmax)
    AP .= A * P
    AZ .= A * Z
    PrAP .= Pr * AP
    PrAZ .= Pr * AZ
    if i == 0
      AZ_PrAZ_frob = frob_inner_prod(AZ, PrAZ)
      Z_AZ_frob = frob_inner_prod(Z, AZ)
      delta = Z_AZ_frob / AZ_PrAZ_frob
      gamma = 1.
    else
      AZ_PrAZ_frob = frob_inner_prod(AZ, PrAZ)
      AP_PrAP_frob = frob_inner_prod(AP, PrAP)
      AZ_PrAP_frob = frob_inner_prod(AZ, PrAP)
      Z_AZ_frob = frob_inner_prod(Z, AZ)
      Z_AP_frob = frob_inner_prod(Z, AP)
      c = AZ_PrAZ_frob * AP_PrAP_frob - AZ_PrAP_frob ^ 2
      delta = (AP_PrAP_frob * Z_AZ_frob - AZ_PrAP_frob * Z_AP_frob) / c
      gamma = (AZ_PrAZ_frob * Z_AP_frob - AZ_PrAP_frob * Z_AZ_frob) / c
    end
    M .+= delta .* Z
    M .+= gamma .* P
    P .*= gamma / delta
    P .+= Z
    R .-= delta .* AZ
    R .-= gamma .* AP
    i += 1
    R_norm[i + 1] = frob_norm(R)
    err = R_norm[i + 1]
    if stopping_criterion == :backward_error
      M_norm = frob_norm(M)
      backward_error[i + 1] = R_norm[i + 1] / (A_norm * M_norm + sqrt(n))
      err = backward_error[i + 1]
    end
    println("lopmr it = $i, err = $err, 
             nnz(M)/n^2 = $(nnz(M)/n^2),
             nnz(P)/n^2 = $(nnz(P)/n^2)")
    if (err < tol)
      break
    end
    Z .= Pr * R
  end
  if stopping_criterion == :backward_error
    return M, R_norm[1:i+1], backward_error[1:i+1]
  end
  return M, R_norm[1:i+1]
end

function lopmr_spai(A, Pr, M, itmax, tol, s; stopping_criterion=:res)
  n = A.n
  m = round(Int, s * n * n)
  R = spzeros(n, n)
  Z = spzeros(n, n)
  P = spzeros(n, n)
  AR = spzeros(n, n)
  AZ = spzeros(n, n)
  AP = spzeros(n, n)
  PrA = spzeros(n, n)
  PrAZ = spzeros(n, n)
  PrAP = spzeros(n, n)
  W = AP
  A_cols_dot_prods = Diagonal(vec(sum(abs2, A, dims=1)))
  R_norm = zeros(itmax + 1)
  if stopping_criterion == :backward_error
    A_norm = frob_norm(A)
    backward_error = zeros(itmax + 1)
  end
  PrA .= Pr * A
  dropping_M!(M, W, R, AR, A_cols_dot_prods, A, m)
  Z .= Pr * R    
  i = 0
  R_norm[i + 1] = frob_norm(R)
  if stopping_criterion == :backward_error
    M_norm = frob_norm(M)
    backward_error[i + 1] = R_norm[i + 1] / (A_norm * M_norm + sqrt(n))
  end
  while (i < itmax)
    AP .= A * P
    AZ .= A * Z
    PrAP .= PrA * P
    PrAZ .= PrA * Z
    if i == 0
      AZ_PrAZ_frob = frob_inner_prod(AZ, PrAZ)
      Z_AZ_frob = frob_inner_prod(Z, AZ)
      delta = Z_AZ_frob / AZ_PrAZ_frob
      gamma = 1.
    else
      AZ_PrAZ_frob = frob_inner_prod(AZ, PrAZ)
      AP_PrAP_frob = frob_inner_prod(AP, PrAP)
      AZ_PrAP_frob = frob_inner_prod(AZ, PrAP)
      Z_AZ_frob = frob_inner_prod(Z, AZ)
      Z_AP_frob = frob_inner_prod(Z, AP)
      c = AZ_PrAZ_frob * AP_PrAP_frob - AZ_PrAP_frob ^ 2
      delta = (AP_PrAP_frob * Z_AZ_frob - AZ_PrAP_frob * Z_AP_frob) / c
      gamma = (AZ_PrAZ_frob * Z_AP_frob - AZ_PrAP_frob * Z_AZ_frob) / c
    end
    M .+= delta .* Z
    M .+= gamma .* P
    P .*= gamma / delta
    P .+= Z
    dropping_M!(M, W, R, AR, A_cols_dot_prods, A, m)
    dropping_P!(P, m)
    i += 1
    R_norm[i + 1] = frob_norm(R)
    err = R_norm[i + 1]
    if stopping_criterion == :backward_error
      M_norm = frob_norm(M)
      backward_error[i + 1] = R_norm[i + 1] / (A_norm * M_norm + sqrt(n))
      err = backward_error[i + 1]
    end
    println("lopmr-spai it = $i, err = $err, 
             nnz(M)/n^2 = $(nnz(M)/n^2), 
             nnz(P)/n^2 = $(nnz(P)/n^2)")
    if (err < tol)
      break
    end
    Z .= Pr * R
  end
  if stopping_criterion == :backward_error
    return M, R_norm[1:i+1], backward_error[1:i+1]
  end
  return M, R_norm[1:i+1]
end

function lopmr_spai_split(A, L, M, itmax, tol, s; stopping_criterion=:res)
  n = A.n
  m = round(Int, s * n * n)
  R = spzeros(n, n)
  P = spzeros(n, n)
  AR = spzeros(n, n)
  AP = spzeros(n, n)
  W = AP
  A_cols_dot_prods = Diagonal(vec(sum(abs2, A, dims=1)))
  R_norm = zeros(itmax + 1)
  if stopping_criterion == :backward_error
    A_norm = frob_norm(A)
    backward_error = zeros(itmax + 1)
  end
  i = 0
  dropping_M!(M, W, R, AR, A_cols_dot_prods, A, m)
  R_norm[i + 1] = frob_norm(R)
  R .= L'R
  if stopping_criterion == :backward_error
    M_norm = frob_norm(M)
    backward_error[i + 1] = R_norm[i + 1] / (A_norm * M_norm + sqrt(n))
  end
  while (i < itmax)
    AP .= A * P
    AP .= L'AP
    AR .= L * R
    AR .= A * AR
    AR .= L'AR
    if i == 0
      AZ_PrAZ_frob = frob_norm_squared(AR)
      Z_AZ_frob = frob_inner_prod(R, AR)
      delta = Z_AZ_frob / AZ_PrAZ_frob
      gamma = 1.
    else
      AZ_PrAZ_frob = frob_norm_squared(AR)
      AP_PrAP_frob = frob_norm_squared(AP)
      AZ_PrAP_frob = frob_inner_prod(AR, AP)
      Z_AZ_frob = frob_inner_prod(R, AR)
      Z_AP_frob = frob_inner_prod(R, AP)
      c = AZ_PrAZ_frob * AP_PrAP_frob - AZ_PrAP_frob ^ 2
      delta = (AP_PrAP_frob * Z_AZ_frob - AZ_PrAP_frob * Z_AP_frob) / c
      gamma = (AZ_PrAZ_frob * Z_AP_frob - AZ_PrAP_frob * Z_AZ_frob) / c
    end
    R .= L * R
    M .+= delta .* R
    M .+= gamma .* P
    P .*= gamma / delta
    P .+= R
    dropping_M!(M, W, R, AR, A_cols_dot_prods, A, m)
    dropping_P!(P, m)
    i += 1
    R_norm[i + 1] = frob_norm(R)
    err = R_norm[i + 1]
    if stopping_criterion == :backward_error
      M_norm = frob_norm(M)
      backward_error[i + 1] = R_norm[i + 1] / (A_norm * M_norm + sqrt(n))
      err = backward_error[i + 1]
    end
    println("lopmr-spai it = $i, err = $err, 
             nnz(M)/n^2 = $(nnz(M)/n^2), 
             nnz(P)/n^2 = $(nnz(P)/n^2)")
    if (err < tol)
      break
    end
    R .= L'R
  end
  if stopping_criterion == :backward_error
    return M, R_norm[1:i+1], backward_error[1:i+1]
  end
  return M, R_norm[1:i+1]
end

function sd(A, M, itmax, tol; stopping_criterion=:res)
  n = A.n
  R = spzeros(n, n)
  P = spzeros(n, n)
  AP = spzeros(n, n)
  R_norm = zeros(itmax + 1)
  if stopping_criterion == :backward_error
    A_norm = frob_norm(A)
    backward_error = zeros(itmax + 1)
  end
  R .= I - A * M
  P .= A * R
  AP .= A * P
  i = 0
  R_norm[i + 1] = frob_norm(R)
  if stopping_criterion == :backward_error
    M_norm = frob_norm(M)
    backward_error[i + 1] = R_norm[i + 1] / (A_norm * M_norm + sqrt(n))
  end
  while (i < itmax)
    R_AP_frob = frob_inner_prod(R, AP)
    AP_norm_square = frob_norm_squared(AP)
    alpha = R_AP_frob / AP_norm_square
    M .+= alpha .* P
    R .-= alpha .* AP
    P .= A * R
    AP .= A * P
    i += 1
    R_norm[i + 1] = frob_norm(R)
    err = R_norm[i + 1]
    if stopping_criterion == :backward_error
      M_norm = frob_norm(M)
      backward_error[i + 1] = R_norm[i + 1] / (A_norm * M_norm + sqrt(n))
      err = backward_error[i + 1]
    end
    println("sd it = $i, err = $err, 
             nnz(M)/n^2 = $(nnz(M)/n^2),
             nnz(P)/n^2 = $(nnz(P)/n^2)")
    if (err < tol)
      break
    end
  end
  if stopping_criterion == :backward_error
    return M, R_norm[1:i+1], backward_error[1:i+1]
  end
  return M, R_norm[1:i+1]
end

function psd(A, Pr, M, itmax, tol; stopping_criterion=:res)
  n = A.n
  R = spzeros(n, n)
  Z = spzeros(n, n)
  P = spzeros(n, n)
  AP = spzeros(n, n)
  PrA = spzeros(n, n)
  PrAP = spzeros(n, n)
  R_norm = zeros(itmax + 1)
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
    err = R_norm[i + 1]
    if stopping_criterion == :backward_error
      M_norm = frob_norm(M)
      backward_error[i + 1] = R_norm[i + 1] / (A_norm * M_norm + sqrt(n))
      err = backward_error[i + 1]
    end
    println("psd it = $i, err = $err, 
             nnz(M)/n^2 = $(nnz(M)/n^2),
             nnz(P)/n^2 = $(nnz(P)/n^2)")
    if (err < tol)
      break
    end
  end
  if stopping_criterion == :backward_error
    return M, R_norm[1:i+1], backward_error[1:i+1]
  end
  return M, R_norm[1:i+1]
end

function cg(A, M, itmax, tol; stopping_criterion=:res)
  n = A.n
  R = spzeros(n, n)
  G = spzeros(n, n)
  P = spzeros(n, n)
  AP = spzeros(n, n)
  R_norm = zeros(itmax + 1)
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
  if stopping_criterion == :backward_error
    M_norm = frob_norm(M)
    backward_error[i + 1] = R_norm[i + 1] / (A_norm * M_norm + sqrt(n))
  end
  while (i < itmax)
    P_AP_frob = frob_inner_prod(P, AP)
    alpha = - R_G_frob / P_AP_frob
    beta = 1. / R_G_frob
    M .+= alpha .* P
    R .-= alpha .* AP
    G .-= A * R
    R_G_frob = frob_inner_prod(R, G)
    i += 1
    R_norm[i + 1] = frob_norm(R)
    err = R_norm[i + 1]
    if stopping_criterion == :backward_error
      M_norm = frob_norm(M)
      backward_error[i + 1] = R_norm[i + 1] / (A_norm * M_norm + sqrt(n))
      err = backward_error[i + 1]
    end
    println("cg it = $i, err = $err, 
             nnz(M)/n^2 = $(nnz(M)/n^2),
             nnz(P)/n^2 = $(nnz(P)/n^2)")
    if (err < tol)
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
  return M, R_norm[1:i+1]
end

function pcg(A, Pr, M, itmax, tol; stopping_criterion=:res)
  n = A.n
  R = spzeros(n, n)
  Z = spzeros(n, n)
  G = spzeros(n, n)
  P = spzeros(n, n)
  AP = spzeros(n, n)
  R_norm = zeros(itmax + 1)
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
  if stopping_criterion == :backward_error
    M_norm = frob_norm(M)
    backward_error[i + 1] = R_norm[i + 1] / (A_norm * M_norm + sqrt(n))
  end
  while (i < itmax)
    P_AP_frob = frob_inner_prod(P, AP)
    alpha = - R_G_frob / P_AP_frob
    beta = 1. / R_G_frob
    M .+= alpha .* P
    R .-= alpha .* AP
    Z .= Pr * R
    G .= - A * Z
    G .= Pr * G
    R_G_frob = frob_inner_prod(R, G)
    i += 1
    R_norm[i + 1] = frob_norm(R)
    err = R_norm[i + 1]
    if stopping_criterion == :backward_error
      M_norm = frob_norm(M)
      backward_error[i + 1] = R_norm[i + 1] / (A_norm * M_norm + sqrt(n))
      err = backward_error[i + 1]
    end
    println("pcg it = $i, err = $err, 
             nnz(M)/n^2 = $(nnz(M)/n^2),
             nnz(P)/n^2 = $(nnz(P)/n^2)")
    if (err < tol)
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
  return M, R_norm[1:i+1]
end

function cr(A, M, itmax, tol; stopping_criterion=:res)
  n = A.n
  R = spzeros(n, n)
  P = spzeros(n, n)
  AP = spzeros(n, n)
  R_norm = zeros(itmax + 1)
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
  if stopping_criterion == :backward_error
    M_norm = frob_norm(M)
    backward_error[i + 1] = R_norm[i + 1] / (A_norm * M_norm + sqrt(n))
  end
  while (i < itmax)
    P_AP_frob = frob_inner_prod(P, AP)
    alpha = R_frob_norm_squared / P_AP_frob
    beta = 1. / R_frob_norm_squared
    M .+= alpha .* P
    R .-= alpha .* AP
    R_frob_norm_squared = frob_norm_squared(R)
    i += 1
    R_norm[i + 1] = sqrt(R_frob_norm_squared)
    err = R_norm[i + 1]
    if stopping_criterion == :backward_error
      M_norm = frob_norm(M)
      backward_error[i + 1] = R_norm[i + 1] / (A_norm * M_norm + sqrt(n))
      err = backward_error[i + 1]
    end
    println("cr it = $i, err = $err, 
             nnz(M)/n^2 = $(nnz(M)/n^2), 
             nnz(P)/n^2 = $(nnz(P)/n^2)")
    if (err < tol)
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
  return M, R_norm[1:i+1]
end

function pcr(A, Pr, M, itmax, tol; stopping_criterion=:res)
  n = A.n
  R = spzeros(n, n)
  P = spzeros(n, n)
  Z = spzeros(n, n)
  AP = spzeros(n, n)
  R_norm = zeros(itmax + 1)
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
  if stopping_criterion == :backward_error
    M_norm = frob_norm(M)
    backward_error[i + 1] = R_norm[i + 1] / (A_norm * M_norm + sqrt(n))
  end
  while (i < itmax)
    P_AP_frob = frob_inner_prod(P, AP)
    alpha = R_Z_frob / P_AP_frob
    beta = 1. / R_Z_frob
    M .+= alpha .* P
    R .-= alpha .* AP
    Z .= Pr * R
    R_Z_frob = frob_inner_prod(R, Z)
    i += 1
    R_norm[i + 1] = frob_norm(R)
    err = R_norm[i + 1]
    if stopping_criterion == :backward_error
      M_norm = frob_norm(M)
      backward_error[i + 1] = R_norm[i + 1] / (A_norm * M_norm + sqrt(n))
      err = backward_error[i + 1]
    end
    println("pcr it = $i, err = $err, 
             nnz(M)/n^2 = $(nnz(M)/n^2), 
             nnz(P)/n^2 = $(nnz(P)/n^2)")
    if (err < tol)
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
  return M, R_norm[1:i+1]
end

function pcr_spai(A, Pr, M, itmax, tol, s; stopping_criterion=:res)
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
  if stopping_criterion == :backward_error
    A_norm = frob_norm(A)
    backward_error = zeros(itmax + 1)
  end
  dropping_M!(M, W, R, AR, A_cols_dot_prods, A, m)
  Z .= Pr * R
  R_Z_frob = frob_inner_prod(R, Z)
  P .= Z
  dropping_P!(P, m)
  AP .= A * P
  i = 0
  R_norm[i + 1] = frob_norm(R)
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
    err = R_norm[i + 1]
    if stopping_criterion == :backward_error
      M_norm = frob_norm(M)
      backward_error[i + 1] = R_norm[i + 1] / (A_norm * M_norm + sqrt(n))
      err = backward_error[i + 1]
    end
    println("pcr-spai it = $i, err = $err, 
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
  return M, R_norm[1:i+1]
end

function lopcr(A, Pr, M, itmax, tol; stopping_criterion=:res)
  n = A.n
  R = spzeros(n, n)
  Z = spzeros(n, n)
  P = spzeros(n, n)
  AZ = spzeros(n, n)
  AP = spzeros(n, n)
  R_norm = zeros(itmax + 1)
  if stopping_criterion == :backward_error
    A_norm = frob_norm(A)
    backward_error = zeros(itmax + 1)
  end
  R .= I - A * M
  Z .= Pr * R
  P = spzeros(n, n)
  i = 0
  R_norm[i + 1] = frob_norm(R)
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
    err = R_norm[i + 1]
    if stopping_criterion == :backward_error
      M_norm = frob_norm(M)
      backward_error[i + 1] = R_norm[i + 1] / (A_norm * M_norm + sqrt(n))
      err = backward_error[i + 1]
    end
    println("lopcr it = $i, err = $err, 
             nnz(M)/n^2 = $(nnz(M)/n^2), 
             nnz(P)/n^2 = $(nnz(P)/n^2)")
    if (err < tol)
      break
    end
    P .*= gamma / delta
    P .+= Z
    Z .= Pr * R
  end
  if stopping_criterion == :backward_error
    return M, R_norm[1:i+1], backward_error[1:i+1]
  end
  return M, R_norm[1:i+1]
end