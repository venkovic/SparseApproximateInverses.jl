function frob_inner_prod(X, Y)
  return dot(X, Y)
end

function frob_norm(X)
  return sqrt(dot(X, X))
end

function frob_norm_squared(X)
  return dot(X, X)
end

function res_frob_norm(A, M)
  return frob_norm(I - A * M)
end