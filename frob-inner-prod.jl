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