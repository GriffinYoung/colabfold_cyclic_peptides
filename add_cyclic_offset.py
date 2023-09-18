def add_cyclic_offset(offset, L):
  max_dist = (L // 2)
  idxs = jnp.arange(offset.shape[0])
  i = idxs[:,None]
  j = idxs[None,:]
  set_indices = (i < L) & (j < L)
  dists = abs(j - i)
  dists = jnp.where((dists > max_dist), L - dists, dists)
  upper_right = (i < j)
  offset = jnp.where(set_indices & upper_right, -dists, offset)
  offset = jnp.where(set_indices & ~upper_right, dists, offset)
  return offset