def add_cyclic_offset(offset, L):
    """
    Add cyclic offset to the offset matrix.
    j - i is how many residues j has to jump
    to get to i without taking into account cyclization.
    If positive, that means going left to right.
    If negative, that means going right to left.
    We only calculate the dists for the upper right, 
    ie j > i, for simplicity, so j will always be 
    in front of i in the sequence.
    
    If j - i > L/2, then j would have to go so many
    residues to the left that it would be better to go
    right instead, taking advantage of the link between
    the first and last residues. To get this offset, 
    we subtract L from the dists, which will give 
    the offset going the other way. This will be negative
    since we're going right to left.

    To fill in the lower left, we just take the negative
    of the upper right since the distance is the same 
    but we're going in the opposite direction.
    """
    max_dist = (L // 2)
    idxs = jnp.arange(offset.shape[0])
    i = idxs[:, None]
    j = idxs[None, :]
    set_indices = (i < L) & (j < L)
    dists = abs(j - i)
    dists = jnp.where((dists > max_dist), dists-L, dists)
    upper_right = (i < j)
    offset = jnp.where(set_indices & upper_right, -dists, offset)
    offset = jnp.where(set_indices & ~upper_right, dists, offset)
    return offset