import numpy as np

def gumbel_top_k(phi, k, rng=None):
    """
    Sample an ordered sample of size k without replacement from 
    the categorical distribution defined by unnormalized log-probs phi,
    using the Gumbel-top-k trick.

    Args:
        phi (array_like): 1D sequence of unnormalized log-probabilities (phi_i).
        k (int): Number of items to sample (k <= len(phi)).
        rng (np.random.Generator, optional): A NumPy random Generator for reproducibility.

    Returns:
        topk_indices (np.ndarray): The indices of the k largest perturbed values,
                                   in descending order of perturbed score.
    """
    phi = np.asarray(phi)
    n = phi.shape[0]
    if k > n:
        raise ValueError(f"k = {k} is larger than number of categories n = {n}")

    # get a random generator
    rng = rng or np.random.default_rng()
    # draw i.i.d. Gumbel(0) noise: -log(-log(U))
    U = rng.random(n)
    gumbels = -np.log(-np.log(U))
    # perturb
    perturbed = phi + gumbels
    # get indices of the top k largest
    # argsort(-x) gives descending order; take first k
    topk_indices = np.argsort(-perturbed)[:k]
    return topk_indices

# Example usage:
if __name__ == "__main__":
    # Suppose we have 5 categories with these log-probs:
    phi = [2.0, 0.5, 1.2, 3.7, 0.1]
    k = 3
    print("Sampling top-3 without replacement from C(exp(phi)):")
    idx = gumbel_top_k(phi, k)
    print("Picked indices:", idx)
