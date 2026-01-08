import numpy as np
import sys
import os

# Add the root directory to path to import tensor_autograd
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tensor_autograd.tensor import Tensor

def svd(A, k=None, epsilon=1e-10, max_iterations=100):
    """
    Computes the Singular Value Decomposition (SVD) of a matrix A using Power Iteration.
    A = U S V^T
    
    Args:
        A: Input matrix (numpy array or Tensor)
        k: Number of singular values/vectors to compute. If None, computes all.
        epsilon: Convergence threshold.
        max_iterations: Maximum iterations for power iteration.
        
    Returns:
        U, S, V_T as numpy arrays.
    """
    if isinstance(A, Tensor):
        A_data = A.data
    else:
        A_data = np.array(A)
        
    n, m = A_data.shape
    if k is None:
        k = min(n, m)
        
    A_rem = A_data.copy().astype(np.float64)
    U = np.zeros((n, k))
    S = np.zeros(k)
    V = np.zeros((m, k))
    
    for i in range(k):
        # Power Iteration to find the first singular value/vector of A_rem
        # We find the eigenvector of A_rem.T @ A_rem which corresponds to the first right singular vector v
        # Alternatively, we can use the Power Iteration on A_rem @ A_rem.T to find u
        
        # Initialize a random vector
        v = np.random.rand(m)
        v /= np.linalg.norm(v)
        
        for _ in range(max_iterations):
            # v_{next} = A^T A v
            v_next = A_rem.T @ (A_rem @ v)
            v_next_norm = np.linalg.norm(v_next)
            if v_next_norm < 1e-12:
                break
            v_next /= v_next_norm
            
            if np.abs(np.dot(v, v_next)) > 1 - epsilon:
                v = v_next
                break
            v = v_next
            
        # Singular value sigma = |A v|
        Av = A_rem @ v
        sigma = np.linalg.norm(Av)
        
        if sigma < 1e-12:
            break
            
        # Left singular vector u = A v / sigma
        u = Av / sigma
        
        U[:, i] = u
        S[i] = sigma
        V[:, i] = v
        
        # Deflate: Subtract the component found from the matrix
        # A_next = A - sigma * u * v^T
        A_rem -= sigma * np.outer(u, v)
        
    return U, S, V.T

if __name__ == "__main__":
    # Simple test
    A = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float64)
    U, S, VT = svd(A)
    
    print("Original Matrix A:\n", A)
    print("\nReconstructed Matrix (U @ S @ VT):\n", U @ np.diag(S) @ VT)
    
    # Verify with numpy
    U_np, S_np, VT_np = np.linalg.svd(A, full_matrices=False)
    print("\nNumpy Singular Values:", S_np)
    print("Our Singular Values:  ", S)
