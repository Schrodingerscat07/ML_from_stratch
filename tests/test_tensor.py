import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tensor_autograd.tensor import Tensor

def test_matmul_grad():
    a = Tensor([[1.0, 2.0], [3.0, 4.0]])
    b = Tensor([[5.0, 6.0], [7.0, 8.0]])
    c = a @ b
    c.backward()
    
    # Check gradients manually
    # dL/da = grad_c @ b.T
    # grad_c is ones_like(c)
    expected_grad_a = np.ones((2, 2)) @ b.data.T
    expected_grad_b = a.data.T @ np.ones((2, 2))
    
    assert np.allclose(a.grad, expected_grad_a)
    assert np.allclose(b.grad, expected_grad_b)
    print("Matmul gradient test passed!")

def test_transpose_grad():
    a = Tensor([[1.0, 2.0], [3.0, 4.0]])
    b = a.T
    b.backward()
    
    assert np.allclose(a.grad, np.ones((2, 2)).T)
    print("Transpose gradient test passed!")

if __name__ == "__main__":
    test_matmul_grad()
    test_transpose_grad()
