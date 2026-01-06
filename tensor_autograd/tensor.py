import numpy as np

class Tensor:
    def __init__(self, data, _children=(), _op='', label=''):
        self.data = np.array(data)
        self.grad = np.zeros_like(self.data, dtype=np.float64)
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
        self.label = label

    def __repr__(self):
        return f"Tensor(data={self.data}, grad={self.grad})"

    def backward(self):
        # Topological sort for tensor graph
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        # Reset gradients might be handled differently in complex scenarios,
        # but for simplicity, we follow the scalar logic here:
        self.grad = np.ones_like(self.data, dtype=np.float64)
        for node in reversed(topo):
            node._backward()
