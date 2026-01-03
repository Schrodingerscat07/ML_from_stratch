class Value:
    def __init__(self, data, _children=(), _op='', label=''):
        self.data = data
        self.grad = 0.0             # Stores the derivative (gradient)
        self._prev = set(_children) # The Memory: Stores the parent nodes
        self._op = _op              # The Operation: Stores how it was made (+, *)
        self.label = label          # Purely Optional: Name the variable (e.g., 'a', 'b')

    def __repr__(self):
        return f"Value(data={self.data})"

    def __add__(self, other):
        # Make sure we are adding two Value objects
        other = other if isinstance(other, Value) else Value(other)
        
        # Create the new node
        out = Value(self.data + other.data, (self, other), '+') # Memory: Stores the parent nodes and the operator
        return out
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*") # Memory: Stores the parent nodes and the operator
        return out



