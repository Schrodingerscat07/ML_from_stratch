import math

class Value:
    # In this constructor takes 4 arguments
    # data: the value of the variable
    # _children: the children of the variable (the elemnts from which it was derived)
    # _op: the operation that was performed to get the value
    # label: the label of the variable (optional)
    def __init__(self, data, _children=(), _op='', label=''):
        self.data = data
        self.grad = 0.0 # stores the gradient
        self._backward = lambda: None # stores the backward function None by default
        self._prev = set(_children) # stores the previous values
        self._op = _op # stores the operation
        self.label = label # stores the label (optional)

    def __repr__(self): # returns the string representation of the value you can try without this to see the dfference
        return f"Value(data={self.data}, grad={self.grad})"

    def __add__(self, other): # overloading the + operator
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward(): # stores the backward function for addition which is just adding the gradients
          self.grad += out.grad #the gradient of the addition is the gradient of the output
          other.grad += out.grad
        out._backward = _backward
        return out
    def __sub__(self, other): # overloading the - operator
        return self + (-other)
    def __mul__(self, other): # overloading the * operator
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward(): # stores the backward function for multiplication which is just adding the gradients
          self.grad += other.data * out.grad #the gradient of the multiplication is the product of the other value and the gradient of the output
          other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def tanh(self): # overloading the tanh operator used to squeeze the values between -1 and 1
        x = self.data
        t = (math.exp(2*x) - 1)/(math.exp(2*x) + 1)
        out = Value(t, (self, ), 'tanh')

        def _backward(): # stores the backward function for tanh which is just adding the gradients
          self.grad += (1 - t**2) * out.grad #the gradient of the tanh is the product of the other value and the gradient of the output
        out._backward = _backward
        return out


    def backward(self): # now you can try using _backward for each variable to understadn the working but this does all that for you
      topo = []
      visited = set()
      def build_topo(v): # builds the topological order of the variables
        if v not in visited:
          visited.add(v)
          for child in v._prev:
            build_topo(child)
          topo.append(v)
      build_topo(self)

      self.grad = 1.0
      for node in reversed(topo):
        node._backward()


    def __neg__(self): # -self
        return self * -1

    def __rmul__(self, other):# overloading the * operator to follow the rules of multiplication
      return self * other


    def exp(self): # overloading the exp operator 
      x = self.data
      out = Value(math.exp(x), (self, ), 'exp')
      def _backward():
        self.grad += out.data * out.grad
      out._backward = _backward
      return out

    def __pow__(self, other): # overloading the ** operator
      assert isinstance(other, (int, float)), "only supporting int/float powers for now"
      out = Value(self.data**other, (self, ), f'**{other}')
      def _backward():
        self.grad += (other * self.data**(other-1)) * out.grad
      out._backward = _backward
      return out

    def __radd__(self, other): # overloading the + operator to follow the rules of addition
      return self + other

    def __truediv__(self, other): # overloading the / operator to follow the rules of division
        return self * other**-1
""" note that you can use different functions to perform the same operation but make sure to update 
the backward function accordingly in my version i used both tanh the shorter form and the other form 
of tanh(including the exp function) to calculate the gradient"""


""" I have added a viz_autograd function to visualize the autograd process to use it to understand how the backward function works"""