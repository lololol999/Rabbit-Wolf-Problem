import torch
from torch.autograd import variable as Variable

x = torch.tensor(torch.ones(2, 2) * 2, device = 'cuda' if torch.cuda.is_available() else 'cpu', requires_grad=True)
z = 2 * (x * x) + 5 * x
z.backward(torch.ones(2, 2))
print(x.grad)