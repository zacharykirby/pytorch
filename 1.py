import torch
print(torch.cuda.is_available())

a = torch.tensor([-1.1, -2, 0, 1, 2], device="cuda")
b = torch.tensor([-1.1, -2, 0, 1, 2], device="cuda")
res = torch.foreach_add_scalar([a], 10)
print(res)

res = torch.foreach_add_scalar([b], 10)
print(res)