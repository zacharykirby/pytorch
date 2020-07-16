import torch
print(torch.cuda.is_available())

a = torch.tensor([-1.1, -2, 0, 1, 2], device="cuda")
b = torch.tensor([-1.1, -2, 0, 1, 2], device="cpu")
res = torch.foreach_add([a], 10)
print(res)

print("running b!")
print(b)
res = torch.foreach_add([b], 10)
print(res)