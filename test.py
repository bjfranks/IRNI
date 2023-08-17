import torch
import time

input = torch.arange(0, 20)
print(input.dtype)
start = time.time()
output = torch.nn.functional.one_hot(input, num_classes=20)
end = time.time()
print(end-start)
