import threading
import time
import keyboard
import os
import torch

a = torch.ones(2,3)
b = 2*torch.ones(2,1)

print(a)
print(b)
print(torch.cat((a,b),1))