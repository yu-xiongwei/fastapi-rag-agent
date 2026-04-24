# check_cuda.py
import torch
print("PyTorch版本:", torch.__version__)
if torch.cuda.is_available():
    print("CUDA可用!")
    print("CUDA版本:", torch.version.cuda)
    print("显卡型号:", torch.cuda.get_device_name(0))
else:
    print("CUDA不可用。")