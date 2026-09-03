import torch

# 1. 检查 GPU (CUDA) 是否可用
gpu_available = torch.cuda.is_available()
print(f"CUDA 是否可用: {gpu_available}")
print(torch.__version__); 
print(torch.version.cuda); 
print(torch.cuda.is_available()); 
print(torch.cuda.device_count()); 
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU')

if gpu_available:
    # 2. 查看可用的 GPU 数量
    device_count = torch.cuda.device_count()
    print(f"检测到的 GPU 数量: {device_count}")
    
    # 3. 查看当前使用的 GPU 名称
    current_device_name = torch.cuda.get_device_name(torch.cuda.current_device())
    print(f"当前设备名称: {current_device_name}")
    
    # 4. 查看 PyTorch 绑定的 CUDA 版本
    print(f"CUDA 版本: {torch.version.cuda}")
else:
    print("未检测到可用的 GPU，将使用 CPU 进行训练。")