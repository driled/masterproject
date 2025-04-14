import torch

# 检查CUDA是否可用
print(f"CUDA是否可用: {torch.cuda.is_available()}")

# 如果CUDA可用，显示详细信息
if torch.cuda.is_available():
    # 显示可用GPU数量
    print(f"可用GPU数量: {torch.cuda.device_count()}")

    # 当前默认GPU设备ID
    print(f"当前GPU设备ID: {torch.cuda.current_device()}")

    # 所有GPU的名称
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        props = torch.cuda.get_device_properties(i)
        print(f"  - 总内存: {props.total_memory / 1e9:.2f} GB")
        print(f"  - 多处理器数量: {props.multi_processor_count}")

    # 测试GPU上的简单计算
    print("\n执行简单测试...")
    x = torch.rand(1000, 1000).cuda()
    y = torch.rand(1000, 1000).cuda()

    # 测试时间
    import time

    start = time.time()
    z = torch.matmul(x, y)
    torch.cuda.synchronize()  # 确保操作完成
    end = time.time()

    print(f"矩阵乘法耗时: {(end - start) * 1000:.2f} 毫秒")
    print("GPU测试完成，计算正常！")
else:
    print("无法使用GPU，PyTorch将使用CPU。")
    print("如果您确信您的计算机有NVIDIA GPU，请检查CUDA和cuDNN是否正确安装。")