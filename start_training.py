#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
快速启动训练脚本
这个脚本提供了一个简单的界面来配置和启动LPRNet训练
"""

import os
import sys
import argparse
from config import Config

def check_environment():
    """检查训练环境"""
    print("检查训练环境...")
    
    # 检查PyTorch
    try:
        import torch
        print(f"✓ PyTorch版本: {torch.__version__}")
        if torch.cuda.is_available():
            print(f"✓ CUDA可用，设备数量: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"  - GPU {i}: {torch.cuda.get_device_name(i)}")
        else:
            print("⚠ CUDA不可用，将使用CPU训练")
    except ImportError:
        print("✗ PyTorch未安装")
        return False
    
    # 检查其他依赖
    required_packages = ['cv2', 'numpy', 'tqdm', 'sklearn']
    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ {package}已安装")
        except ImportError:
            print(f"✗ {package}未安装")
            return False
    
    return True

def check_data():
    """检查数据目录"""
    print("\n检查数据目录...")
    
    if not os.path.exists(Config.TRAIN_DIR):
        print(f"✗ 训练数据目录不存在: {Config.TRAIN_DIR}")
        return False
    
    # 统计训练数据
    train_files = [f for f in os.listdir(Config.TRAIN_DIR) 
                   if f.endswith(('.jpg', '.png', '.jpeg'))]
    print(f"✓ 训练数据: {len(train_files)} 张图片")
    
    if len(train_files) == 0:
        print("✗ 训练数据目录为空")
        return False
    
    # 检查验证数据
    if os.path.exists(Config.VAL_DIR):
        val_files = [f for f in os.listdir(Config.VAL_DIR) 
                     if f.endswith(('.jpg', '.png', '.jpeg'))]
        print(f"✓ 验证数据: {len(val_files)} 张图片")
    else:
        print(f"⚠ 验证数据目录不存在: {Config.VAL_DIR}")
        print("  建议创建验证集以监控训练效果")
    
    return True

def setup_directories():
    """创建必要的目录"""
    print("\n创建必要的目录...")
    Config.create_dirs()
    print(f"✓ 检查点目录: {Config.SAVE_DIR}")
    print(f"✓ 日志目录: {Config.LOG_DIR}")

def get_user_config():
    """获取用户配置"""
    print("\n=== 训练配置 ===")
    
    # 批次大小
    while True:
        try:
            batch_size = input(f"批次大小 (默认 {Config.BATCH_SIZE}): ").strip()
            if not batch_size:
                batch_size = Config.BATCH_SIZE
            else:
                batch_size = int(batch_size)
            break
        except ValueError:
            print("请输入有效的数字")
    
    # 训练轮数
    while True:
        try:
            epochs = input(f"训练轮数 (默认 {Config.EPOCHS}): ").strip()
            if not epochs:
                epochs = Config.EPOCHS
            else:
                epochs = int(epochs)
            break
        except ValueError:
            print("请输入有效的数字")
    
    # 学习率
    while True:
        try:
            lr = input(f"学习率 (默认 {Config.LEARNING_RATE}): ").strip()
            if not lr:
                lr = Config.LEARNING_RATE
            else:
                lr = float(lr)
            break
        except ValueError:
            print("请输入有效的数字")
    
    return {
        'batch_size': batch_size,
        'epochs': epochs,
        'lr': lr
    }

def start_training(config):
    """启动训练"""
    print("\n=== 开始训练 ===")
    
    # 构建训练命令
    cmd = [
        sys.executable,
        os.path.join('lprr', 'train.py'),
        '--train_dir', Config.TRAIN_DIR,
        '--batch_size', str(config['batch_size']),
        '--epochs', str(config['epochs']),
        '--lr', str(config['lr']),
        '--save_dir', Config.SAVE_DIR
    ]
    
    # 添加验证目录（如果存在）
    if os.path.exists(Config.VAL_DIR):
        cmd.extend(['--val_dir', Config.VAL_DIR])
    
    print(f"执行命令: {' '.join(cmd)}")
    print("\n" + "="*50)
    
    # 执行训练
    import subprocess
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"\n训练过程中出现错误: {e}")
        return False
    except KeyboardInterrupt:
        print("\n训练被用户中断")
        return False
    
    return True

def main():
    parser = argparse.ArgumentParser(description='LPRNet训练启动器')
    parser.add_argument('--auto', action='store_true', help='使用默认配置自动开始训练')
    parser.add_argument('--check-only', action='store_true', help='仅检查环境，不开始训练')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("LPRNet 中国车牌识别训练项目")
    print("=" * 60)
    
    # 检查环境
    if not check_environment():
        print("\n环境检查失败，请安装必要的依赖包:")
        print("pip install -r requirements.txt")
        return
    
    # 检查数据
    if not check_data():
        print("\n数据检查失败，请确保数据目录存在且包含训练图片")
        print("数据格式: 图片文件名为车牌号，如 '京A12345.jpg'")
        return
    
    # 创建目录
    setup_directories()
    
    if args.check_only:
        print("\n环境检查完成！")
        return
    
    # 获取配置
    if args.auto:
        config = {
            'batch_size': Config.BATCH_SIZE,
            'epochs': Config.EPOCHS,
            'lr': Config.LEARNING_RATE
        }
        print("\n使用默认配置:")
        print(f"批次大小: {config['batch_size']}")
        print(f"训练轮数: {config['epochs']}")
        print(f"学习率: {config['lr']}")
    else:
        config = get_user_config()
    
    # 确认开始训练
    if not args.auto:
        confirm = input("\n确认开始训练? (y/N): ").strip().lower()
        if confirm not in ['y', 'yes']:
            print("训练已取消")
            return
    
    # 开始训练
    success = start_training(config)
    
    if success:
        print("\n" + "="*50)
        print("训练完成！")
        print(f"模型保存在: {Config.SAVE_DIR}")
        print("\n可以使用以下命令测试模型:")
        print(f"python test.py --model_path {Config.SAVE_DIR}/Final_LPRNet_model.pth")
    else:
        print("\n训练未能完成")

if __name__ == '__main__':
    main()