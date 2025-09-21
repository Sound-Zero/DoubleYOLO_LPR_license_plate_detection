import cv2
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
from LPRNet import build_lprnet, CHARS,CN
import argparse
import time
from tqdm import tqdm

# 数据集类
class LPRDataset(Dataset):
    def __init__(self, data_dir, transform=None, max_len=8):
        self.data_dir = data_dir
        self.transform = transform
        self.max_len = max_len
        self.samples = []
        
        # 遍历数据目录，获取所有图片文件
        for filename in os.listdir(data_dir):
            if filename.endswith('.jpg') or filename.endswith('.png'):
                # 从文件名提取车牌号
                plate_text = filename.split('.')[0]
                self.samples.append((filename, plate_text))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        filename, plate_text = self.samples[idx]
        img_path = os.path.join(self.data_dir, filename)
        
        # 读取图片
        img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), -1)
        if img is None:
            img = cv2.imread(img_path)
        
        # 调整图片大小
        img = cv2.resize(img, (94, 24))
        
        # 应用变换
        if self.transform:
            img = self.transform(img)
        
        # 将车牌文字转换为标签
        label = self.encode_text(plate_text)
        
        return img, label, len(label)
    
    def encode_text(self, text):
        """将车牌文字编码为数字标签"""
        encoded = []
        for char in text:
            if char in CHARS:
                encoded.append(CHARS.index(char))
            elif char in CN:# 中文字符
                encoded.append(CN.index(char))
                if char == '京':
                    pass
                elif char == '宁':
                    pass
            else:
                # 如果字符不在CHARS、CN中，跳过或用特殊标记
                continue
        return encoded

# 图像变换
def get_transforms():
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

# CTC损失函数
class CTCLoss(nn.Module):
    def __init__(self):
        super(CTCLoss, self).__init__()
        self.ctc_loss = nn.CTCLoss(blank=len(CHARS)-1, reduction='mean', zero_infinity=True)
    
    def forward(self, log_probs, targets, input_lengths, target_lengths):
        return self.ctc_loss(log_probs, targets, input_lengths, target_lengths)

# 训练函数
def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    num_batches = 0
    
    for batch_idx, (images, labels, label_lengths) in enumerate(tqdm(dataloader, desc="Training")):
        images = images.to(device)
        
        # 前向传播
        outputs = model(images)
        outputs = torch.log_softmax(outputs, dim=1)
        
        # 准备CTC损失的输入
        batch_size = outputs.size(0)
        input_lengths = torch.full((batch_size,), outputs.size(2), dtype=torch.long)
        
        # 将标签转换为一维张量
        targets = []
        target_lengths = []
        for label in labels:
            targets.extend(label)
            target_lengths.append(len(label))
        
        targets = torch.tensor(targets, dtype=torch.long)
        target_lengths = torch.tensor(target_lengths, dtype=torch.long)
        
        # 计算损失
        loss = criterion(outputs.permute(2, 0, 1), targets, input_lengths, target_lengths)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / num_batches

# 验证函数
def validate_epoch(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    num_batches = 0
    correct_predictions = 0
    total_predictions = 0
    
    with torch.no_grad():
        for images, labels, label_lengths in tqdm(dataloader, desc="Validating"):
            images = images.to(device)
            
            # 前向传播
            outputs = model(images)
            outputs = torch.log_softmax(outputs, dim=1)
            
            # 准备CTC损失的输入
            batch_size = outputs.size(0)
            input_lengths = torch.full((batch_size,), outputs.size(2), dtype=torch.long)
            
            # 将标签转换为一维张量
            targets = []
            target_lengths = []
            for label in labels:
                targets.extend(label)
                target_lengths.append(len(label))
            
            targets = torch.tensor(targets, dtype=torch.long)
            target_lengths = torch.tensor(target_lengths, dtype=torch.long)
            
            # 计算损失
            loss = criterion(outputs.permute(2, 0, 1), targets, input_lengths, target_lengths)
            total_loss += loss.item()
            num_batches += 1
            
            # 计算准确率（简单的贪心解码）
            for i in range(batch_size):
                pred_text = decode_prediction(outputs[i])
                true_text = ''.join([CHARS[idx] for idx in labels[i]])
                if pred_text == true_text:
                    correct_predictions += 1
                total_predictions += 1
    
    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
    return total_loss / num_batches, accuracy

# 解码预测结果
def decode_prediction(output):
    """简单的贪心解码"""
    pred_indices = torch.argmax(output, dim=0)
    decoded = []
    prev_idx = -1
    
    for idx in pred_indices:
        idx = idx.item()
        if idx != prev_idx and idx != len(CHARS) - 1:  # 不是空白字符且不重复
            decoded.append(CHARS[idx])
        prev_idx = idx
    
    return ''.join(decoded)

# 自定义collate函数
def collate_fn(batch):
    images, labels, lengths = zip(*batch)
    images = torch.stack(images)
    return images, labels, lengths

# 主训练函数
def main():
    parser = argparse.ArgumentParser(description='LPRNet Training')
    parser.add_argument('--train_dir', type=str, default='./data/train', help='训练数据目录')
    parser.add_argument('--val_dir', type=str, default='./data/val', help='验证数据目录')
    parser.add_argument('--batch_size', type=int, default=256, help='批次大小')
    parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--lr', type=float, default=0.0001, help='学习率')
    parser.add_argument('--save_dir', type=str, default='./checkpoints', help='模型保存目录')
    parser.add_argument('--resume', type=str, default='', help='恢复训练的模型路径')
    
    args = parser.parse_args()
    
    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 设备设置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')
    
    # 数据变换
    transform = get_transforms()
    
    # 数据集和数据加载器
    train_dataset = LPRDataset(args.train_dir, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, 
                             collate_fn=collate_fn, num_workers=4)
    
    # 验证集（如果存在）
    val_loader = None
    if os.path.exists(args.val_dir):
        val_dataset = LPRDataset(args.val_dir, transform=transform)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                               collate_fn=collate_fn, num_workers=4)
    
    # 模型
    model = build_lprnet(lpr_max_len=8, phase="train", class_num=len(CHARS), dropout_rate=0.5)
    model = model.to(device)
    
    # 损失函数和优化器
    criterion = CTCLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    
    # 恢复训练
    start_epoch = 0
    if args.resume and os.path.exists(args.resume):
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f'从第 {start_epoch} 轮恢复训练')
    
    # 训练循环
    best_val_loss = float('inf')
    
    for epoch in range(start_epoch, args.epochs):
        print(f'\n轮次 {epoch+1}/{args.epochs}')
        print('-' * 50)
        
        # 训练
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        print(f'训练损失: {train_loss:.4f}')
        
        # 验证
        if val_loader:
            val_loss, val_acc = validate_epoch(model, val_loader, criterion, device)
            print(f'验证损失: {val_loss:.4f}, 验证准确率: {val_acc:.4f}')
        else:
            val_loss = train_loss
        
        # 学习率调度
        scheduler.step()
        
        # 保存模型
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss
        }
        
        # 保存最新模型
        torch.save(checkpoint, os.path.join(args.save_dir, 'latest_model.pth'))
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(checkpoint, os.path.join(args.save_dir, 'best_model.pth'))
            torch.save(model.state_dict(), os.path.join(args.save_dir, 'Final_LPRNet_model.pth'))
            print('保存最佳模型')
        
        # 定期保存
        if (epoch + 1) % 10 == 0:
            torch.save(checkpoint, os.path.join(args.save_dir, f'model_epoch_{epoch+1}.pth'))
    
    print('训练完成！')

if __name__ == '__main__':
    main()
