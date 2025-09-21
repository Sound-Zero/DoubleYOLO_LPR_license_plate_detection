''' 
测试LPRNet模型性能
'''
import cv2
import os
import torch
import numpy as np
from torchvision import transforms
from lprr.LPRNet import build_lprnet, CHARS
import argparse
from tqdm import tqdm

def get_transforms():
    """获取图像预处理变换"""
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def decode_prediction(output):
    """解码模型预测结果"""
    pred_indices = torch.argmax(output, dim=0)
    decoded = []
    prev_idx = -1
    
    for idx in pred_indices:
        idx = idx.item()
        if idx != prev_idx and idx != len(CHARS) - 1:  # 不是空白字符且不重复
            decoded.append(CHARS[idx])
        prev_idx = idx
    
    return ''.join(decoded)

def test_single_image(model, image_path, transform, device):
    """测试单张图片"""
    # 读取图片
    img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), -1)
    if img is None:
        img = cv2.imread(image_path)
    
    if img is None:
        print(f"无法读取图片: {image_path}")
        return None, None
    
    # 调整图片大小
    img_resized = cv2.resize(img, (94, 24))
    
    # 应用变换
    img_tensor = transform(img_resized).unsqueeze(0).to(device)
    
    # 模型预测
    model.eval()
    with torch.no_grad():
        output = model(img_tensor)
        output = torch.log_softmax(output, dim=1)
        pred_text = decode_prediction(output[0])
    
    return pred_text, img

def test_directory(model, test_dir, transform, device):
    """测试目录中的所有图片"""
    results = []
    correct = 0
    total = 0

    # 省份映射（中文到拼音）
    PROVINCE_MAP = {
        '京': 'jing', '沪': 'hu', '津': 'jin', '渝': 'yu', '冀': 'yi',
        '晋': 'jin', '蒙': 'meng', '辽': 'liao', '吉': 'ji', '黑': 'hei',
        '苏': 'su', '浙': 'zhe', '皖': 'wan', '闽': 'min', '赣': 'gan',
        '鲁': 'lu', '豫': 'yu', '鄂': 'e', '湘': 'xiang', '粤': 'yue',
        '桂': 'gui', '琼': 'qiong', '川': 'chuan', '贵': 'gui', '云': 'yun',
        '藏': 'zang', '陕': 'shan', '甘': 'gan', '青': 'qing', '宁': 'ning',
        '新': 'xin'
    }
    
    # 获取所有图片文件
    image_files = [f for f in os.listdir(test_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
    
    for filename in tqdm(image_files, desc="测试图片"):
        image_path = os.path.join(test_dir, filename)
        
        # 从文件名获取真实标签
        true_label = filename.split('.')[0]
        for char in true_label:
            # 检查字符是否为中文字（通过 Unicode 范围判断）
            if '\u4e00' <= char <= '\u9fff':
                # 如果中文字在字典的 key 中，用对应的 value 替换
                if char in PROVINCE_MAP:
                    true_label = true_label.replace(char, PROVINCE_MAP[char])
                    break  # 假设字符串中单只有个中文字，替换后退出循环

        
        # 预测
        pred_text, img = test_single_image(model, image_path, transform, device)
        
        if pred_text is not None:
            results.append({
                'filename': filename,
                'true_label': true_label,
                'prediction': pred_text,
                'correct': pred_text == true_label
            })
            
            if pred_text == true_label:
                correct += 1
            total += 1
            
            print(f"文件: {filename}")
            print(f"真实: {true_label}")
            print(f"预测: {pred_text}")
            print(f"正确: {'✓' if pred_text == true_label else '✗'}")
            print("-" * 50)
    
    accuracy = correct / total if total > 0 else 0
    print(f"\n总体准确率: {accuracy:.4f} ({correct}/{total})")
    
    return results, accuracy

def main(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser(description='LPRNet Testing')
        parser.add_argument('--model_path', type=str, default='./checkpoints/Final_LPRNet_model.pth', 
                        help='模型权重文件路径')
        parser.add_argument('--test_dir', type=str, default='./data/train', 
                        help='测试图片目录')
        parser.add_argument('--single_image', type=str, default='', 
                        help='测试单张图片路径')
        parser.add_argument('--save_results', type=str, default='', 
                        help='保存测试结果的文件路径')
        
    
    args = parser.parse_args()
    
    # 设备设置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')
    
    # 加载模型
    model = build_lprnet(lpr_max_len=8, phase="test", class_num=len(CHARS), dropout_rate=0.5)
    
    if os.path.exists(args.model_path):
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        print(f"加载模型: {args.model_path}")
    else:
        print(f"模型文件不存在: {args.model_path}")
        return
    
    model = model.to(device)
    
    # 图像变换
    transform = get_transforms()
    
    if args.single_image:
        # 测试单张图片
        if os.path.exists(args.single_image):
            pred_text, img = test_single_image(model, args.single_image, transform, device)
            if pred_text is not None:
                print(f"预测结果: {pred_text}")
                
                # 显示图片
                cv2.imshow('Test Image', img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            else:
                print("预测失败")
        else:
            print(f"图片文件不存在: {args.single_image}")
    
    elif os.path.exists(args.test_dir):
        # 测试目录
        results, accuracy = test_directory(model, args.test_dir, transform, device)
        
        # 保存结果
        if args.save_results:
            import json
            with open(args.save_results, 'w', encoding='utf-8') as f:
                json.dump({
                    'accuracy': accuracy,
                    'results': results
                }, f, ensure_ascii=False, indent=2)
            print(f"结果已保存到: {args.save_results}")
    
    else:
        print(f"测试目录不存在: {args.test_dir}")

def LPRNet_predict(parser=None):#仅预测，不计算正确率

    dic={
        'jing': '京', 'hu': '沪', 'jin': '津', 'yu': '渝', 'yi': '冀',
        'jin': '晋', 'meng': '蒙', 'liao': '辽', 'ji': '吉', 'hei': '黑',
        'su': '苏', 'zhe': '浙', 'wan': '皖','min': '闽', 'gan': '赣',
        'lu': '鲁', 'yu': '豫', 'e': '鄂', 'xiang': '湘', 'yue': '粤',
        'gui': '桂', 'qiong': '琼', 'chuan': '川', 'gui': '贵', 'yun': '云',
        'zang': '藏', 'shan': '陕', 'gan': '甘', 'qing': '青', 'ning': '宁',
        'xin': '新',
    }
    if parser is None:
        parser = argparse.ArgumentParser(description='LPRNet Testing')
        parser.add_argument('--model_path', type=str, default='./checkpoints/Final_LPRNet_model.pth', 
                        help='模型权重文件路径')
        parser.add_argument('--test_dir', type=str, default='./data/train', 
                        help='测试图片目录')
        parser.add_argument('--single_image', type=str, default='', 
                        help='测试单张图片路径')
        parser.add_argument('--save_results', type=str, default='', 
                        help='保存测试结果的文件路径')
        
    
    args = parser.parse_args()
    
    # 设备设置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')
    
    # 加载模型
    model = build_lprnet(lpr_max_len=8, phase="test", class_num=len(CHARS), dropout_rate=0.5)
    
    if os.path.exists(args.model_path):
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        print(f"加载模型: {args.model_path}")
    else:
        print(f"模型文件不存在: {args.model_path}")
        return
    
    model = model.to(device)
    
    # 图像变换
    transform = get_transforms()
    
    if args.single_image:
        # 测试单张图片
        if os.path.exists(args.single_image):
            pred_text, img = test_single_image(model, args.single_image, transform, device)
            if pred_text is not None:
                #拼音转中文
                province_text="".join([char for char in pred_text if char.islower()])
                if province_text in dic:
                    pred_text=pred_text.replace(province_text,dic[province_text])
                print(f"预测结果: {pred_text}")
            else:
                print("预测失败")
        else:
            print(f"图片文件不存在: {args.single_image}")
    elif args.test_dir:
        # 获取所有图片文件
        image_files = [f for f in os.listdir(args.test_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
        for filename in image_files:
            file_path = os.path.join(args.test_dir, filename)
            if os.path.exists(file_path):
                pred_text, img = test_single_image(model, file_path, transform, device)
                if pred_text is not None:
                    #拼音转中文
                    province_text="".join([char for char in pred_text if char.islower()])
                    if province_text in dic:
                        pred_text=pred_text.replace(province_text,dic[province_text])
                    print(f"图片{filename}预测结果: {pred_text}")
                else:
                    print("预测失败")
            else:
                print(f"图片文件不存在: {file_path}")
    else:
        print("请指定测试图片目录或单张图片路径")






if __name__ == '__main__':
    LPRNet_predict()