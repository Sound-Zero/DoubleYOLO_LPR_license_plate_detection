import cv2
import matplotlib.pyplot as plt
from my_yolo import predict_once
import os
from ultralytics import YOLO
import cv2
from LPR_detect import LPRNet_predict
from my_tool import *
import numpy as np
import argparse


def get_obj_img(img_path,detect_classes=['1'],bottom_half=True):
    car_img_list=[]#存储所有车辆图片cv_img
    img_path=img_path
    img_name_with_ext = os.path.basename(img_path)
    img_name, img_ext = os.path.splitext(img_name_with_ext)


    #找出最新生成的图片预测文件夹
    folders = [entry for entry in os.listdir("./runs/detect") if os.path.isdir(os.path.join("./runs/detect", entry))]
    latest_folder = ''
    if folders:
        latest_folder = max(folders, key=lambda folder: os.path.getmtime(os.path.join("./runs/detect", folder)))
    label_path='./runs/detect/'+latest_folder+'/labels/'+img_name+'.txt'
    
    if os.path.exists(img_path) and os.path.exists(label_path):
        img=cv2.imread(img_path)
        vehicle_obj=[]
        #读取txt每一行内容
        context=[line.strip() for line in open(label_path, 'r').readlines()]
        print(f"该图片共有{len(context)}个实例",context)
        for obj in context:
            for classes in detect_classes:
                if  obj.startswith(classes):
                    vehicle_obj.append(obj)
                    break
        for obj in vehicle_obj:
            print(f"检测到指定类型目标{vehicle_obj.index(obj)+1}/{len(vehicle_obj)}")
            x_center, y_center, width, height = [float(i) for i in obj.split()[1:]]
            cv_img=crop_img(img_path, x_center, y_center, width, height,bottom_half=bottom_half)
            car_img_list.append(cv_img)
    else:
        print("图片不存在或yolo预测结果路径有误")
    return car_img_list



def crop_img(img_path,x_center, y_center, width, height,bottom_half=True):#默认截取目标图片下半部分
    img=cv2.imread(img_path)
    H,W=img.shape[:2]
    #检测框左上右下角坐标
    x_min = int((x_center - width / 2) * W)
    y_min = int((y_center - height / 2) * H)
    x_max = int((x_center + width / 2) * W)
    y_max = int((y_center + height / 2) * H)
    
    if bottom_half:
        #仅保留物体下半部分区域
        delta_y= y_max-y_min
        y_min=y_min+delta_y//2
    # 确保坐标在图片范围内
    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(W, x_max)
    y_max = min(H, y_max)
    # 提取检测框中的图片部分
    cropped_img = img[y_min:y_max, x_min:x_max]
    # # 显示裁剪后的图片
    # cv2.imshow('Cropped Image', cropped_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return cropped_img

def plate_img_save(cv_img,save_dir):#调整图片形状，保存图片
    # 图像配置
    IMG_WIDTH = 94
    IMG_HEIGHT = 24
    IMG_CHANNELS = 3
    # 调整图片形状
    cv_img = cv2.resize(cv_img, (IMG_WIDTH, IMG_HEIGHT))

    # 保存图片
    cv2.imwrite(os.path.join(save_dir, 'plate_img.jpg'), cv_img)



def main(kwargs=None):
    if kwargs is None:
        kwargs = {
                'find_car_model':'./model/car.pt',
                'find_plate_license_model':'./model/car_license.pt',
                'lpr_model':'./model/Final_LPRNet_model.pth',
                'img_path':'./test_img/car1.jpg',

                'car_img_save_dir':'./car_img_cache',
                'plate_img_save_dir':'./plate_img_cache',


                "close_operation_kernel_size": (17, 2),
                "close_operation_iteration": 2,
                "dilate_erode_kernel_sizeX": (15, 2),
                "dilate_erode_kernel_sizeY": (2, 15),
                "median_blur_ksize": 15
            }

    tools=Utils()
    #清空YOLO、图片缓存文件
    if os.path.exists(kwargs['car_img_save_dir']):
        tools.clear_folder(kwargs['car_img_save_dir'])
    if os.path.exists(kwargs['plate_img_save_dir']):
        tools.clear_folder(kwargs['plate_img_save_dir'])
    if os.path.exists('./runs/detect'):
        tools.clear_folder('./runs/detect')


    '''
    检测车辆
    '''
    print("尝试定位车辆...")
    car_imgs=[]
    predict_once(model=kwargs["find_car_model"],img_path=kwargs['img_path'],conf=0.4,show=False,detect_classes=[1])#使用yolo检测车辆目标，car.pt默认使用1代表vehicle类
    car_imgs=get_obj_img(kwargs['img_path'],detect_classes=['1'])#获取每辆车下半部


    for img in car_imgs:#保存所有车的图片
        cv2.imwrite(os.path.join(kwargs['car_img_save_dir'],f'cropped_{car_imgs.index(img)+1}.jpg'), img)
    predict_once(model=kwargs["find_plate_license_model"],img_path=kwargs['car_img_save_dir'],conf=0.4,show=False,detect_classes=[0])#使用yolo检测车牌目标，默认使用0代表车牌类
    

    ''' 
    检测车牌
    '''
    #检测car_img_save_dir是否为空
    if not os.listdir(kwargs['car_img_save_dir']):
        print('未检测到车辆')
        return
    print("尝试定位车牌...")
    plate_img=[]
    for img_name in os.listdir(kwargs['car_img_save_dir']):#获取每辆车的车牌
        if img_name.endswith('.jpg'):
            img_path=os.path.join(kwargs['car_img_save_dir'],img_name)
            plate_img.extend(get_obj_img(img_path,detect_classes=['0'],bottom_half=False))
    for img in plate_img:#保存所有车牌图片
        cv2.imwrite(os.path.join(kwargs['plate_img_save_dir'],f'plate_{plate_img.index(img)+1}.jpg'), img)
      
    '''
    检测车牌字符
    '''
    #检测plate_img_save_dir是否为空
    if not os.listdir(kwargs['plate_img_save_dir']):
        print('未检测到车牌')
        return
    print("尝试识别车牌...")
    parser = argparse.ArgumentParser(description='LPRNet Predictor')
    parser.add_argument('--model_path', type=str, default=kwargs['lpr_model'], 
                    help='模型权重文件路径')
    parser.add_argument('--test_dir', type=str, default=kwargs['plate_img_save_dir'], 
                    help='测试图片目录')
    parser.add_argument('--single_image', type=str, default='', 
                    help='测试单张图片路径')
    parser.add_argument('--save_results', type=str, default='', 
                    help='保存测试结果的文件路径')
    LPRNet_predict(parser=parser)



# def cv_plate_detection():
#     obj_img_list=[]
#     predict_once(model=kwargs["model_path"],img_path=kwargs['img_path'],conf=0.4,show=False)
#     obj_img_list=get_obj_img(kwargs['img_path'])
#     if obj_img_list:
#         for img in obj_img_list:
#             show,tools,_=Show(),Utils(),Edge_img(img, **kwargs)
#             edge_img=_.get_edge_img()#获取边缘图
#             contours=tools.find_contours(edge_img)#找出轮廓
#             plate_img=tools.filter_contours(contours,img,2,5)#筛选轮廓
            
#             #show.cv_show('img_contours',cv2.drawContours(img,tools.find_contours(edge_img),-1,(0,255,0),2))
#             if plate_img:
#                 print('检测出疑似车牌物')
#                 for i in plate_img:

#                     img_save(i,'./')
#                     my_ocr()
          
#     else:#YOLO预测结果为空
#         print('YOLO未检测到车辆目标')
#         img=cv2.imread(kwargs['img_path'])
#         show,tools,_=Show(),Utils(),Edge_img(img, **kwargs)
#         edge_img=_.get_edge_img()#获取边缘图
#         contours=tools.find_contours(edge_img)#找出轮廓
#         plate_img=tools.filter_contours(contours,img,2,5)#筛选轮廓
        
#         if plate_img:
#             print('检测出疑似车牌物')
#             for i in plate_img:
#                 img_save(i,'./')
#                 my_ocr()


#     show.cv_show('edge_img',edge_img)
#     show.cv_show('img_contours',cv2.drawContours(img,tools.find_contours(edge_img),-1,(0,255,0),2))

if __name__ == '__main__':
    main()
