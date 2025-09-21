import cv2
import matplotlib.pyplot as plt
import os
import shutil
class Show:
    def cv_show(self,name,img_data):#展示
        cv2.imshow(name,img_data)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def plt_show(self,name,img_data):#展示
        plt.imshow(img_data)
        plt.title(name)
        plt.axis('off')
        plt.show()
    def plt_show_gray(self,name,img_data):#展示灰度图
        plt.imshow(img_data,cmap='gray')
        plt.title(name)
        plt.axis('off')
        plt.show()

class Utils:
    def clear_folder(self,folder_path):
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)  # 删除文件
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)  # 删除子文件夹
            except Exception as e:
                print(f'清空文件夹出错: {file_path}. 原因: {e}')

    def cv_chanel_change(self,cv_img):#通道转换
        b,g,r=cv2.split(cv_img)
        return cv2.merge([r,g,b])
    def cv_convert_gray(self,cv_img):#灰度图转换
        return cv2.cvtColor(cv_img,cv2.COLOR_BGR2GRAY)
    def edge_detection(self,cv_gray_img):#sobel算子边缘检测
        x = cv2.Sobel(cv_gray_img,cv2.CV_16S,1,0)
        y = cv2.Sobel(cv_gray_img,cv2.CV_16S,0,1)
        absX = cv2.convertScaleAbs(x)
        absY = cv2.convertScaleAbs(y)
        return cv2.addWeighted(absX,0.5,absY,0.5,0)
    def edge_detection_x(self,cv_gray_img):#sobel算子边缘检测,仅在y方向
        x = cv2.Sobel(cv_gray_img,cv2.CV_16S,1,0)
        return cv2.convertScaleAbs(x)
    def threshold(self,edge_img,low_threshold=0,high_threshold=255,my_type=None):#阈值处理
        if my_type==None: my_type=cv2.THRESH_OTSU
        return cv2.threshold(edge_img, low_threshold, high_threshold, type=my_type)

    def close_operation(self,edge_img,kernel_size=(17,5),iteration=3):#闭运算,将边缘部分连成一个整体
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,kernel_size)
        return cv2.morphologyEx(edge_img, cv2.MORPH_CLOSE, kernel,iterations=iteration)
    def dilate_erode(self,edge_img,kernel_sizeX=(15,1),kernel_sizeY=(1,15)):#膨胀腐蚀
        kernelX=cv2.getStructuringElement(cv2.MORPH_RECT,kernel_sizeX)
        kernelY=cv2.getStructuringElement(cv2.MORPH_RECT,kernel_sizeY)
        edge_img=cv2.dilate(#膨胀
            cv2.erode(#腐蚀
                cv2.erode(#腐蚀
                    cv2.dilate(edge_img,kernelX)#膨胀
                    ,kernelX)
                ,kernelY)
        ,kernelY)
        return edge_img

    def median_blur(self,cv_img,ksize=3):#中值滤波
        return cv2.medianBlur(cv_img,ksize)
    #轮廓检测
    def find_contours(self,edge_img,mode=cv2.RETR_EXTERNAL,method=cv2.CHAIN_APPROX_SIMPLE):
        contours, hierarchy = cv2.findContours(edge_img, mode, method)
        return contours
    #筛选轮廓
    def filter_contours(self,contours,cv_img,min_rate=3,max_rate=4):
        plate_img_list=[]#疑似车牌图片的存储列表
        for i in contours:
            rectangle=cv2.boundingRect(i)
            x,y=rectangle[0],rectangle[1]
            weight,height=rectangle[2],rectangle[3]
            if (weight>(height*min_rate)) and(weight<(height*max_rate)):
                target_img=plate_img_list.append(cv_img[y:y+height,x:x+weight])
                
        return plate_img_list

class Edge_img:
    def __init__(self,cv_img,**kwargs):
        self.cv_img=cv_img
        self.kwargs=kwargs
        self.edge_img=self.work_flow()
    def dege_img_transfer(self,edge_img):
        #水平对称
        self.edge_img=cv2.flip(edge_img,1)
    def work_flow(self):
        kwargs=self.kwargs
        tools=Utils()

        edge_img=tools.threshold(#阈值处理,像素值仅为0或255
            tools.edge_detection_x(#sobel算子边缘检测
                tools.cv_convert_gray(self.cv_img)#灰度图转换
            )
        )[1]
        edge_img=tools.close_operation(edge_img,kernel_size=kwargs['close_operation_kernel_size'],iteration=kwargs['close_operation_iteration'])#闭运算,将边缘部分连成一个整体
        edge_img=tools.dilate_erode(edge_img,kernel_sizeX=kwargs['dilate_erode_kernel_sizeX'],kernel_sizeY=kwargs['dilate_erode_kernel_sizeY'])#膨胀腐蚀
        edge_img=tools.median_blur(edge_img,ksize=kwargs['median_blur_ksize'])#中值滤波
        return edge_img
    def get_edge_img(self):
        return self.edge_img
