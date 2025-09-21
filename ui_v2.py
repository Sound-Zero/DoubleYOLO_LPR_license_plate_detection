import sys
import os
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QScrollArea, QLabel, QTextEdit, QPushButton, QFrame, QFileDialog, 
                             QGridLayout, QSpinBox, QGroupBox, QMessageBox)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QObject, pyqtSlot
from PyQt5.QtGui import QPixmap, QFont
from PyQt5.Qt import QSettings

class OutputRedirect(QObject):
    """用于线程安全的输出重定向"""
    output_signal = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.stdout = sys.stdout
        self.stderr = sys.stderr

    def write(self, text):
        if text.strip():
            self.output_signal.emit(text.strip())

    def flush(self):
        pass

class DetectionThread(QThread):
    """检测线程，避免UI冻结"""
    finished = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(self, kwargs):
        super().__init__()
        self.kwargs = kwargs

    def run(self):
        try:
            from main import main
            main(self.kwargs)
            self.finished.emit()
        except Exception as e:
            self.error.emit(str(e))

class CarDetectionUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.kwargs = {
            'find_car_model': './model/car.pt',
            'find_plate_license_model': './model/car_license.pt',
            'lpr_model': './model/Final_LPRNet_model.pth',
            'img_path': './car1.jpg',
            'car_img_save_dir': './car_img_cache',
            'plate_img_save_dir': './plate_img_cache',
            "close_operation_kernel_size": (17, 2),
            "close_operation_iteration": 2,
            "dilate_erode_kernel_sizeX": (15, 2),
            "dilate_erode_kernel_sizeY": (2, 15),
            "median_blur_ksize": 15
        }
        self.detection_thread = None
        self.font_size = 14
        self.settings = QSettings('LicensePlateUI', 'Settings')
        self.load_settings()
        self.initUI()
        self.apply_font_settings()
    def load_settings(self):
        """加载设置"""
        self.font_size = self.settings.value('font_size', 14, type=int)

    def save_settings(self):
        """保存设置"""
        self.settings.setValue('font_size', self.font_size)

    def change_font_size(self, size):
        """改变字体大小"""
        self.font_size = size
        self.apply_font_settings()
        self.save_settings()
    def apply_font_settings(self):
        """应用字体设置"""
        if hasattr(self, 'output_text'):
            
            self.output_text.setStyleSheet(f"""
                QTextEdit {{
                    background-color: #2b2b2b;
                    color: #ffffff;
                    border: 1px solid #ced4da;
                    border-radius: 5px;
                    padding: 10px;
                    font-family: 'Consolas', 'Monaco', monospace;
                    font-size: {self.font_size}px;
                    line-height: 1.4;
                }}
            """)


    def initUI(self):
        self.setWindowTitle('车牌检测系统')
        #设置Icon
        if os.path.exists('./icon.jpg'):
            from PyQt5.QtGui import QIcon
            self.setWindowIcon(QIcon('./icon.jpg'))
        self.setGeometry(100, 100, 1920, 1080)#设置窗口大小为1200*700
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # 顶部控制区域
        top_layout = QHBoxLayout()
        
        # 左侧控制面板
        control_frame = QFrame()
        control_frame.setFixedWidth(300)    # 设置宽度
        control_frame.setStyleSheet("QFrame { background-color: #f8f9fa; border: 1px solid #dee2e6; border-radius: 8px; }")
        control_layout = QVBoxLayout(control_frame)
        control_layout.setSpacing(12)
        control_layout.setContentsMargins(15, 15, 15, 15)#设置边距
        
        # 字体设置区域
        font_group = QGroupBox('字体设置')
        font_group.setFont(QFont('Arial', 10, QFont.Bold))
        font_group.setStyleSheet("QGroupBox { color: #495057; font-weight: bold; }")
        font_layout = QHBoxLayout(font_group)
        
        font_label = QLabel('大小:')
        self.font_spinbox = QSpinBox()
        self.font_spinbox.setRange(10, 20)#设置最小值和最大值
        self.font_spinbox.setValue(self.font_size)
        self.font_spinbox.setSuffix('px')
        self.font_spinbox.valueChanged.connect(self.change_font_size)
        
        font_layout.addWidget(font_label)
        font_layout.addWidget(self.font_spinbox)
        font_layout.addStretch()
        control_layout.addWidget(font_group)
        
        # 按钮样式
        button_style = """
            QPushButton {
                background-color: #007bff;
                color: white;
                border: none;
                border-radius: 6px;
                padding: 10px;
                font-size: 20px;
                font-weight: 500;
            }
            QPushButton:hover {
                background-color: #0056b3;
            }
            QPushButton:pressed {
                background-color: #004085;
            }
        """
        
        start_button_style = """
            QPushButton {
                background-color: #28a745;
                color: white;
                border: none;
                border-radius: 6px;
                padding: 12px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #218838;
            }
            QPushButton:pressed {
                background-color: #1e7e34;
            }
        """
        
        # 创建按钮
        self.btn_select_image = QPushButton('选择图片')
        self.btn_select_image.setStyleSheet(button_style)
        self.btn_select_image.clicked.connect(self.select_image)
        
        self.btn_add_car_model = QPushButton('车辆检测模型')
        self.btn_add_car_model.setStyleSheet(button_style)
        self.btn_add_car_model.clicked.connect(self.select_car_model)
        
        self.btn_add_plate_model = QPushButton('车牌检测模型')
        self.btn_add_plate_model.setStyleSheet(button_style)
        self.btn_add_plate_model.clicked.connect(self.select_plate_model)
        
        self.btn_add_lpr_model = QPushButton('LPR识别模型')
        self.btn_add_lpr_model.setStyleSheet(button_style)
        self.btn_add_lpr_model.clicked.connect(self.select_lpr_model)
        
        self.btn_start_detection = QPushButton('开始检测')
        self.btn_start_detection.setStyleSheet(start_button_style)
        self.btn_start_detection.clicked.connect(self.start_detection)
        
        # 添加按钮到布局
        control_layout.addWidget(self.btn_select_image)
        control_layout.addWidget(self.btn_add_car_model)
        control_layout.addWidget(self.btn_add_plate_model)
        control_layout.addWidget(self.btn_add_lpr_model)
        control_layout.addStretch()
        control_layout.addWidget(self.btn_start_detection)
        
        # 右侧图片显示区域
        self.create_image_areas(top_layout, control_frame)
        
        main_layout.addLayout(top_layout, 1)
        
        # 底部输出区域
        self.create_output_area(main_layout)
        
        # 设置输出重定向
        self.output_redirect = OutputRedirect()
        self.output_redirect.output_signal.connect(self.append_output_text)
        sys.stdout = self.output_redirect
        sys.stderr = self.output_redirect

    def create_output_area(self, main_layout):
        """创建底部程序输出区域"""
        output_frame = QFrame()
        output_frame.setStyleSheet("QFrame { border: 1px solid #dee2e6; border-radius: 8px; background-color: #f8f9fa; }")
        output_layout = QVBoxLayout(output_frame)
        output_layout.setContentsMargins(5, 5, 5, 5)#设置边距

        title = QLabel('程序输出')
        title.setAlignment(Qt.AlignLeft)
        title.setStyleSheet("font-weight: bold; color: #495057; font-size: 14px; margin-bottom: 5px;")
        output_layout.addWidget(title)

        self.output_text = QTextEdit()
        self.output_text.setMaximumHeight(120)
        self.output_text.setStyleSheet("background-color: #2b2b2b; color: #ffffff; font-family: Consolas, Monaco, monospace; border: 1px solid #495057; border-radius: 4px;")
        self.output_text.append("等待检测开始...")
        output_layout.addWidget(self.output_text)
        
        main_layout.addWidget(output_frame)

    @pyqtSlot(str)
    def append_output_text(self, text):
        """线程安全的输出添加方法"""
        self.output_text.append(text)
        self.output_text.ensureCursorVisible()

    def create_image_areas(self, top_layout, control_frame):
        """创建图片显示区域"""
        top_layout.addWidget(control_frame)
        
        # 右侧图片显示区域
        display_widget = QWidget()
        display_layout = QVBoxLayout(display_widget)
        display_layout.setContentsMargins(10, 0, 0, 0)
        
        # 原始图片区域
        origin_frame = QFrame()
        origin_frame.setStyleSheet("QFrame { border: 1px solid #dee2e6; border-radius: 8px; background-color: #ffffff; }")
        origin_layout = QVBoxLayout(origin_frame)
        origin_layout.setContentsMargins(10, 10, 10, 10)
        
        origin_title = QLabel('原始图片')
        origin_title.setAlignment(Qt.AlignLeft)
        origin_title.setStyleSheet("font-weight: bold; color: #495057; font-size: 14px; margin-bottom: 5px;")
        
        self.origin_img_label = QLabel()
        self.origin_img_label.setAlignment(Qt.AlignCenter)
        self.origin_img_label.setMinimumSize(400, 250)
        self.origin_img_label.setStyleSheet("border: 2px dashed #dee2e6; border-radius: 4px; background-color: #f8f9fa;")
        self.origin_img_label.setText("请选择图片")
        
        origin_layout.addWidget(origin_title)
        origin_layout.addWidget(self.origin_img_label, 1)
        
        # 检测结果区域
        result_layout = QHBoxLayout()
        
        # 车辆检测结果
        car_frame = QFrame()
        car_frame.setStyleSheet("QFrame { border: 1px solid #dee2e6; border-radius: 8px; background-color: #ffffff; }")
        car_layout = QVBoxLayout(car_frame)
        car_layout.setContentsMargins(10, 10, 10, 10)
        
        car_title = QLabel('检测到的车辆')
        car_title.setAlignment(Qt.AlignLeft)
        car_title.setStyleSheet("font-weight: bold; color: #495057; font-size: 14px; margin-bottom: 5px;")
        
        car_scroll = QScrollArea()
        car_scroll.setWidgetResizable(True)
        car_scroll.setMinimumHeight(150)
        car_scroll.setStyleSheet("QScrollArea { border: none; background-color: #f8f9fa; }")
        
        self.car_img_widget = QWidget()
        self.car_img_layout = QVBoxLayout(self.car_img_widget)
        car_scroll.setWidget(self.car_img_widget)
        
        car_layout.addWidget(car_title)
        car_layout.addWidget(car_scroll, 1)
        
        # 车牌检测结果
        plate_frame = QFrame()
        plate_frame.setStyleSheet("QFrame { border: 1px solid #dee2e6; border-radius: 8px; background-color: #ffffff; }")
        plate_layout = QVBoxLayout(plate_frame)
        plate_layout.setContentsMargins(10, 10, 10, 10)
        
        plate_title = QLabel('检测到的车牌')
        plate_title.setAlignment(Qt.AlignLeft)
        plate_title.setStyleSheet("font-weight: bold; color: #495057; font-size: 14px; margin-bottom: 5px;")
        
        plate_scroll = QScrollArea()
        plate_scroll.setWidgetResizable(True)
        plate_scroll.setMinimumHeight(150)
        plate_scroll.setStyleSheet("QScrollArea { border: none; background-color: #f8f9fa; }")
        
        self.plate_img_widget = QWidget()
        self.plate_img_layout = QGridLayout(self.plate_img_widget)
        plate_scroll.setWidget(self.plate_img_widget)
        
        plate_layout.addWidget(plate_title)
        plate_layout.addWidget(plate_scroll, 1)
        
        result_layout.addWidget(car_frame)
        result_layout.addWidget(plate_frame)
        
        display_layout.addWidget(origin_frame, 2)
        display_layout.addLayout(result_layout, 1)
        
        top_layout.addWidget(display_widget, 1)

    def display_image(self, image_path, label, max_size=None):
        """在标签中显示图片，等比例缩放"""
        try:
            if not os.path.exists(image_path):
                return False
                
            pixmap = QPixmap(image_path)
            if pixmap.isNull():
                return False
            
            # 设置最大尺寸
            if max_size:
                max_width, max_height = max_size
            else:
                # 确保有合理的默认尺寸
                max_width, max_height = 300, 200
            
            # 等比例缩放
            scaled_pixmap = pixmap.scaled(
                max_width, max_height, 
                Qt.KeepAspectRatio, 
                Qt.SmoothTransformation
            )
            
            label.setPixmap(scaled_pixmap)
            return True
        except Exception as e:
            # 避免直接操作输出，改用信号
            self.output_redirect.write(f"显示图片失败 {image_path}: {str(e)}")
            return False

    def select_image(self):
        """选择本地图像"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, '选择图片', '', 
            'Image files (*.jpg *.jpeg *.png *.bmp *.tiff *.tif)'
        )
        if file_path:
            self.kwargs['img_path'] = file_path
            self.display_image(file_path, self.origin_img_label)
            self.output_redirect.write(f"已选择图片: {file_path}")
    
    def select_car_model(self):
        """选择车辆检测模型"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, '选择车辆检测模型', '.', 
            'Model files (*.pt *.pth)'
        )
        if file_path:
            # 转换为相对路径
            rel_path = os.path.relpath(file_path)
            self.kwargs['find_car_model'] = rel_path
            self.output_redirect.write(f"已选择车辆检测模型: {rel_path}")
    
    def select_plate_model(self):
        """选择车牌检测模型"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, '选择车牌检测模型', '.', 
            'Model files (*.pt *.pth)'
        )
        if file_path:
            rel_path = os.path.relpath(file_path)
            self.kwargs['find_plate_license_model'] = rel_path
            self.output_redirect.write(f"已选择车牌检测模型: {rel_path}")
    
    def select_lpr_model(self):
        """选择LPR网络模型"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, '选择LPR网络模型', '.', 
            'Model files (*.pt *.pth)'
        )
        if file_path:
            rel_path = os.path.relpath(file_path)
            self.kwargs['lpr_model'] = rel_path
            self.output_redirect.write(f"已选择LPR网络模型: {rel_path}")
    
    def show_settings(self):
        """显示其他设置"""
        QMessageBox.information(self, '设置', '其他设置功能待实现')
    
    def validate_paths(self):
        """验证路径有效性"""
        errors = []
        
        # 检查图片路径
        if not os.path.exists(self.kwargs['img_path']):
            errors.append(f"图片路径不存在: {self.kwargs['img_path']}")
        
        # 检查模型路径
        models = {
            '车辆检测模型': self.kwargs['find_car_model'],
            '车牌检测模型': self.kwargs['find_plate_license_model'],
            'LPR网络模型': self.kwargs['lpr_model']
        }
        
        for name, path in models.items():
            if not os.path.exists(path):
                errors.append(f"{name}路径不存在: {path}")
        
        # 检查保存目录
        save_dirs = {
            '车辆图片保存目录': self.kwargs['car_img_save_dir'],
            '车牌图片保存目录': self.kwargs['plate_img_save_dir']
        }
        
        for name, path in save_dirs.items():
            if not os.path.exists(path):
                try:
                    os.makedirs(path, exist_ok=True)
                    self.output_redirect.write(f"已创建目录: {path}")
                except Exception as e:
                    errors.append(f"无法创建{name}: {path} - {str(e)}")
        
        return errors
    
    def start_detection(self):
        """开始检测"""
        # 验证路径
        errors = self.validate_paths()
        if errors:
            error_msg = '\n'.join(errors)
            QMessageBox.critical(self, '路径验证失败', f'请检查以下路径:\n{error_msg}')
            return
        
        # 禁用开始按钮防止重复点击
        self.btn_start_detection.setEnabled(False)
        self.btn_start_detection.setText('检测中...')
        
        # 清空输出区域
        self.output_text.clear()
        self.output_text.append("开始检测...")
        self.output_text.append(f"图片路径: {self.kwargs['img_path']}")
        
        # 重定向输出
        sys.stdout = self.output_redirect
        sys.stderr = self.output_redirect
        
        # 创建并启动检测线程
        self.detection_thread = DetectionThread(self.kwargs.copy())
        self.detection_thread.finished.connect(self.on_detection_finished)
        self.detection_thread.error.connect(self.on_detection_error)
        self.detection_thread.start()
    
    def on_detection_finished(self):
        """检测完成回调"""
        # 恢复输出
        sys.stdout = self.output_redirect.stdout
        sys.stderr = self.output_redirect.stderr
        
        # 恢复按钮状态
        self.btn_start_detection.setEnabled(True)
        self.btn_start_detection.setText('开始检测')
        
        self.output_text.append("\n检测完成!")
        
        # 更新显示区域
        self.update_result_display()
    
    def on_detection_error(self, error_msg):
        """检测错误回调"""
        # 恢复输出
        sys.stdout = self.output_redirect.stdout
        sys.stderr = self.output_redirect.stderr
        
        # 恢复按钮状态
        self.btn_start_detection.setEnabled(True)
        self.btn_start_detection.setText('开始检测')
        
        self.output_text.append(f"\n检测出错: {error_msg}")
        QMessageBox.critical(self, '检测失败', f'检测过程中出现错误:\n{error_msg}')
    
    def update_result_display(self):
        """更新结果显示区域"""
        try:
            # 更新车辆图片显示
            car_dir = self.kwargs['car_img_save_dir']
            if os.path.exists(car_dir):
                self.update_car_images(car_dir)
            
            # 更新车牌图片显示
            plate_dir = self.kwargs['plate_img_save_dir']
            if os.path.exists(plate_dir):
                self.update_plate_images(plate_dir)
                
        except Exception as e:
            self.output_text.append(f"更新显示时出错: {str(e)}")
    
    def update_car_images(self, car_dir):
        """更新车辆图片显示"""
        # 清空现有的车辆图片
        for i in reversed(range(self.car_img_layout.count())):
            self.car_img_layout.itemAt(i).widget().setParent(None)
        
        # 添加新的车辆图片
        for filename in os.listdir(car_dir):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                img_path = os.path.join(car_dir, filename)
                img_label = QLabel()
                img_label.setAlignment(Qt.AlignCenter)
                img_label.setStyleSheet("border: 1px solid #aaa; margin: 2px;")
                
                if self.display_image(img_path, img_label, max_size=(200, 150)):
                    self.car_img_layout.addWidget(img_label)
                else:
                    img_label.setText(filename)
                    img_label.setMinimumSize(200, 50)
                    self.car_img_layout.addWidget(img_label)
    
    def update_plate_images(self, plate_dir):
        """更新车牌图片显示"""
        # 清空现有的车牌图片
        for i in reversed(range(self.plate_img_layout.count())):
            item = self.plate_img_layout.itemAt(i)
            if item:
                item.widget().setParent(None)
        
        # 添加新的车牌图片
        row, col = 0, 0
        for filename in os.listdir(plate_dir):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                img_path = os.path.join(plate_dir, filename)
                img_label = QLabel()
                img_label.setAlignment(Qt.AlignCenter)
                img_label.setStyleSheet("border: 1px solid #aaa; margin: 2px;")
                img_label.setMinimumSize(150, 50)
                
                if self.display_image(img_path, img_label, max_size=(150, 50)):
                    self.plate_img_layout.addWidget(img_label, row, col)
                else:
                    img_label.setText(filename)
                    self.plate_img_layout.addWidget(img_label, row, col)
                
                col += 1
                if col >= 2:
                    col = 0
                    row += 1

def main():
    app = QApplication(sys.argv)
    
    # 设置应用程序样式
    app.setStyleSheet("""
        QMainWindow {
            background-color: #ffffff;
        }
        QLabel {
            font-size: 12px;
        }
        QTextEdit {
            font-size: 11px;
        }
    """)
    
    window = CarDetectionUI()
    window.show()
    
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()