"""
美颜相机 - Kivy移动应用
实时人脸美颜处理应用

功能：
- 实时摄像头预览
- 实时美颜处理
- 美颜参数调节
- 拍照保存

依赖：
- kivy
- opencv-python
- numpy
- dlib
"""

import os
import time
from datetime import datetime

import cv2
import numpy as np

from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.button import Button
from kivy.uix.slider import Slider
from kivy.uix.label import Label
from kivy.uix.image import Image
from kivy.uix.popup import Popup
from kivy.graphics.texture import Texture
from kivy.clock import Clock
from kivy.core.window import Window
from kivy.properties import BooleanProperty, NumericProperty, StringProperty
from kivy.logger import Logger

# 导入美颜模块
try:
    from beautifier import FaceBeautifier
except ImportError as e:
    Logger.error(f"无法导入美颜模块: {e}")
    FaceBeautifier = None


class CameraPreview(Image):
    """
    相机预览组件
    继承自Kivy Image，用于显示实时视频流
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.capture = None
        self.beautifier = None
        self.beauty_enabled = True
        self.fps = 0
        self.frame_count = 0
        self.last_time = time.time()
        
        # 美颜参数
        self.skin_smooth = 0.6
        self.skin_whiten = 0.3
        self.eye_enlarge = 1.15
        self.face_slim = 0.03
        
        # 初始化相机
        self.init_camera()
        
        # 初始化美颜处理器
        if FaceBeautifier is not None:
            try:
                self.beautifier = FaceBeautifier()
                self.update_beauty_params()
                Logger.info("美颜处理器初始化成功")
            except Exception as e:
                Logger.error(f"美颜处理器初始化失败: {e}")
                self.beautifier = None
        
        # 启动更新循环（30fps）
        Clock.schedule_interval(self.update, 1.0 / 30.0)
        
    def init_camera(self):
        """初始化摄像头"""
        try:
            # 尝试打开默认摄像头
            self.capture = cv2.VideoCapture(0)
            
            # 设置分辨率（移动端常用分辨率）
            self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            
            # 设置帧率
            self.capture.set(cv2.CAP_PROP_FPS, 30)
            
            if self.capture.isOpened():
                Logger.info("摄像头初始化成功")
            else:
                Logger.error("无法打开摄像头")
                self.capture = None
        except Exception as e:
            Logger.error(f"摄像头初始化失败: {e}")
            self.capture = None
            
    def update_beauty_params(self):
        """更新美颜参数"""
        if self.beautifier:
            self.beautifier.set_params(
                skin_smooth_strength=self.skin_smooth,
                skin_whiten_strength=self.skin_whiten,
                eye_enlarge_scale=self.eye_enlarge,
                face_slim_strength=self.face_slim,
                lip_enhance=True
            )
            
    def update(self, dt):
        """更新帧（由Clock调度）"""
        if self.capture is None:
            return
            
        ret, frame = self.capture.read()
        if not ret or frame is None:
            return
            
        # 水平翻转（镜像效果，自拍更自然）
        frame = cv2.flip(frame, 1)
        
        # 计算FPS
        self.frame_count += 1
        current_time = time.time()
        if current_time - self.last_time >= 1.0:
            self.fps = self.frame_count
            self.frame_count = 0
            self.last_time = current_time
            
        # 应用美颜
        if self.beauty_enabled and self.beautifier is not None:
            try:
                frame = self.beautifier.process_frame(frame)
            except Exception as e:
                Logger.error(f"美颜处理错误: {e}")
                
        # 转换为Kivy纹理
        # BGR -> RGB
        buf = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 创建纹理
        texture = Texture.create(
            size=(frame.shape[1], frame.shape[0]),
            colorfmt='rgb'
        )
        texture.blit_buffer(buf.tobytes(), colorfmt='rgb', bufferfmt='ubyte')
        texture.flip_vertical()
        
        self.texture = texture
        
    def capture_photo(self):
        """拍照"""
        if self.capture is None:
            return None
            
        ret, frame = self.capture.read()
        if ret:
            frame = cv2.flip(frame, 1)
            
            # 如果开启了美颜，应用美颜
            if self.beauty_enabled and self.beautifier is not None:
                try:
                    frame = self.beautifier.process_frame(frame)
                except Exception as e:
                    Logger.error(f"拍照美颜处理错误: {e}")
                    
            return frame
        return None
        
    def on_stop(self):
        """释放资源"""
        if self.capture:
            self.capture.release()
            self.capture = None
            

class BeautyCameraApp(App):
    """
    美颜相机应用主类
    """
    
    def build(self):
        """构建应用界面"""
        # 设置窗口背景色
        Window.clearcolor = (0.1, 0.1, 0.1, 1)
        
        # 创建根布局
        root = FloatLayout()
        
        # 相机预览
        self.preview = CameraPreview()
        self.preview.size_hint = (1, 0.85)
        self.preview.pos_hint = {'x': 0, 'top': 1}
        root.add_widget(self.preview)
        
        # 控制面板
        controls = BoxLayout(
            orientation='vertical',
            size_hint=(1, 0.35),
            pos_hint={'x': 0, 'y': 0},
            padding=10,
            spacing=5
        )
        
        # 美颜开关按钮
        btn_layout = BoxLayout(size_hint_y=None, height=50, spacing=10)
        
        self.beauty_btn = Button(
            text='美颜: 开',
            background_color=(0.2, 0.8, 0.2, 1),
            font_size='16sp'
        )
        self.beauty_btn.bind(on_press=self.toggle_beauty)
        btn_layout.add_widget(self.beauty_btn)
        
        # 拍照按钮
        capture_btn = Button(
            text='📷 拍照',
            background_color=(0.9, 0.2, 0.2, 1),
            font_size='18sp',
            bold=True
        )
        capture_btn.bind(on_press=self.capture_photo)
        btn_layout.add_widget(capture_btn)
        
        # 切换摄像头按钮
        switch_btn = Button(
            text='切换摄像头',
            font_size='14sp'
        )
        switch_btn.bind(on_press=self.switch_camera)
        btn_layout.add_widget(switch_btn)
        
        controls.add_widget(btn_layout)
        
        # 美颜参数滑块
        # 磨皮
        smooth_layout = BoxLayout(size_hint_y=None, height=40)
        smooth_layout.add_widget(Label(text='磨皮:', size_hint_x=None, width=60))
        self.smooth_slider = Slider(
            min=0, max=1.0, value=0.6,
            value_track=True,
            value_track_color=[0.2, 0.8, 0.2, 1]
        )
        self.smooth_slider.bind(value=self.on_smooth_change)
        smooth_layout.add_widget(self.smooth_slider)
        self.smooth_label = Label(text='60%', size_hint_x=None, width=50)
        smooth_layout.add_widget(self.smooth_label)
        controls.add_widget(smooth_layout)
        
        # 美白
        whiten_layout = BoxLayout(size_hint_y=None, height=40)
        whiten_layout.add_widget(Label(text='美白:', size_hint_x=None, width=60))
        self.whiten_slider = Slider(
            min=0, max=1.0, value=0.3,
            value_track=True,
            value_track_color=[0.2, 0.6, 0.9, 1]
        )
        self.whiten_slider.bind(value=self.on_whiten_change)
        whiten_layout.add_widget(self.whiten_slider)
        self.whiten_label = Label(text='30%', size_hint_x=None, width=50)
        whiten_layout.add_widget(self.whiten_label)
        controls.add_widget(whiten_layout)
        
        # 大眼
        eye_layout = BoxLayout(size_hint_y=None, height=40)
        eye_layout.add_widget(Label(text='大眼:', size_hint_x=None, width=60))
        self.eye_slider = Slider(
            min=1.0, max=1.5, value=1.15,
            value_track=True,
            value_track_color=[0.9, 0.5, 0.2, 1]
        )
        self.eye_slider.bind(value=self.on_eye_change)
        eye_layout.add_widget(self.eye_slider)
        self.eye_label = Label(text='1.15x', size_hint_x=None, width=50)
        eye_layout.add_widget(self.eye_label)
        controls.add_widget(eye_layout)
        
        # 瘦脸
        slim_layout = BoxLayout(size_hint_y=None, height=40)
        slim_layout.add_widget(Label(text='瘦脸:', size_hint_x=None, width=60))
        self.slim_slider = Slider(
            min=0, max=0.1, value=0.03,
            value_track=True,
            value_track_color=[0.8, 0.2, 0.8, 1]
        )
        self.slim_slider.bind(value=self.on_slim_change)
        slim_layout.add_widget(self.slim_slider)
        self.slim_label = Label(text='3%', size_hint_x=None, width=50)
        slim_layout.add_widget(self.slim_label)
        controls.add_widget(slim_layout)
        
        # FPS显示
        self.fps_label = Label(
            text='FPS: --',
            size_hint_y=None,
            height=30,
            font_size='12sp'
        )
        controls.add_widget(self.fps_label)
        
        root.add_widget(controls)
        
        # 启动FPS更新
        Clock.schedule_interval(self.update_fps, 0.5)
        
        return root
        
    def toggle_beauty(self, instance):
        """切换美颜开关"""
        self.preview.beauty_enabled = not self.preview.beauty_enabled
        if self.preview.beauty_enabled:
            instance.text = '美颜: 开'
            instance.background_color = (0.2, 0.8, 0.2, 1)
        else:
            instance.text = '美颜: 关'
            instance.background_color = (0.5, 0.5, 0.5, 1)
            
    def capture_photo(self, instance):
        """拍照并保存"""
        frame = self.preview.capture_photo()
        if frame is not None:
            # 生成文件名
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"BeautyPhoto_{timestamp}.jpg"
            
            # 保存到相册目录
            # Android: /storage/emulated/0/DCIM/BeautyCamera/
            # iOS: 相册
            # 桌面: 当前目录
            save_dir = self.get_save_directory()
            os.makedirs(save_dir, exist_ok=True)
            
            filepath = os.path.join(save_dir, filename)
            cv2.imwrite(filepath, frame)
            
            Logger.info(f"照片已保存: {filepath}")
            self.show_popup(f"照片已保存!\n{filename}")
        else:
            self.show_popup("拍照失败，请重试")
            
    def get_save_directory(self):
        """获取保存目录"""
        # 尝试获取移动端存储路径
        try:
            from android.storage import primary_external_storage_path
            from android.permissions import request_permissions, Permission
            
            # 请求存储权限
            request_permissions([
                Permission.WRITE_EXTERNAL_STORAGE,
                Permission.READ_EXTERNAL_STORAGE
            ])
            
            base_path = primary_external_storage_path()
            return os.path.join(base_path, 'DCIM', 'BeautyCamera')
        except ImportError:
            # 桌面环境
            return os.path.join(os.path.expanduser('~'), 'Pictures', 'BeautyCamera')
            
    def switch_camera(self, instance):
        """切换前后摄像头"""
        # 释放当前摄像头
        if self.preview.capture:
            self.preview.capture.release()
            
        # 切换摄像头索引
        current = 0
        if hasattr(self.preview, 'camera_index'):
            current = self.preview.camera_index
            
        new_index = 1 if current == 0 else 0
        
        # 尝试打开新摄像头
        new_capture = cv2.VideoCapture(new_index)
        if new_capture.isOpened():
            self.preview.capture = new_capture
            self.preview.camera_index = new_index
            self.preview.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.preview.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            Logger.info(f"切换到摄像头 {new_index}")
        else:
            # 切换失败，恢复原摄像头
            self.preview.capture = cv2.VideoCapture(current)
            self.show_popup("无法切换摄像头")
            
    def on_smooth_change(self, instance, value):
        """磨皮强度变化"""
        self.preview.skin_smooth = value
        self.preview.update_beauty_params()
        self.smooth_label.text = f"{int(value*100)}%"
        
    def on_whiten_change(self, instance, value):
        """美白强度变化"""
        self.preview.skin_whiten = value
        self.preview.update_beauty_params()
        self.whiten_label.text = f"{int(value*100)}%"
        
    def on_eye_change(self, instance, value):
        """大眼比例变化"""
        self.preview.eye_enlarge = value
        self.preview.update_beauty_params()
        self.eye_label.text = f"{value:.2f}x"
        
    def on_slim_change(self, instance, value):
        """瘦脸强度变化"""
        self.preview.face_slim = value
        self.preview.update_beauty_params()
        self.slim_label.text = f"{int(value*100)}%"
        
    def update_fps(self, dt):
        """更新FPS显示"""
        self.fps_label.text = f"FPS: {self.preview.fps}"
        
    def show_popup(self, message):
        """显示弹出消息"""
        popup = Popup(
            title='提示',
            content=Label(text=message),
            size_hint=(None, None),
            size=(300, 150),
            auto_dismiss=True
        )
        popup.open()
        
    def on_stop(self):
        """应用退出时释放资源"""
        self.preview.on_stop()
        

if __name__ == '__main__':
    BeautyCameraApp().run()
