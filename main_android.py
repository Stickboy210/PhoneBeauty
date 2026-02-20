"""
美颜相机 - Android简化版
使用轻量级美颜处理器，不依赖dlib

功能：
- 实时摄像头预览
- 实时美颜处理（基于肤色检测的磨皮美白）
- 美颜参数调节
- 拍照保存
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
from kivy.logger import Logger

# 导入轻量级美颜模块
try:
    from beautifier_light import FaceBeautifierLight
except ImportError as e:
    Logger.error(f"无法导入美颜模块: {e}")
    FaceBeautifierLight = None


class CameraPreview(Image):
    """相机预览组件"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.capture = None
        self.beautifier = None
        self.beauty_enabled = True
        self.fps = 0
        self.frame_count = 0
        self.last_time = time.time()
        
        # 美颜参数
        self.skin_smooth = 0.5
        self.skin_whiten = 0.3
        self.brightness = 1.0
        self.contrast = 1.0
        
        # 初始化相机
        self.init_camera()
        
        # 初始化美颜处理器
        if FaceBeautifierLight is not None:
            try:
                self.beautifier = FaceBeautifierLight()
                self.update_beauty_params()
                Logger.info("美颜处理器初始化成功")
            except Exception as e:
                Logger.error(f"美颜处理器初始化失败: {e}")
                self.beautifier = None
        
        # 启动更新循环（目标25fps）
        Clock.schedule_interval(self.update, 1.0 / 25.0)
        
    def init_camera(self):
        """初始化摄像头"""
        try:
            self.capture = cv2.VideoCapture(0)
            # 移动端常用分辨率
            self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
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
                brightness=self.brightness,
                contrast=self.contrast
            )
            
    def update(self, dt):
        """更新帧"""
        if self.capture is None:
            return
            
        ret, frame = self.capture.read()
        if not ret or frame is None:
            return
            
        # 水平翻转（镜像效果）
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
        buf = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
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


class BeautyCameraAndroidApp(App):
    """Android美颜相机应用"""
    
    def build(self):
        Window.clearcolor = (0.1, 0.1, 0.1, 1)
        
        root = FloatLayout()
        
        # 相机预览
        self.preview = CameraPreview()
        self.preview.size_hint = (1, 0.8)
        self.preview.pos_hint = {'x': 0, 'top': 1}
        root.add_widget(self.preview)
        
        # 控制面板
        controls = BoxLayout(
            orientation='vertical',
            size_hint=(1, 0.25),
            pos_hint={'x': 0, 'y': 0},
            padding=5,
            spacing=3
        )
        
        # 按钮行
        btn_layout = BoxLayout(size_hint_y=None, height=50, spacing=5)
        
        self.beauty_btn = Button(
            text='美颜: 开',
            background_color=(0.2, 0.8, 0.2, 1),
            font_size='14sp'
        )
        self.beauty_btn.bind(on_press=self.toggle_beauty)
        btn_layout.add_widget(self.beauty_btn)
        
        capture_btn = Button(
            text='📷 拍照',
            background_color=(0.9, 0.2, 0.2, 1),
            font_size='16sp',
            bold=True
        )
        capture_btn.bind(on_press=self.capture_photo)
        btn_layout.add_widget(capture_btn)
        
        controls.add_widget(btn_layout)
        
        # 磨皮滑块
        smooth_layout = BoxLayout(size_hint_y=None, height=35)
        smooth_layout.add_widget(Label(text='磨皮:', size_hint_x=None, width=50, font_size='12sp'))
        self.smooth_slider = Slider(min=0, max=1.0, value=0.5)
        self.smooth_slider.bind(value=self.on_smooth_change)
        smooth_layout.add_widget(self.smooth_slider)
        self.smooth_label = Label(text='50%', size_hint_x=None, width=40, font_size='11sp')
        smooth_layout.add_widget(self.smooth_label)
        controls.add_widget(smooth_layout)
        
        # 美白滑块
        whiten_layout = BoxLayout(size_hint_y=None, height=35)
        whiten_layout.add_widget(Label(text='美白:', size_hint_x=None, width=50, font_size='12sp'))
        self.whiten_slider = Slider(min=0, max=1.0, value=0.3)
        self.whiten_slider.bind(value=self.on_whiten_change)
        whiten_layout.add_widget(self.whiten_slider)
        self.whiten_label = Label(text='30%', size_hint_x=None, width=40, font_size='11sp')
        whiten_layout.add_widget(self.whiten_label)
        controls.add_widget(whiten_layout)
        
        # 亮度滑块
        bright_layout = BoxLayout(size_hint_y=None, height=35)
        bright_layout.add_widget(Label(text='亮度:', size_hint_x=None, width=50, font_size='12sp'))
        self.bright_slider = Slider(min=0.5, max=1.5, value=1.0)
        self.bright_slider.bind(value=self.on_bright_change)
        bright_layout.add_widget(self.bright_slider)
        self.bright_label = Label(text='1.0', size_hint_x=None, width=40, font_size='11sp')
        bright_layout.add_widget(self.bright_label)
        controls.add_widget(bright_layout)
        
        # FPS显示
        self.fps_label = Label(
            text='FPS: --',
            size_hint_y=None,
            height=25,
            font_size='11sp'
        )
        controls.add_widget(self.fps_label)
        
        root.add_widget(controls)
        
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
        """拍照"""
        frame = self.preview.capture_photo()
        if frame is not None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"Beauty_{timestamp}.jpg"
            
            save_dir = self.get_save_directory()
            os.makedirs(save_dir, exist_ok=True)
            
            filepath = os.path.join(save_dir, filename)
            cv2.imwrite(filepath, frame)
            
            Logger.info(f"照片已保存: {filepath}")
            self.show_popup(f"照片已保存!\n{filename}")
        else:
            self.show_popup("拍照失败")
            
    def get_save_directory(self):
        """获取保存目录"""
        try:
            from android.storage import primary_external_storage_path
            from android.permissions import request_permissions, Permission
            
            request_permissions([
                Permission.WRITE_EXTERNAL_STORAGE,
                Permission.READ_EXTERNAL_STORAGE
            ])
            
            base_path = primary_external_storage_path()
            return os.path.join(base_path, 'DCIM', 'BeautyCamera')
        except ImportError:
            return os.path.join(os.path.expanduser('~'), 'Pictures', 'BeautyCamera')
            
    def on_smooth_change(self, instance, value):
        self.preview.skin_smooth = value
        self.preview.update_beauty_params()
        self.smooth_label.text = f"{int(value*100)}%"
        
    def on_whiten_change(self, instance, value):
        self.preview.skin_whiten = value
        self.preview.update_beauty_params()
        self.whiten_label.text = f"{int(value*100)}%"
        
    def on_bright_change(self, instance, value):
        self.preview.brightness = value
        self.preview.update_beauty_params()
        self.bright_label.text = f"{value:.2f}"
        
    def update_fps(self, dt):
        self.fps_label.text = f"FPS: {self.preview.fps}"
        
    def show_popup(self, message):
        popup = Popup(
            title='提示',
            content=Label(text=message),
            size_hint=(None, None),
            size=(250, 120),
            auto_dismiss=True
        )
        popup.open()
        
    def on_stop(self):
        self.preview.on_stop()


if __name__ == '__main__':
    BeautyCameraAndroidApp().run()
