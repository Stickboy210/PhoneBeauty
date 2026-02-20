"""
轻量级美颜处理模块（Android适用）
不依赖dlib，使用OpenCV DNN进行人脸检测

注意: 这是一个简化版本，功能较完整版有所减少：
- 使用OpenCV DNN代替dlib进行人脸检测
- 不使用68点特征点，基于人脸区域进行近似处理
- 适合移动端部署
"""

import cv2
import numpy as np
import os


class FaceBeautifierLight:
    """
    轻量级实时人脸美颜处理器
    适用于移动端，不依赖dlib
    """
    
    def __init__(self, face_detector_path=None):
        """
        初始化美颜处理器
        
        Args:
            face_detector_path: OpenCV DNN人脸检测模型路径（可选）
                               如果不提供，使用Haar级联分类器
        """
        # 尝试加载DNN人脸检测器
        self.dnn_detector = None
        self.use_dnn = False
        
        if face_detector_path and os.path.exists(face_detector_path):
            try:
                self.dnn_detector = cv2.dnn.readNetFromCaffe(
                    face_detector_path + "/deploy.prototxt",
                    face_detector_path + "/res10_300x300_ssd_iter_140000.caffemodel"
                )
                self.use_dnn = True
            except:
                pass
                
        # 备用：Haar级联分类器
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        # 美颜参数
        self.params = {
            'skin_smooth_strength': 0.5,
            'skin_whiten_strength': 0.3,
            'brightness': 1.0,
            'contrast': 1.0,
        }
        
        # 缓存
        self.last_face = None
        self.frame_count = 0
        self.detect_interval = 3
        
    def set_params(self, **kwargs):
        """设置美颜参数"""
        for key, value in kwargs.items():
            if key in self.params:
                self.params[key] = value
                
    def process_frame(self, img):
        """处理单帧图像"""
        if img is None or img.size == 0:
            return img
            
        # 限制处理尺寸
        h, w = img.shape[:2]
        max_size = 480  # 移动端使用更小尺寸
        if max(h, w) > max_size:
            scale = max_size / max(h, w)
            img_small = cv2.resize(img, None, fx=scale, fy=scale)
            need_resize = True
        else:
            img_small = img
            scale = 1.0
            need_resize = False
            
        result = img_small.copy()
        
        # 隔帧检测
        self.frame_count += 1
        if self.frame_count % self.detect_interval == 0 or self.last_face is None:
            face = self.detect_face(img_small)
            if face is not None:
                self.last_face = face
        else:
            face = self.last_face
            
        if face is not None:
            x, y, fw, fh = face
            
            # 扩展人脸区域（包含颈部）
            y_ext = max(0, y - int(fh * 0.3))
            h_ext = min(img_small.shape[0] - y_ext, int(fh * 1.5))
            
            face_roi = (x, y_ext, fw, h_ext)
            
            # 应用美颜
            if self.params['skin_smooth_strength'] > 0:
                result = self._skin_smooth(result, face_roi)
                
            if self.params['skin_whiten_strength'] > 0:
                result = self._skin_whiten(result, face_roi)
                
        # 全局亮度和对比度调整
        if self.params['brightness'] != 1.0 or self.params['contrast'] != 1.0:
            result = self._adjust_brightness_contrast(
                result, 
                self.params['brightness'],
                self.params['contrast']
            )
            
        # 恢复尺寸
        if need_resize:
            result = cv2.resize(result, (w, h))
            
        return result
        
    def detect_face(self, img):
        """检测人脸"""
        if self.use_dnn and self.dnn_detector:
            return self._detect_face_dnn(img)
        else:
            return self._detect_face_haar(img)
            
    def _detect_face_dnn(self, img):
        """使用DNN检测人脸"""
        h, w = img.shape[:2]
        blob = cv2.dnn.blobFromImage(
            cv2.resize(img, (300, 300)), 1.0,
            (300, 300), (104.0, 177.0, 123.0)
        )
        self.dnn_detector.setInput(blob)
        detections = self.dnn_detector.forward()
        
        best_face = None
        best_confidence = 0
        
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5 and confidence > best_confidence:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                x, y, x2, y2 = box.astype(int)
                best_face = (x, y, x2 - x, y2 - y)
                best_confidence = confidence
                
        return best_face
        
    def _detect_face_haar(self, img):
        """使用Haar级联检测人脸"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(60, 60),
            maxSize=(img.shape[1], img.shape[0])
        )
        
        if len(faces) > 0:
            # 返回最大的人脸
            return max(faces, key=lambda f: f[2] * f[3])
        return None
        
    def _skin_smooth(self, img, face_roi):
        """皮肤磨皮"""
        x, y, w, h = face_roi
        
        # 提取ROI
        roi = img[y:y+h, x:x+w]
        
        # 双边滤波
        strength = self.params['skin_smooth_strength']
        d = int(5 + strength * 10)
        sigma_color = int(40 + strength * 60)
        sigma_space = int(20 + strength * 30)
        
        smoothed = cv2.bilateralFilter(roi, d, sigma_color, sigma_space)
        
        # 额外高斯模糊
        if strength > 0.5:
            ksize = int(5 + (strength - 0.5) * 8)
            if ksize % 2 == 0:
                ksize += 1
            smoothed = cv2.GaussianBlur(smoothed, (ksize, ksize), 0)
            
        # 创建皮肤掩码（基于颜色）
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        # 肤色范围
        lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        upper_skin = np.array([30, 150, 255], dtype=np.uint8)
        
        mask = cv2.inRange(hsv, lower_skin, upper_skin)
        
        # 形态学操作优化掩码
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # 模糊掩码边缘
        mask = cv2.GaussianBlur(mask, (15, 15), 0)
        mask = mask.astype(np.float32) / 255.0
        mask = np.expand_dims(mask, axis=2)
        
        # 混合
        blended = (roi * (1 - mask) + smoothed * mask).astype(np.uint8)
        
        result = img.copy()
        result[y:y+h, x:x+w] = blended
        return result
        
    def _skin_whiten(self, img, face_roi):
        """皮肤美白"""
        x, y, w, h = face_roi
        
        roi = img[y:y+h, x:x+w]
        
        # 转换为LAB
        lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Gamma校正提升亮度
        strength = self.params['skin_whiten_strength']
        gamma = 1.0 - strength * 0.3
        l_float = l.astype(np.float32) / 255.0
        l_enhanced = np.power(l_float, gamma) * 255
        l_enhanced = np.clip(l_enhanced, 0, 255).astype(np.uint8)
        
        # 混合
        l_result = (l * (1 - strength) + l_enhanced * strength).astype(np.uint8)
        
        # 合并
        lab_result = cv2.merge([l_result, a, b])
        whitened = cv2.cvtColor(lab_result, cv2.COLOR_LAB2BGR)
        
        # 创建掩码
        mask = np.zeros(roi.shape[:2], dtype=np.uint8)
        cv2.ellipse(
            mask,
            (w//2, h//2),
            (w//2 - 10, h//2 - 10),
            0, 0, 360,
            255, -1
        )
        mask = cv2.GaussianBlur(mask, (21, 21), 0)
        mask = mask.astype(np.float32) / 255.0
        mask = np.expand_dims(mask, axis=2)
        
        # 混合
        blended = (roi * (1 - mask) + whitened * mask).astype(np.uint8)
        
        result = img.copy()
        result[y:y+h, x:x+w] = blended
        return result
        
    def _adjust_brightness_contrast(self, img, brightness, contrast):
        """调整亮度和对比度"""
        # 对比度
        img = cv2.convertScaleAbs(img, alpha=contrast, beta=0)
        
        # 亮度
        if brightness > 1.0:
            beta = (brightness - 1.0) * 50
            img = cv2.convertScaleAbs(img, alpha=1.0, beta=beta)
        elif brightness < 1.0:
            img = cv2.convertScaleAbs(img, alpha=brightness, beta=0)
            
        return img
