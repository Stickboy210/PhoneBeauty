"""
实时人脸美颜处理模块
基于OpenCV和传统数字图像处理方法
针对移动端实时处理进行优化
"""

import cv2
import numpy as np
import dlib
import os


class FaceBeautifier:
    """
    实时人脸美颜处理器
    
    功能：
    - 人脸检测与68点特征点定位
    - 皮肤磨皮（快速NLM降噪）
    - 皮肤美白（亮度提升）
    - 眼睛放大
    - 瘦脸
    - 嘴唇增强
    """
    
    def __init__(self, predictor_path="shape_predictor_68_face_landmarks.dat"):
        """
        初始化美颜处理器
        
        Args:
            predictor_path: dlib 68点特征点检测器模型路径
        """
        # 初始化人脸检测器（Haar级联分类器 - 速度较快）
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        # 初始化dlib的68点特征检测器
        self.detector = dlib.get_frontal_face_detector()
        
        # 检查模型文件是否存在
        if not os.path.exists(predictor_path):
            raise FileNotFoundError(f"找不到特征点模型文件: {predictor_path}")
        self.predictor = dlib.shape_predictor(predictor_path)
        
        # 美颜参数（可调节）
        self.params = {
            'skin_smooth_strength': 0.6,      # 磨皮强度 0.0-1.0
            'skin_whiten_strength': 0.3,      # 美白强度 0.0-1.0
            'eye_enlarge_scale': 1.15,        # 大眼比例 1.0-1.5
            'face_slim_strength': 0.03,       # 瘦脸强度 0.0-0.1
            'lip_enhance': True,              # 是否启用嘴唇增强
        }
        
        # 缓存检测结果（用于提高帧率）
        self.last_faces = None
        self.last_landmarks = None
        self.frame_count = 0
        self.detect_interval = 3  # 每3帧检测一次人脸
        
    def set_params(self, **kwargs):
        """
        设置美颜参数
        
        Args:
            skin_smooth_strength: 磨皮强度 (0.0-1.0)
            skin_whiten_strength: 美白强度 (0.0-1.0)
            eye_enlarge_scale: 大眼比例 (1.0-1.5)
            face_slim_strength: 瘦脸强度 (0.0-0.1)
            lip_enhance: 是否启用嘴唇增强 (bool)
        """
        for key, value in kwargs.items():
            if key in self.params:
                self.params[key] = value
                
    def process_frame(self, img, return_debug_info=False):
        """
        处理单帧图像（实时美颜）
        
        Args:
            img: 输入图像 (BGR格式)
            return_debug_info: 是否返回调试图像
            
        Returns:
            处理后的图像
            如果return_debug_info为True，还返回包含特征点的调试图像
        """
        if img is None or img.size == 0:
            return img if not return_debug_info else (img, None)
            
        # 限制处理尺寸以提高速度
        h, w = img.shape[:2]
        max_size = 640  # 移动端最大处理尺寸
        if max(h, w) > max_size:
            scale = max_size / max(h, w)
            img_small = cv2.resize(img, None, fx=scale, fy=scale)
            need_resize = True
        else:
            img_small = img
            scale = 1.0
            need_resize = False
            
        result = img_small.copy()
        debug_img = img_small.copy() if return_debug_info else None
        
        # 隔帧检测人脸（提高帧率）
        self.frame_count += 1
        if self.frame_count % self.detect_interval == 0 or self.last_faces is None:
            faces = self.detect_faces(img_small)
            landmarks = self.detect_landmarks(img_small, faces)
            if len(faces) > 0:
                self.last_faces = faces
                self.last_landmarks = landmarks
        else:
            faces = self.last_faces
            landmarks = self.last_landmarks
            
        # 绘制人脸框和特征点（调试用）
        if return_debug_info and faces is not None:
            for (x, y, fw, fh) in faces:
                cv2.rectangle(debug_img, (x, y), (x + fw, y + fh), (0, 255, 0), 2)
                
        if len(landmarks) > 0:
            # 获取第一个人脸的特征点和轮廓
            face_landmarks = landmarks[0]
            points = face_landmarks[0].astype(np.int32)
            
            if return_debug_info:
                for (x, y) in points:
                    cv2.circle(debug_img, (x, y), 1, (0, 0, 255), -1)
                    
            # 计算脸部轮廓
            contour = self._get_face_contour(points)
            
            # 美颜处理流程（按顺序应用）
            if contour is not None:
                # 1. 皮肤磨皮
                if self.params['skin_smooth_strength'] > 0:
                    result = self._skin_smoothing_fast(
                        result, points, contour, 
                        self.params['skin_smooth_strength']
                    )
                    
                # 2. 皮肤美白
                if self.params['skin_whiten_strength'] > 0:
                    result = self._skin_whitening_fast(
                        result, points, contour,
                        self.params['skin_whiten_strength']
                    )
                    
                # 3. 眼睛放大
                if self.params['eye_enlarge_scale'] > 1.0:
                    result = self._eye_enlargement_fast(
                        result, points,
                        self.params['eye_enlarge_scale']
                    )
                    
                # 4. 瘦脸
                if self.params['face_slim_strength'] > 0:
                    result = self._face_slimming_fast(
                        result, points,
                        self.params['face_slim_strength']
                    )
                    
                # 5. 嘴唇增强
                if self.params['lip_enhance']:
                    result = self._lip_enhancement_fast(result, points)
                    
        # 如果进行了缩放，恢复原始尺寸
        if need_resize:
            result = cv2.resize(result, (w, h))
            if debug_img is not None:
                debug_img = cv2.resize(debug_img, (w, h))
                
        if return_debug_info:
            return result, debug_img
        return result
        
    def detect_faces(self, img):
        """检测人脸位置"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=5, 
            minSize=(60, 60),
            maxSize=(img.shape[1], img.shape[0])
        )
        return faces
        
    def detect_landmarks(self, img, faces):
        """检测68点人脸特征点"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        landmarks = []
        for (x, y, w, h) in faces:
            dlib_rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))
            shape = self.predictor(gray, dlib_rect)
            points = np.array([[p.x, p.y] for p in shape.parts()], dtype=np.int32)
            landmarks.append([points])
        return landmarks
        
    def _get_face_contour(self, points):
        """获取脸部轮廓（使用椭圆拟合优化）"""
        # 使用下巴点 (0-16) 进行椭圆拟合
        jaw_points = points[0:17].astype(np.float32)
        if len(jaw_points) < 5:
            return None
            
        try:
            ellipse = cv2.fitEllipse(jaw_points)
            center, axes, angle = ellipse
            
            # 生成椭圆轮廓点
            theta = np.linspace(0, 2 * np.pi, 50)
            a, b = axes[0] / 2, axes[1] / 2
            x = center[0] + a * np.cos(theta) * np.cos(np.deg2rad(angle)) - b * np.sin(theta) * np.sin(np.deg2rad(angle))
            y = center[1] + a * np.cos(theta) * np.sin(np.deg2rad(angle)) + b * np.sin(theta) * np.cos(np.deg2rad(angle))
            contour = np.vstack((x, y)).T.astype(np.int32)
            return contour
        except:
            return jaw_points.astype(np.int32)
            
    def _skin_smoothing_fast(self, img, points, contour, strength):
        """快速皮肤磨皮"""
        # 使用双边滤波 + 高斯模糊的组合
        # 根据强度调整参数
        d = int(5 + strength * 10)  # 滤波直径
        sigma_color = int(50 + strength * 50)
        sigma_space = int(25 + strength * 25)
        
        # 双边滤波（保边去噪）
        smoothed = cv2.bilateralFilter(img, d, sigma_color, sigma_space)
        
        # 额外的高斯平滑
        if strength > 0.5:
            ksize = int(5 + (strength - 0.5) * 10)
            if ksize % 2 == 0:
                ksize += 1
            smoothed = cv2.GaussianBlur(smoothed, (ksize, ksize), 0)
            
        # 创建脸部掩码
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)
        
        # 排除眼睛、眉毛、嘴巴区域
        left_combo = np.vstack((points[17:22], points[39:41], points[36].reshape(1, 2)))
        right_combo = np.vstack((points[22:27], points[45:47], points[42].reshape(1, 2)))
        mouth = points[48:60]
        
        cv2.fillPoly(mask, [left_combo.reshape((-1, 1, 2))], 0)
        cv2.fillPoly(mask, [right_combo.reshape((-1, 1, 2))], 0)
        cv2.fillPoly(mask, [mouth.reshape((-1, 1, 2))], 0)
        
        # 对掩码进行高斯模糊，使边缘过渡自然
        mask = cv2.GaussianBlur(mask, (21, 21), 0)
        mask = mask.astype(np.float32) / 255.0
        mask = np.expand_dims(mask, axis=2)
        
        # 混合原图和磨皮图
        result = (img * (1 - mask) + smoothed * mask).astype(np.uint8)
        return result
        
    def _skin_whitening_fast(self, img, points, contour, strength):
        """快速皮肤美白"""
        # 创建脸部掩码
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)
        
        # 排除眼睛、眉毛、嘴巴
        left_combo = np.vstack((points[17:22], points[39:41], points[36].reshape(1, 2)))
        right_combo = np.vstack((points[22:27], points[45:47], points[42].reshape(1, 2)))
        mouth = points[48:60]
        
        cv2.fillPoly(mask, [left_combo.reshape((-1, 1, 2))], 0)
        cv2.fillPoly(mask, [right_combo.reshape((-1, 1, 2))], 0)
        cv2.fillPoly(mask, [mouth.reshape((-1, 1, 2))], 0)
        
        # 转换为LAB颜色空间，提升L通道亮度
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # 对L通道进行亮度提升（使用gamma校正）
        gamma = 1.0 - strength * 0.3  # gamma < 1 提升亮度
        l_float = l.astype(np.float32) / 255.0
        l_enhanced = np.power(l_float, gamma) * 255
        l_enhanced = np.clip(l_enhanced, 0, 255).astype(np.uint8)
        
        # 混合原亮度和美白后的亮度
        l_result = (l * (1 - strength) + l_enhanced * strength).astype(np.uint8)
        
        # 合并通道
        lab_result = cv2.merge([l_result, a, b])
        result = cv2.cvtColor(lab_result, cv2.COLOR_LAB2BGR)
        
        # 只在掩码区域应用美白
        mask = mask.astype(np.float32) / 255.0
        mask = np.expand_dims(mask, axis=2)
        result = (img * (1 - mask) + result * mask).astype(np.uint8)
        
        return result
        
    def _eye_enlargement_fast(self, img, points, scale):
        """快速眼睛放大（基于局部缩放）"""
        if scale <= 1.0:
            return img
            
        result = img.copy()
        h, w = img.shape[:2]
        
        # 左右眼中心
        left_eye_center = np.mean(points[36:42], axis=0).astype(np.int32)
        right_eye_center = np.mean(points[42:48], axis=0).astype(np.int32)
        
        # 计算眼宽
        left_eye_width = np.linalg.norm(points[36] - points[39])
        right_eye_width = np.linalg.norm(points[42] - points[45])
        
        # 影响半径
        radius = int(min(left_eye_width, right_eye_width) * 0.8)
        
        # 对左右眼分别放大
        result = self._local_zoom(result, left_eye_center, radius, scale)
        result = self._local_zoom(result, right_eye_center, radius, scale)
        
        return result
        
    def _local_zoom(self, img, center, radius, scale):
        """局部缩放变形"""
        h, w = img.shape[:2]
        
        # 创建映射表
        map_x = np.zeros((h, w), dtype=np.float32)
        map_y = np.zeros((h, w), dtype=np.float32)
        
        cx, cy = center
        
        for y in range(h):
            for x in range(w):
                dx = x - cx
                dy = y - cy
                dist = np.sqrt(dx*dx + dy*dy)
                
                if dist < radius and dist > 0:
                    # 距离越近，缩放越大
                    factor = 1.0 - (dist / radius) * (scale - 1.0) / scale
                    new_x = cx + dx / factor
                    new_y = cy + dy / factor
                    map_x[y, x] = max(0, min(w-1, new_x))
                    map_y[y, x] = max(0, min(h-1, new_y))
                else:
                    map_x[y, x] = x
                    map_y[y, x] = y
                    
        result = cv2.remap(img, map_x, map_y, 
                          interpolation=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REPLICATE)
        return result
        
    def _face_slimming_fast(self, img, points, strength):
        """快速瘦脸（基于局部移动）"""
        if strength <= 0:
            return img
            
        h, w = img.shape[:2]
        
        # 计算脸部中心
        face_center = np.mean(points[27:36], axis=0)  # 鼻子区域中心
        
        # 脸颊关键点
        cheek_indices = list(range(1, 16))  # 排除下巴尖端
        cheek_points = points[cheek_indices].astype(np.float32)
        
        # 创建控制点
        src_points = cheek_points.copy()
        dst_points = cheek_points.copy()
        
        # 向脸部中心移动脸颊点
        for i, idx in enumerate(cheek_indices):
            direction = face_center - points[idx]
            dst_points[i] = points[idx] + direction * strength
            
        # 添加固定点（额头、眼睛、鼻子、嘴巴）
        fixed_indices = list(range(17, 68))
        src_points = np.vstack([src_points, points[fixed_indices].astype(np.float32)])
        dst_points = np.vstack([dst_points, points[fixed_indices].astype(np.float32)])
        
        # 使用移动最小二乘法(MLS)或简化版RBF
        return self._simple_warp(img, src_points, dst_points)
        
    def _simple_warp(self, img, src_points, dst_points):
        """简化版图像变形（基于稀疏控制点）"""
        h, w = img.shape[:2]
        
        # 创建网格
        grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
        grid = np.stack([grid_x, grid_y], axis=-1).astype(np.float32)
        
        # 计算变形（使用径向基函数）
        n = len(src_points)
        if n == 0:
            return img
            
        displacement = dst_points - src_points
        
        # 使用简化版RBF（只考虑最近的几个控制点）
        grid_flat = grid.reshape(-1, 2)
        
        # 计算到所有控制点的距离
        dists = np.linalg.norm(
            src_points[:, np.newaxis, :] - grid_flat[np.newaxis, :, :],
            axis=2
        )  # shape: (n_points, n_pixels)
        
        # RBF核（高斯函数）
        sigma = max(w, h) * 0.1
        weights = np.exp(-(dists**2) / (2 * sigma**2))
        
        # 归一化权重
        weights = weights / (weights.sum(axis=0) + 1e-8)
        
        # 计算变形
        deformation = weights.T @ displacement  # shape: (n_pixels, 2)
        
        # 应用变形
        new_grid = grid_flat + deformation
        new_grid = new_grid.reshape(h, w, 2)
        
        # 分离坐标
        map_x = new_grid[:, :, 0].clip(0, w-1).astype(np.float32)
        map_y = new_grid[:, :, 1].clip(0, h-1).astype(np.float32)
        
        result = cv2.remap(img, map_x, map_y,
                          interpolation=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REPLICATE)
        return result
        
    def _lip_enhancement_fast(self, img, points):
        """快速嘴唇增强"""
        # 嘴唇区域
        lip_points = points[48:60]
        
        # 创建嘴唇凸包
        lip_hull = cv2.convexHull(lip_points)
        
        # 创建掩码
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [lip_hull], -1, 255, thickness=cv2.FILLED)
        
        # 模糊掩码边缘
        mask = cv2.GaussianBlur(mask, (11, 11), 0)
        mask = mask.astype(np.float32) / 255.0
        mask = np.expand_dims(mask, axis=2)
        
        # 轻微提升饱和度
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        s = np.clip(s * 1.15, 0, 255).astype(np.uint8)
        hsv_enhanced = cv2.merge([h, s, v])
        bgr_enhanced = cv2.cvtColor(hsv_enhanced, cv2.COLOR_HSV2BGR)
        
        # 轻微提升红色通道
        b, g, r = cv2.split(bgr_enhanced)
        r = np.clip(r * 1.1, 0, 255).astype(np.uint8)
        result = cv2.merge([b, g, r])
        
        # 混合
        final = (img * (1 - mask) + result * mask).astype(np.uint8)
        return final
