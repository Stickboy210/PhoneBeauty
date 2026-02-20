# 美颜相机 - 移动端实时美颜应用

基于Python + Kivy + OpenCV开发的跨平台实时美颜相机应用。

## 功能特性

- **实时美颜预览**: 拍摄时实时看到美颜效果
- **人脸检测与特征点定位**: 使用dlib 68点人脸特征点检测
- **美颜功能**:
  - ✨ 智能磨皮（双边滤波 + 高斯平滑）
  - 🌟 皮肤美白（LAB色彩空间亮度提升）
  - 👁️ 眼睛放大（局部几何变形）
  - 😊 瘦脸（RBF径向基函数插值）
  - 💋 嘴唇增强（饱和度提升）
- **可调节参数**: 通过滑块实时调整各项美颜强度
- **拍照保存**: 一键保存美颜后的照片到相册

## 项目结构

```
Phoneapp/
├── main.py              # Kivy应用主程序
├── beautifier.py        # 美颜核心算法模块
├── beautycamera.kv      # UI样式定义
├── buildozer.spec       # Android打包配置
├── requirements.txt     # Python依赖
├── README.md           # 项目说明
└── shape_predictor_68_face_landmarks.dat  # dlib人脸特征点模型
```

## 环境要求

### 桌面端（Windows/macOS/Linux）
- Python 3.8+
- OpenCV 4.x
- Kivy 2.x
- NumPy
- dlib

### 移动端（Android）
- Android 5.0+ (API 21+)
- 摄像头权限
- 存储权限

## 安装与运行

### 1. 安装依赖

```bash
# 创建虚拟环境（推荐）
python -m venv venv

# 激活虚拟环境
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# 安装依赖
pip install -r requirements.txt
```

### 2. 下载人脸特征点模型

```bash
# 模型文件已包含在项目中
# shape_predictor_68_face_landmarks.dat (约97MB)
```

如果模型文件不存在，可以从以下地址下载：
- http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2

### 3. 运行应用（桌面端）

```bash
python main.py
```

## Android打包

### 方法1：使用Buildozer（推荐）

Buildozer可以自动化Android打包过程。

#### 安装Buildozer

```bash
# 安装buildozer
pip install buildozer

# Linux: 安装依赖
sudo apt-get install -y \
    git zip unzip openjdk-17-jdk python3-pip autoconf libtool pkg-config \
    zlib1g-dev libncurses5-dev libncursesw5-dev liblzma-dev \
    libffi-dev libssl-dev libsqlite3-dev
```

#### 配置并打包

```bash
# 初始化buildozer（如果还没有buildozer.spec）
buildozer init

# 调试模式打包
buildozer android debug

# 部署到连接的设备
buildozer android debug deploy run

# 发布版本打包
buildozer android release
```

**注意**: 打包OpenCV和dlib到Android需要额外的配置：

1. **OpenCV**: 需要使用`opencv-python`的Android预编译版本，或者使用python-for-android的recipe
2. **dlib**: 需要在python-for-android中添加dlib的recipe（比较复杂）

#### 简化版Android方案

考虑到dlib在Android上的编译复杂性，提供以下替代方案：

1. **使用预训练ONNX模型**: 使用轻量级的人脸检测模型（如YuNet）
2. **使用Android原生Camera2 + OpenCV SDK**: 将核心算法用Java/Kotlin重写
3. **使用Chaquopy**: 支持在Android上运行Python，自动处理依赖

### 方法2：使用Chaquopy（简化方案）

Chaquopy可以更容易地在Android上运行Python代码：

1. 在Android Studio中创建新项目
2. 添加Chaquopy插件到build.gradle
3. 在Python代码目录放入本项目文件
4. 在build.gradle中添加依赖：
   ```gradle
   python {
       pip {
           install "kivy"
           install "numpy"
           install "opencv-python"
           install "dlib"
       }
   }
   ```

## 使用说明

### 界面说明

| 控件 | 功能 |
|------|------|
| 相机预览区 | 显示实时美颜效果 |
| 美颜开关 | 开启/关闭美颜效果 |
| 📷 拍照按钮 | 拍摄照片并保存 |
| 切换摄像头 | 切换前置/后置摄像头 |
| 磨皮滑块 | 调整磨皮强度 (0-100%) |
| 美白滑块 | 调整美白强度 (0-100%) |
| 大眼滑块 | 调整眼睛放大比例 (1.0-1.5x) |
| 瘦脸滑块 | 调整瘦脸强度 (0-10%) |

### 拍照保存位置

- **Android**: `/storage/emulated/0/DCIM/BeautyCamera/`
- **桌面**: `~/Pictures/BeautyCamera/`

## 性能优化

针对移动端进行了以下优化：

1. **降采样处理**: 大尺寸图像先缩放处理，再恢复
2. **隔帧检测**: 人脸检测每3帧执行一次，美颜处理每帧执行
3. **快速算法**: 使用双边滤波代替NLM降噪，使用简化RBF代替完整MLS
4. **区域裁剪**: 只在人脸区域内进行美颜处理

## 注意事项

1. **人脸检测要求**: 需要在光线充足的环境下使用，确保人脸清晰可见
2. **性能**: 在低端设备上可能需要降低预览分辨率
3. **隐私**: 所有处理都在本地完成，不会上传任何人脸数据

## 故障排除

### 摄像头无法打开
- 检查摄像头权限是否已授予
- 尝试切换前置/后置摄像头
- 检查是否有其他应用占用了摄像头

### 美颜效果不明显
- 调整各项美颜参数的滑块
- 确保人脸在画面中清晰可见
- 检查美颜开关是否开启

### 运行卡顿
- 降低预览分辨率（修改代码中的max_size参数）
- 关闭部分美颜功能
- 在人脸检测设置中增大minSize参数

### dlib模型加载失败
- 确保`shape_predictor_68_face_landmarks.dat`文件存在
- 检查文件大小是否正确（约97MB）
- 重新下载模型文件

## 许可证

本项目基于MIT许可证开源。

人脸特征点检测使用dlib库，遵循Boost Software License。

## 致谢

- OpenCV: 计算机视觉库
- dlib: 机器学习库
- Kivy: 跨平台GUI框架
- python-for-android: Android打包工具
