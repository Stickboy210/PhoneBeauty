[app]

# 应用标题
title = 美颜相机

# 包名
package.name = beautycamera

# 包域名
package.domain = org.beautycamera

# 源文件
source.dir = .

# 包含的文件
source.include_exts = py,png,jpg,kv,atlas,dat,yaml

# 版本
version = 1.0.0

# 应用程序需求
# 包含完整的dlib人脸检测功能
requirements = python3,kivy,numpy,opencv-python,dlib

# 针对Android的设置
# 添加本地recipes路径以支持dlib编译
p4a.local_recipes = ./recipes

# 自动接受Android SDK许可
android.accept_sdk_license = True

android.api = 31
android.minapi = 21
android.ndk = 25.2.9519653
android.sdk = 31

# 权限
android.permissions = CAMERA,WRITE_EXTERNAL_STORAGE,READ_EXTERNAL_STORAGE,INTERNET

# 特性
android.features = android.hardware.camera,android.hardware.camera.autofocus

# 屏幕方向
orientation = portrait

# 全屏模式
fullscreen = 0

# Android应用程序图标
# android.presplash_icon = icon.png
# android.icon = icon.png

# 服务模式（后台运行）
services = 

# 添加Android特定文件
# android.add_aars = 
# android.add_jars = 

# 添加资源
# android.add_src = 

# 需要的架构
android.archs = arm64-v8a, armeabi-v7a

# 优化设置
android.allow_backup = True

# 混淆
android.release_artifact = apk

[buildozer]

# 日志级别 (2 = 错误, 1 = 警告, 0 = 信息)
log_level = 0

# 构建目录
build_dir = ./.buildozer

# 使用源码目录
bin_dir = ./bin

# 颜色输出
color = True

# 是否自动更新buildozer
auto_update = True
