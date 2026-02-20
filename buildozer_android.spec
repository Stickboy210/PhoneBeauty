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
source.include_exts = py,png,jpg,kv,atlas

# 版本
version = 1.0.0

# 应用程序需求 - 轻量级版本，不需要dlib
requirements = python3,kivy,numpy,opencv-python

# Android设置
android.api = 33
android.minapi = 21
android.ndk = 25b
android.sdk = 33

# 权限
android.permissions = CAMERA,WRITE_EXTERNAL_STORAGE,READ_EXTERNAL_STORAGE,INTERNET

# 特性
android.features = android.hardware.camera,android.hardware.camera.autofocus

# 屏幕方向
orientation = portrait

# 全屏模式
fullscreen = 0

# 需要的架构
android.archs = arm64-v8a, armeabi-v7a

# 主程序入口（使用Android版本）
# 打包时使用: buildozer -v android debug
# 会自动使用main.py作为入口

[buildozer]

# 日志级别
log_level = 0

# 构建目录
build_dir = ./.buildozer

# 使用源码目录
bin_dir = ./bin

# 颜色输出
color = True
