# GitHub Actions 云编译部署指南

本项目已配置好GitHub Actions，可以在云端自动编译Android APK。

## 前置条件

1. 拥有GitHub账号（如果没有，请先注册：https://github.com/signup）
2. 本地已安装git（已完成）
3. 代码已完成git提交（已完成）

## 部署步骤

### 1. 创建GitHub仓库

1. 访问 https://github.com/new
2. 填写仓库信息：
   - **Repository name**: `beautycamera`（或您喜欢的名字）
   - **Description**: `美颜相机 - 基于Kivy的实时美颜应用`
   - **Visibility**: 选择 `Public`（公开）或 `Private`（私有）
   - **不要勾选** "Initialize this repository with a README"
3. 点击 **Create repository**

### 2. 推送代码到GitHub

在PowerShell或命令提示符中执行以下命令：

```bash
# 进入Phoneapp目录
cd d:\.AAA学习\大三下学期\数字图像处理\作业\大作业\Phoneapp

# 添加远程仓库（请替换YOUR_USERNAME为您的GitHub用户名）
git remote add origin https://github.com/YOUR_USERNAME/beautycamera.git

# 推送代码到GitHub
git push -u origin main
```

**注意**：如果遇到错误提示 "main branch doesn't exist"，请先重命名分支：

```bash
git branch -M main
git push -u origin main
```

### 3. 触发编译

推送代码后，GitHub Actions会自动开始编译：

1. 访问您的GitHub仓库页面
2. 点击 **Actions** 标签
3. 您会看到 "Build Android APK" 工作流正在运行
4. 点击工作流可以查看详细的编译日志

### 4. 下载编译好的APK

编译成功后（预计1-2小时）：

1. 在Actions页面找到已完成的工作流
2. 点击工作流运行记录
3. 在页面底部找到 **Artifacts** 部分
4. 点击 **beautycamera-debug** 下载APK文件
5. 解压下载的zip文件，得到APK

### 5. 安装APK到手机

#### 方法1：直接安装
1. 将APK文件传输到Android手机
2. 在手机上打开APK文件
3. 允许安装未知来源的应用
4. 完成安装

#### 方法2：使用ADB（推荐）
1. 在手机上开启USB调试模式
2. 连接手机到电脑
3. 运行命令：
```bash
adb install beautycamera-debug.apk
```

## 编译说明

### 编译时间

- 首次编译：约 **1-2小时**（需要下载和编译大量依赖）
- 后续编译：约 **30-60分钟**（使用了缓存加速）

### 编译失败怎么办？

如果编译失败，请检查：

1. **查看日志**：在Actions页面查看详细的错误日志
2. **常见问题**：
   - dlib编译失败：可能是内存不足，GitHub Actions提供7GB内存
   - 网络问题：依赖下载可能需要重试
   - 配置错误：检查buildozer.spec文件

3. **重新编译**：
   - 修改任意文件后推送即可触发重新编译
   - 或在Actions页面手动触发（Run workflow）

### 查看编译进度

在Actions工作流页面，您可以看到：
- 每个步骤的执行状态
- 实时输出的日志
- 预计剩余时间

## 常见问题

### Q1: 如何修改应用信息？

编辑 `buildozer.spec` 文件中的以下字段：

```ini
title = 美颜相机              # 应用标题
package.name = beautycamera  # 包名
package.domain = org.beautycamera  # 包域名
version = 1.0.0              # 版本号
```

修改后推送代码，会触发重新编译。

### Q2: 如何生成发布版本（Release APK）？

修改 `.github/workflows/android-build.yml` 文件：

```yaml
# 将这行：
buildozer -v android debug

# 改为：
buildozer -v android release
```

发布版本需要签名，建议先使用debug版本测试。

### Q3: 编译速度太慢怎么办？

可以使用以下方法加速：

1. **使用GitHub缓存**：已配置，会自动加速
2. **减少架构**：在buildozer.spec中只保留一个架构：
   ```ini
   android.archs = arm64-v8a
   ```
3. **使用轻量级依赖**：如果不需要dlib，可以移除该依赖

### Q4: 如何查看APK是否包含所有功能？

安装后检查：
- 摄像头权限是否正常
- 美颜功能是否可用
- 人脸检测是否工作
- 拍照保存功能是否正常

### Q5: 代码修改后如何更新APK？

1. 修改代码
2. 提交更改：
   ```bash
   git add .
   git commit -m "描述您的修改"
   ```
3. 推送到GitHub：
   ```bash
   git push
   ```
4. 等待编译完成，下载新APK

## 技术支持

如果遇到问题：

1. 查看 [ANDROID_BUILD.md](ANDROID_BUILD.md) 了解详细的编译说明
2. 查看 GitHub Actions 日志定位问题
3. 搜索相关错误信息

## 项目文件说明

```
Phoneapp/
├── .github/workflows/
│   └── android-build.yml     # GitHub Actions配置
├── recipes/
│   └── dlib/
│       └── __init__.py       # dlib编译配置
├── buildozer.spec             # Buildozer配置文件
├── main.py                    # 应用主程序
├── beautifier.py              # 美颜算法
├── beautycamera.kv            # UI界面
├── requirements.txt           # Python依赖
├── .gitignore                 # Git忽略文件
└── DEPLOY_GUIDE.md            # 本文件
```

## 下一步

1. 按照上述步骤创建GitHub仓库并推送代码
2. 等待编译完成
3. 下载APK并测试
4. 根据需要修改代码并重新编译

祝您使用愉快！🎉