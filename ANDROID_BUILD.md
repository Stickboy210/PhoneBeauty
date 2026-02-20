# Android打包指南

## 方案1：使用Buildozer（完整功能，需要编译dlib）

### 前提条件
- Linux环境（推荐Ubuntu 20.04+）
- 至少20GB磁盘空间
- Python 3.8+

### 安装依赖

```bash
# 更新系统
sudo apt-get update

# 安装必要工具
sudo apt-get install -y \
    git zip unzip openjdk-17-jdk python3-pip autoconf libtool pkg-config \
    zlib1g-dev libncurses5-dev libncursesw5-dev liblzma-dev \
    libffi-dev libssl-dev libsqlite3-dev cmake

# 安装buildozer
pip3 install buildozer

# 安装Cython
pip3 install Cython
```

### 配置dlib Recipe（重要）

由于dlib需要单独编译，你需要创建python-for-android的recipe：

```bash
# 创建recipes目录
mkdir -p recipes/dlib

# 创建__init__.py文件
cat > recipes/dlib/__init__.py << 'EOF'
from pythonforandroid.recipe import CppCompiledComponentsPythonRecipe

class DlibRecipe(CppCompiledComponentsPythonRecipe):
    version = '19.24'
    url = 'https://github.com/davisking/dlib/archive/v{version}.tar.gz'
    depends = ['numpy', 'python3']
    site_packages_name = 'dlib'
    
    def get_recipe_env(self, arch):
        env = super().get_recipe_env(arch)
        env['CMAKE_BUILD_TYPE'] = 'Release'
        return env

recipe = DlibRecipe()
EOF
```

### 修改buildozer.spec

```ini
# 添加recipe路径
p4a.local_recipes = ./recipes

# 添加dlib到requirements
requirements = python3,kivy,numpy,opencv-python,dlib
```

### 打包

```bash
# 调试版本
buildozer -v android debug

# 部署到设备
buildozer android debug deploy run

# 发布版本
buildozer android release
```

**注意**：编译dlib可能需要很长时间（30分钟到数小时），且容易出错。

---

## 方案2：使用轻量级版本（推荐）

使用`main_android.py`和`beautifier_light.py`，不依赖dlib。

### 步骤

1. **备份原文件**
```bash
cp main.py main_full.py
cp beautifier.py beautifier_full.py
```

2. **使用Android版本**
```bash
cp main_android.py main.py
cp beautifier_light.py beautifier.py
```

3. **修改buildozer.spec**
```ini
requirements = python3,kivy,numpy,opencv-python
```

4. **打包**
```bash
buildozer -v android debug
```

这个版本功能有所简化，但打包成功率更高。

---

## 方案3：使用Chaquopy（最简单）

Chaquopy可以在Android Studio中直接运行Python代码。

### 步骤

1. **安装Android Studio**

2. **创建新项目**
   - 选择"Empty Activity"
   - Minimum SDK: API 21

3. **配置build.gradle (Project)**
```gradle
buildscript {
    dependencies {
        classpath "com.chaquo.python:gradle:14.0.2"
    }
}
```

4. **配置build.gradle (Module: app)**
```gradle
plugins {
    id 'com.android.application'
    id 'com.chaquo.python'
}

android {
    // ... 其他配置
    
    sourceSets {
        main {
            python {
                srcDirs = ["src/main/python"]
            }
        }
    }
}

dependencies {
    implementation "com.chaquo.python:android:14.0.2"
}

python {
    buildPython "/usr/bin/python3"  // 或 Windows 上的 python.exe 路径
    
    pip {
        install "kivy"
        install "numpy"
        install "opencv-python"
        // install "dlib"  // Chaquopy可能不支持dlib
    }
}
```

5. **复制Python代码**
   - 将`main_android.py`和`beautifier_light.py`复制到`app/src/main/python/`
   - 重命名`main_android.py`为`main.py`

6. **创建Java入口**
```java
// MainActivity.java
package org.beautycamera;

import android.os.Bundle;
import com.chaquo.python.Python;
import com.chaquo.python.android.AndroidPlatform;

public class MainActivity extends AppCompatActivity {
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        
        if (!Python.isStarted()) {
            Python.start(new AndroidPlatform(this));
        }
        
        Python py = Python.getInstance();
        py.getModule("main").callAttr("run");
    }
}
```

7. **修改main.py**
在最后添加：
```python
def run():
    BeautyCameraAndroidApp().run()
```

8. **构建APK**
   - Build → Build Bundle(s) / APK(s) → Build APK(s)

---

## 常见问题

### Q1: 打包时内存不足
**解决**: 增加swap空间
```bash
sudo fallocate -l 8G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

### Q2: OpenCV编译失败
**解决**: 使用预编译的opencv-python-android
```ini
requirements = python3,kivy,numpy
android.gradle_dependencies = "org.opencv:opencv-android:4.5.3"
```

### Q3: 运行时崩溃
**解决**: 检查日志
```bash
buildozer android logcat
```

### Q4: 摄像头权限问题
**解决**: 确保AndroidManifest.xml包含：
```xml
<uses-permission android:name="android.permission.CAMERA" />
<uses-feature android:name="android.hardware.camera" android:required="true" />
```

---

## 推荐方案总结

| 方案 | 难度 | 功能完整度 | 推荐度 |
|------|------|-----------|--------|
| Buildozer + dlib | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ |
| Buildozer + 轻量版 | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| Chaquopy | ⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| 原生重写 (Java/Kotlin) | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |

**新手推荐**: 方案3 (Chaquopy)
**功能优先**: 方案1 (完整dlib功能)
**平衡选择**: 方案2 (轻量版)
