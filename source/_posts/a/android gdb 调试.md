---
title: Linux 中使用adb 连接手机
index_img: https://s2.loli.net/2024/03/13/m8h1yPiQsuoNzMT.png
tags: [linux, adb]
date: 2024-03-05 10:00:00
comment: 'utterances'
---


# Linux 中使用adb 连接手机

在做实验时，使用linux服务器连接手机，一直出现 connect 后，手机直接 offline 的情况。
解决： 不能使用`sudo apt install adb` 安装adb，来连接手机。
解决方法：
```bash
mkdir cli-tools
wget -c https://dl.google.com/android/repository/platform-tools-latest-linux.zip
unzip platform-tools-latest-linux.zip 
cd platform-tools/
./adb connect yourIP:Port
```

