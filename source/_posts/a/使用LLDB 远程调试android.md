---
title: 使用LLDB 远程调试 安卓native C++ 程序的vscode 配置
index_img: https://s2.loli.net/2024/03/11/4sRNlcjAr73IMaU.png
tags: [linux, adb,LLDB,android,Debug]
date: 2024-03-05 10:00:00
comment: 'utterances'
---
# 使用LLDB 远程调试 安卓native C++ 程序的vscode 配置

## 背景
我目前使用windows ssh连接到linux 服务器上,对android native C++代码进行编写,编译(因为服务器核心数量多,编译速度快). 目前高版本的NDK如r26等,已经不再对gdbserver 提供支撑, 所以迁移到LLDB调试,也是主流的技术方向.

## 相关配置
1. 安装vscode 插件
CodeLLDB 
![20240311140633](https://s2.loli.net/2024/03/11/pFxMIkRtHvYUyBL.png)
2. 下载NDK
   [NDK下载网站链接](https://developer.android.com/ndk/downloads?hl=zh-cn)
   ```bash
    wget https://dl.google.com/android/repository/android-ndk-r26c-linux.zip?hl=zh-cn
    unzip android-ndk-r26c-linux.zip\?hl\=zh-cn

   ```
3. adb 上传lldb-server
    ```bash
    cd  android-ndk-r26c
    find ./ -name "lldb-server"
    ```
    ![20240311141315](https://s2.loli.net/2024/03/11/1X3Hesi5EPVQ9oy.png)
    选取aarch64 版本的lldb-server
    ```bash
    adb push ./toolchains/llvm/prebuilt/linux-x86_64/lib/clang/17/lib/linux/aarch64/lldb-server /data/local/tmp
    adb shell "chmod +xrw /data/local/tmp/lldb-server"
    ```
4. 启动lldb-server
   `lldb-server platform --server --listen *:9999` 端口可以自己选,不冲突就行
5. 配置vscode 的launch.json 文件
   ```json
    {
        "name": "Remote launch",
        "type": "lldb",
        "request": "launch",
        "program": "${workspaceFolder}/build/debuggee", // Local path. 
        "initCommands": [
            "platform select <platform>", // For example: 'remote-linux', 'remote-macosx', 'remote-android', etc.
            "platform connect connect://<remote_host>:<port>",
            "settings set target.inherit-env false", // See note below.
        ],
        "env": {
            "PATH": "xxx", // remote 的path
            "LD_LIBRARY_PATH": "/data/local/tmp/code"
        },
        "args":[
            "/data/local/tmp/code/pose.mnn",
            "/data/local/tmp/code/input.png",
            "/data/local/tmp/code/out.png"
        ],
        "breakpointMode": "file"
    }
   ```
    有几个点需要注意
   - program 是linux本地的路径,不是在remote手机上的路径
   - args 是程序运行时的参数,如果需要路径,是remote上的路径
   - env 配置的是remote上运行的env,比如链接库地址,PATH等
   - breakpointMode 设置为file就行
##  建议
- 如果配置不成功,首先不使用vscode插件,在自己的电脑上命令行能够启动LLDB,LLDB-server和调试native c++程序.
- codeLLDB 使用的是自己的lldb,需要检查该lldb能够正常运行和调试
  ![20240311142552](https://s2.loli.net/2024/03/11/i1LOAze2saZJ6VK.png)

