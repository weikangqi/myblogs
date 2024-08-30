---
title: ARM neon 指令 （Android）代码环境配置
index_img: https://s2.loli.net/2024/03/18/JUWMRmv6fEaBHAu.png
tags: [neon,arm,simd]
date: 2024-03-18:42:30
categories: neon 学习系列
published: true
excerpt: 学习arm neon,最终实现一个FFT为目标。
comment: 'utterances'
---


本次使用的ARM平台是手机平台

1. 首先确保安装好了NDK
   配置好了PATH变量，将NDK路径加入PATH
2. CMakeList 设置
    ```bash
    cmake_minimum_required (VERSION 3.16)
    project (Dome)
    add_subdirectory(./src/neon)
    add_executable(neon neon.cpp)
    set_target_properties(neon PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/build")

    target_link_libraries(neon PRIVATE c++_static dl)

    set(THIS_COMPILE_FLAGS -march=armv8-a)

    target_compile_options(neon PRIVATE ${THIS_COMPILE_FLAGS})
    ```
3. 编译脚本
    ```bash
    cmake .. \
    -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
    -DCMAKE_BUILD_TYPE=Debug \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
    -DANDROID_ABI="arm64-v8a" \
    -DANDROID_STL=c++_static \
    -DANDROID_NATIVE_API_LEVEL=android-21  \
    -DMNN_BUILD_FOR_ANDROID_COMMAND=true \
    -DANDROID_TOOLCHAIN=clang \
    -DCMAKE_GENERATOR="Unix Makefiles"
    ```
4. 运行
    ```shell
    adb push neon  /data/local/tmp && adb shell /data/local/tmp/neon
    ```
5. 源代码
   ```Cpp
    #include <stdio.h>
    #include <stdlib.h>
    #include <arm_neon.h>
    #include <math.h>
    #include <cstddef>

    int main()
    {
        //定义a, b, c
        unsigned char a[8] = {0, 1, 2, 3, 4, 5, 6, 7};
        unsigned char b[8] = {8, 9, 10, 11, 12, 13, 14, 15};
        unsigned char c[8];
        uint8x8_t rega, regb, regc;	//定义3个8x8bit无符号整型的 NEON 寄存器
        //加载 a, b 到寄存器
        rega = vld1_u8(&a[0]);
        regb = vld1_u8(&b[0]);
        regc = vadd_u8(rega, regb);	//做加法
        vst1_u8(&c[0], regc);		//回写到c中
        //测试
        for(int i = 0 ; i < 8 ; i++)
        {
            printf("%d 	",c[i] );
        }
        printf("\n");
    }
   ```