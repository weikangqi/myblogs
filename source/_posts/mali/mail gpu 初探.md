---
title: Mali GPU 初探
index_img: https://s2.loli.net/2024/03/13/5Q9Bf2eLNEdqTZy.png
tags: [模型部署,MNN]
date: 2024-03-012 18:42:30
categories: Mali
comment: 'utterances'
---

## 
手机上的GPU,如果是MTK(联发科)话,使用的是ARM的公版GPU mali,以及RK系列等也都是使用的mali GPU.

ARM GPU 有 `Bifrost`, `Valhall` ,`5th Gen`架构,不同架构,不同家族的GPU对opencl的版本支持也不一样,但是基本上也都支持 OpenCL2 以上了, 5th Gen都支持OpenCl 3.0,大部分Valhall 架构支撑opencl3.0.
不太幸运的是我手上的Mali-G77,Mali-G57,是Valhall 架构,但是只支持到Opencl2.1

## 一些概念
"Warp width" 和 "Thread count (max)" 是与图形处理器（GPU）相关的两个重要概念。


1. **Warp width（线程束宽度）**：
   - 在GPU中，线程束（warp）是执行单元的基本调度单位。线程束宽度指的是在一个时钟周期内，GPU处理器能够同时执行的线程束中线程的数量。
   - 例如，NVIDIA的GPU通常有32个线程组成一个线程束，这意味着这些32个线程会在一个时钟周期内同时执行。

2. **Thread count (max)（最大线程数）**：
   - 这是指一个GPU处理器能够支持的最大线程数。
   - GPU的最大线程数取决于其架构和型号，通常是成千上万甚至数十万个线程。
3. **warp (线程束)**
   一个warp中的所有线程会同时执行相同的指令，但是对于不同的warp，它们可以执行不同的指令。这意味着在同一个时钟周期内，不同的warp可以执行不同的指令，从而实现了指令级并行

**关系**：
- "Warp width" 和 "Thread count (max)" 之间的关系在于，"Thread count (max)" 可以被 "Warp width" 整除，因为线程束的宽度决定了在一个时钟周期内能够同时执行的线程数量。
- 例如，如果一个GPU的线程束宽度是32，而它的最大线程数是1024，那么这个GPU能够同时执行的线程数应该是32的整数倍，即32、64、96、128、...、992、1024。

G77,以及G55的`Warp width`都是16,`Thread count (max)`最大线程数为1024,还算不错.


> 参考
> ARM_GPU_Data_Sheet