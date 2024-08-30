---
title: MNN源码分析5 Pre-inference
index_img: https://s2.loli.net/2024/03/12/AUQ3MRt2ndakyup.png
tags: [模型部署,MNN]
date: 2024-03-18 13:07:30
categories: MNN 源码系列
published: true
comment: 'utterances'
---
MNN基本的工作流由两部分组成，即Offline Conversion和On-device Inference。On-device Inference由三部分组成，分别是：Pre-inference、算子级优化和Backend Abstraction.在Pre-inference模块中引入了一种对可选计算方案的代价评估机理，在已知输入大小和内核形状的前提下，从多种方案中选择一种最优的方案。具体需要结合代码。 
## CPU Backend
`Interpreter`在创建session中，在session.size()时，会allocMemory(),这时会调用相应的后端去执行onCreate，如果时卷积，则会调用`ConvolutionFloatFactory`,使用`_createUnit`选择不同的卷积实现。

```Cpp

static Execution* _createUnit(const Tensor* input, const Tensor* output, Backend* backend,
                              const Convolution2D* conv2d, const float* originWeight, size_t originWeightSize, const float* bias, size_t biasSize, std::shared_ptr<ConvolutionCommon::Int8Common> weightQuantInfo, bool supportSparse) {
    auto cpuBackend = (CPUBackend*)backend;
#ifdef MNN_LOW_MEMORY
    bool lowMemory = cpuBackend->memoryMode() == BackendConfig::Memory_Low;
#else
    bool lowMemory = false;
#endif
    auto common = conv2d->common();
#ifdef MNN_USE_ONEDNN
    return OneDNN::createConvolution(common, backend, originWeight, originWeightSize, bias, biasSize);
#endif

#ifdef MNN_USE_SPARSE_COMPUTE
    if (conv2d->sparseParameter() && nullptr != weightQuantInfo.get()) {
        if (supportSparse) {
            return new SparseConvolutionTiledExecutor(common, backend, weightQuantInfo->quan,
                                                      conv2d->sparseParameter(), bias, biasSize);
        }
    }
#endif
    bool fastWay = common->kernelY() == 1 && common->kernelX() == 1
        && output->width() == input->width() && output->height() == input->height()
        && common->strideX() == 1 && common->strideY() == 1;

    if (lowMemory) {
        if (fastWay && nullptr != weightQuantInfo.get()) {
            return new ConvolutionHybrid(common, backend, originWeight, originWeightSize, bias, biasSize, weightQuantInfo);
        } else {
            return new DenseConvolutionTiledExecutor(common, backend, originWeight, originWeightSize, bias, biasSize, weightQuantInfo);
        }
    }
    //如果卷积核是1 * 1， 使用 Convolution1x1Strassen
    if (fastWay) {
        return new Convolution1x1Strassen(common, backend, originWeight, originWeightSize, bias, biasSize, weightQuantInfo);
    }
    //如果权重大小为0 使用DenseConvolution
    if (originWeightSize == 0) {
        return new DenseConvolutionTiledExecutor(common, backend, originWeight, originWeightSize, bias, biasSize, weightQuantInfo);
    }
    // 判断能否使用Winograd 卷积
    if (!ConvolutionWinogradBridge::canUseWinograd(common)) {
        return new DenseConvolutionTiledExecutor(common, backend, originWeight, originWeightSize, bias, biasSize, nullptr);
    }
    PerfConfig convPerfconfig = DenseConvolutionTiledExecutor::bestTileConvolutionConfig(common, input, output, cpuBackend->threadNumber(), backend);
    auto winogradConfig = ConvolutionWinogradBridge::bestWinogradUnit(common, input, output, cpuBackend->threadNumber(), backend, convPerfconfig);
    if (winogradConfig.unit <= 1) {
        return new DenseConvolutionTiledExecutor(common, backend, originWeight, originWeightSize, bias, biasSize, nullptr);
    }
    //返回 winograd的实现 
    return ConvolutionWinogradBridge::createWinogradImpl(common, input, output, backend, originWeight, originWeightSize, bias, biasSize,
                                   winogradConfig);
}

```
整体来讲MNN会根据你的lowMemory的要求，以及kernel的大小来动态去选择卷积实现。


## Opencl Backend
![20240318135208](https://s2.loli.net/2024/03/18/TsUOxqVF4HAt6jz.png)

```Cpp
if(runtime->getCLTuneLevel() == Heavy) {
        while(lws[1] <= gws[1] || lws[1] <= 6) {
            lws[0] = 1;
            while(lws[0] <= gws[0] || lws[0] <= 6) {
                if(lws[0] <= maxWorkItemSizes[0] && lws[1] <= maxWorkItemSizes[1] && lws[0]*lws[1] <= maxWorkGroupSize) {
                    cl::Event event;
                    std::vector<uint32_t> internalGlobalWS(2, 1);
                    for (size_t i = 0; i < gws.size(); ++i) {
                        internalGlobalWS[i] = ROUND_UP(gws[i], std::max((uint32_t)1, lws[i]));
                    }
                    cl_int res = runtime->commandQueue().enqueueNDRangeKernel(
                                    mKernel, cl::NullRange,
                                    cl::NDRange(internalGlobalWS[0], internalGlobalWS[1]),
                                    cl::NDRange(lws[0], lws[1]),
                                    nullptr, &event);
                    MNN_CHECK_CL_SUCCESS(res, kernelName.c_str());
                    if (res != CL_SUCCESS) {
                        MNN_PRINT("lws tune res %s\n", kernelName.c_str());
                    }
                    
                    int cost_time = (int)runtime->getCostTime(&event);
                    //获得最短时间
                    if(cost_time < min_cost) {
                        min_cost = cost_time;
                        lws_prefer[0] = lws[0];
                        lws_prefer[1] = lws[1];
                    }
                }
                lws[0]++;
            }
            lws[1]++;
        }
    } 

```
对于`opencl backend` MNN 则会根据`TuneLevel`动态搜索最合的`local_work_size`和`global_work_size` 得到执行的最小时间。