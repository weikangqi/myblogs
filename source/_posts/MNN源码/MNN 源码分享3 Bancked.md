---
title: MNN源码分析3 Banckend 
index_img: https://s2.loli.net/2024/03/12/AUQ3MRt2ndakyup.png
tags: [模型部署,MNN]
date: 2024-03-16 18:42:30
categories: MNN 源码系列
published: true
excerpt: 以Opencl Banckend 为例
comment: 'utterances'
---

## 分析opencl banckend 

在了解banckend 特定后端实现前，有几个函数需要注意
```Cpp
#define ALIMIN(x, y) ((x) < (y) ? (x) : (y))
#define ALIMAX(x, y) ((x) > (y) ? (x) : (y))

#define UP_DIV(x, y) (((x) + (y) - (1)) / (y))
#define ROUND_UP(x, y) (((x) + (y) - (1)) / (y) * (y))
#define ALIGN_UP4(x) ROUND_UP((x), 4)
#define ALIGN_UP8(x) ROUND_UP((x), 8)
```
` ALIMIN(x, y) ((x) < (y) ? (x) : (y))`
这个宏定义是一个简单的三元运算符，用来比较两个值 x 和 y 的大小，然后返回较小的那个值。具体来说，它的作用是：

1. 检查 x 是否小于 y。
2. 如果是，返回 x；如果不是，返回 y。

这个宏定义可以用于需要在两个值中选择较小值的情况，比如在编程中需要取两个数中的最小值时使用。

` UP_DIV(x, y) (((x) + (y) - (1)) / (y))`
1. 如果x/y 刚好除尽 
`UP_DIV(10,2)= 5`  `UP_DIV(4,2)= 2`
2.  如果x/y 除不尽 结果向上取整
`UP_DIV(10,3)= 4`

`ROUND_UP(x, y) (((x) + (y) - (1)) / (y) * (y))`
宏定义可以用于需要将数值向上取整到最接近的某个倍数的情况，比如内存对齐等
比如 
`ROUND_UP(4, 8) = 8`  `ROUND_UP(9, 8) = 16`  `ROUND_UP(8, 8) = 8`

## opencl runtime
一个backend 都会由一个runtime。runtime也就是backend的核心成员,runtime 主要包装了Opencl的一些方法，以及Context，Events等底层变量。
```Cpp
class OpenCLRuntime {
public:
    OpenCLRuntime(const BackendConfig::PrecisionMode precision, const int cl_mode, int platformSize, int platformId, int deviceId);
    ~OpenCLRuntime();
    OpenCLRuntime(const OpenCLRuntime &) = delete;
    OpenCLRuntime &operator=(const OpenCLRuntime &) = delete;

    bool isSupportedFP16() const;
    bool isWeightCpuTransHalf() const;
    bool isDeviceSupportedFP16() const;
    bool isDeviceSupportedLowPower() const;
    bool isSupportedDotInt8() const;
    bool isSupportedDotAccInt8() const;
    bool isSupportedIntelSubgroup() const;
    ::cl::Context &context();
    ::cl::CommandQueue &commandQueue();
    ::cl::CommandQueue &recordableQueue();
    uint64_t deviceGlobalMemeryCacheSize() const;
    uint32_t deviceComputeUnits() const;
    uint32_t MaxThreadsPerDevice() const;
    uint32_t MaxWorkGroupSize() const;
    uint32_t maxFreq() const;
    uint64_t getMaxWorkGroupSize(const ::cl::Kernel &kernel);
    uint64_t GetKernelWaveSize(const cl::Kernel &kernel);
    std::vector<uint32_t> getMaxWorkItemSizes();
    uint64_t getMaxLocalMem() const;
    uint32_t getUseRecordableQueueSize(){
        return mUseRecordableQueueSize;
    }
    bool isSupportRecordQueue(){
        return mUseRecordQueue;
    }
    bool isDevideOpRecord(){
        return mDevideOpRecord;
    }
    GpuType getGpuType() {
        return mGpuType;
    }
    MaliAr getMaliAr() {
        return mMaliAr;
    }
    float getCLVersion() {
        return mCLVersion;
    }
#ifdef MNN_OPENCL_SVM_ENABLE
    cl_device_svm_capabilities getSvmCapabilities() {
        return mSvmCapabilities;
    }
#endif
    GpuMemObject getGpuMemType() {
        return mMemType;
    }
    CLTuneLevel getCLTuneLevel() {
        return mTuneLevel;
    }
    std::string getDeviceName() {
        return mDeviceName;
    }
    void pushEvent(std::pair<std::string, cl::Event> data) {
        return mEvents.push_back(data);
    }
    void printEventTime();
    void clearEvent(){
        mKernelTime = 0;
        mEvents.clear();
    }
    uint64_t maxAllocSize() const;
    void setCommandQueueProfileEnable();
    void setCommandQueueProfileDisable();

    unsigned int mQueueCount = 0;
    unsigned int getQueueNum();
    
    unsigned int mKernelTime = 0;

    std::map<std::pair<std::string, std::vector<uint32_t>>, std::pair<std::vector<uint32_t>, uint32_t>>& tunedLwsMap();
    
    std::map<std::string, std::vector<std::pair<std::vector<uint32_t>, std::pair<std::vector<uint32_t>, uint32_t>>>>& getTuneLwsMap();
    
    ::cl::Kernel buildKernel(const std::string &programName, const std::string &kernelName,
                             const std::set<std::string> &buildOptions);
    ::cl::Kernel buildKernelFromSource(const std::string&, const std::string &kernelName,
                                       const std::set<std::string> &buildOptions);

    std::vector<size_t> getMaxImage2DSize();
    bool isCreateError() const {
        return mIsCreateError;
    }

    float flops() const {
        return mFlops;
    }

    double getCostTime(const cl::Event *event);
    double getQueuedTime(const cl::Event *event);
    double getSubmitTime(const cl::Event *event);

    std::pair<const void*, size_t> makeCache(void* tuneInfo);
    bool setCache(std::pair<const void*, size_t> cache);  //将build的东西保存
    std::shared_ptr<::cl::CommandQueue> mCommandQueuePtr;
private:
    bool loadProgram(const std::string &programName, cl::Program *program);
    bool buildProgram(const std::string &buildOptionsStr, cl::Program *program);
    bool getDeviceSupportsExtension(const cl::Device &device, const char *extensionName);
    void setGpuMode(const int cl_mode_num);

private:
    std::shared_ptr<::cl::Context> mContext;     // opencl 上下文 COntext
    std::shared_ptr<::cl::Device> mFirstGPUDevicePtr; //根据DeviceID 选择的GPU devices
     // <  <kernel name> <build arges> <cl program> > cl program 是根据kernel name 和build arges build的结果，也会存在cache中 
    std::map<std::tuple<std::string, std::string>, ::cl::Program> mBuildProgramMap; 
    std::shared_ptr<::cl::CommandQueue> mRecordableQueuePtr;   //opencl 命令队列
    // 下面是GPU的一些参数信息
    uint64_t mGPUGlobalMemeryCacheSize;
    uint32_t mGPUComputeUnits;
    uint32_t mMaxFreq; 
    uint32_t mMaxMemAllocSize; 
    uint64_t mMaxLocalMemSize;
    uint32_t mMaxThreadsPerDevice;
    uint32_t mMaxWorkGroupSize;
    uint32_t mUseRecordableQueueSize;
    bool mUseRecordQueue = false;  //recordQueue 之后可以分析时间
    bool mDevideOpRecord = true;
    bool mIsSupportedFP16     = false;  
    bool mIsDeviceSupportedFP16 = false;
    bool mIsDeviceSupportedLowPower = false;
    bool mSupportDotInt8 = false;
    bool mSupportDotAccInt8 = false;
    bool mSupportedIntelSubgroup = false;
    GpuType mGpuType;
    MaliAr mMaliAr;    //mali GPU的架构
    float mCLVersion = 1.0f;  //mCLVersion 是指opencl的 version
    std::vector<std::pair<std::string, cl::Event>> mEvents;  // opencl event

#ifdef MNN_OPENCL_SVM_ENABLE
    cl_device_svm_capabilities mSvmCapabilities;
#endif
    GpuMemObject mMemType = AUTO;  //构造函数 根据gpu类型来选择是buffer 还是image  （mali 以及intel）的是buffer
    CLTuneLevel mTuneLevel = Wide;
    std::string mDeviceName;
    bool isSetWorkGroupAttribute = false;
    std::string mDefaultBuildParams;
    float mFlops = 4.0f;
    bool mIsCreateError{false};
    
    double mStartNanos;
    double mStopNanos;

    std::map<std::pair<std::string, std::vector<uint32_t>>, std::pair<std::vector<uint32_t>,  uint32_t>> mTunedLws;
    std::map<std::string, std::vector<std::pair<std::vector<uint32_t>, std::pair<std::vector<uint32_t>,  uint32_t>>>> mTuneLws;
    std::vector<uint8_t> mBuffer;
    const void* mCacheOutside = nullptr;
    size_t mCacheOutsideSize = 0;
};

} // namespace MNN
#endif  /* OpenCLRuntime_hpp */




```



## opencl CLRuntime
CLRuntime 是第二次封装。OpenclRuntime是对底层的抽象，而CLRuntime可以理解为中间层，顶层OpenclBancked和底层OpenclRuntime。
让我们来看看CLRuntime的构造函数
```Cpp
CLRuntime::CLRuntime(const Backend::Info& info, int platformSize, int platformId, int deviceId){
    mInfo = info;

    BackendConfig::PrecisionMode precision = BackendConfig::Precision_Normal;
    BackendConfig::PowerMode power         = BackendConfig::Power_Normal;
    BackendConfig::MemoryMode memory       = BackendConfig::Memory_Normal;
    if (nullptr != mInfo.user) {
        precision = mInfo.user->precision;
        power     = mInfo.user->power;
        memory    = mInfo.user->memory;
    }

    // Shader precision
    mOpenCLRuntime.reset(new OpenCLRuntime(precision, mInfo.gpuMode, platformSize, platformId, deviceId));
    //Whether runtimeError
    mCLRuntimeError = mOpenCLRuntime->isCreateError();
    mPrecision = precision;
    mMemory = memory;
    mTunedInfo = new TuneInfo;
    
    mImagePool.reset(new ImagePool(mOpenCLRuntime->context()));
    mBufferPool.reset(new BufferPool(mOpenCLRuntime->context(), CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR));
}
```

主要新增了`mImagePool` Image池和`Buffer` 池，来管理内存。

### Buffer Pool

```Cpp
class BufferPool : public NonCopyable {
public:
    BufferPool(cl::Context& context, cl_mem_flags flags) : mContext(context) {
        mFlag = flags;
    }

    cl::Buffer* alloc(int size, bool separate = false);  //向内存池申请内存
    void recycle(cl::Buffer* buffer, bool release = false);//向内存池释放内存
    void clear();
    void releaseFreeList();
    size_t totalSize() { return mTotalSize; }

    struct Node {
        int size;
        std::shared_ptr<cl::Buffer> buffer;
    };

private:
    std::map<cl::Buffer*, std::shared_ptr<Node>> mAllBuffer;
    std::multimap<int, std::shared_ptr<Node>> mFreeList;

    cl::Context& mContext;
    cl_mem_flags mFlag;
    size_t mTotalSize = 0;
};
```
在创建CLRuntime的时候，`mBufferPool.reset(new BufferPool(mOpenCLRuntime->context(), CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR));` 来创建`bufferpool`.
### bufferpool 申请内存
```Cpp
cl::Buffer* BufferPool::alloc(int size, bool separate) {
    if (!separate) {
        auto iter = mFreeList.lower_bound(size);
        if (iter != mFreeList.end()) {
            auto buffer = iter->second->buffer.get();
            mFreeList.erase(iter);
            return buffer;
        }
    }
    std::shared_ptr<Node> node(new Node);  
    cl_int ret = CL_SUCCESS;
    mTotalSize += size;        //记录容量
    node->size = size;
    node->buffer.reset(new cl::Buffer(mContext, mFlag, size, NULL, &ret));   //使用opencl 创建buffer 内存
    if (nullptr == node->buffer.get() || ret != CL_SUCCESS) {
        MNN_ERROR("Alloc Buffer %d error, code:%d \n", size, ret);
        return nullptr;
    }
    mAllBuffer.insert(std::make_pair(node->buffer.get(), node));   // 插入到pool中

    return node->buffer.get(); //返回内存指针
}
```
其中`node`的定义为
```Cpp
    struct Node {
        int size;
        std::shared_ptr<cl::Buffer> buffer;
    };
```

### bufferpool回收内存
```Cpp
class CLMemReleaseBuffer : public Backend::MemObj {
public:
    CLMemReleaseBuffer(cl::Buffer* bId, BufferPool* bufferPool) {
        mBuffer = bId;
        mBufferPool = bufferPool;
    }
    virtual ~ CLMemReleaseBuffer() {
        mBufferPool->recycle(mBuffer);
    }
private:
    cl::Buffer* mBuffer;
    BufferPool* mBufferPool;
};
```
在向内存池申请完内存后，对外不是直接返回内存指针而是使用`new CLMemReleaseBuffer(buffer, mStaticBufferPool.get())` 来返回一个`CLMemReleaseBuffer`对象，这个对象记录你的内存地址，和内存池。通过这个类来访问内存。当他析构时，将内存回收到内存池中。




   
需要提出的是，这个时候内存池虽然建好了，真正分配内存，是各个OP实例化时，调用算子类比如卷积Cov类时，onsize()才会在内存中池中，创建或者申请一段内存。
比如：

```Cpp
 ConvBufWinograd::onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs)
 {
    ...
    mOpenCLBackend->onAcquireBuffer(mSource.get(), Backend::DYNAMIC);
    ...
 }
```
### OpenCLBackend::onAcquire

```Cpp
Backend::MemObj* OpenCLBackend::onAcquire(const Tensor* nativeTensor, StorageType storageType) {
    #ifdef LOG_VERBOSE
    MNN_PRINT("Start OpenCLBackend::onAcquireBuffer !\n");
    #endif

    auto tensorShape = OpenCL::tensorShapeFormat(nativeTensor);
    int N = tensorShape.at(0);
    int H = tensorShape.at(1);
    int W = tensorShape.at(2);
    int C = tensorShape.at(3);

    #ifdef LOG_VERBOSE
    MNN_PRINT("OpenCLBackend::onAcquireBuffer: NHWC:[%d, %d, %d, %d]\n", N, H, W, C);
    #endif

    #ifndef MNN_OPENCL_BUFFER_CLOSED
    if(mOpenCLRuntime->getGpuMemType() == BUFFER) {
        size_t size;
        if (nativeTensor->dimensions() >= 2) {
            auto alignC = ROUND_UP(C, 8);
            // increment of height and width
            auto hR = ROUND_UP(H + 3, 4) - H;
            auto wR = ROUND_UP(W + 3, 4) - W;
            size = N * alignC * W * H;
            size = size + hR * W * 4 + wR * 4;
        } else {
            size = nativeTensor->elementSize();
            size = ROUND_UP(size, 4);
        }

        if (mOpenCLRuntime->isSupportedIntelSubgroup()) {
            int cPack = TensorUtils::getTensorChannelPack(nativeTensor);
            auto pads  = TensorUtils::getDescribe(nativeTensor)->mPads;
            size_t imageWidth  = (size_t) ROUND_UP(UP_DIV(C, cPack), 2) * ROUND_UP(pads.left + W + pads.right, 4);//C-round to 8,W-round to 4, for memory alloc
            size_t imageHeight = (size_t)N * H;
            size = imageWidth*imageHeight*cPack;
        }
        cl_channel_type dataType = CL_FLOAT;
        //when support and want fp16, use half datatype
        if (getOpenCLRuntime()->isSupportedFP16()) {
            dataType = CL_HALF_FLOAT;
        }

        if (storageType == DYNAMIC_SEPERATE) {
            auto buffer = mBufferPool->alloc(size*
                          (dataType==CL_HALF_FLOAT?sizeof(half_float::half):sizeof(float)), true);
            ((Tensor*)nativeTensor)->buffer().device = (uint64_t)buffer;
            return new CLMemReleaseBuffer(buffer, mBufferPool.get());
        }
        if (storageType == DYNAMIC) {
            auto buffer = mBufferPool->alloc(size*
                          (dataType==CL_HALF_FLOAT?sizeof(half_float::half):sizeof(float)));
            ((Tensor*)nativeTensor)->buffer().device = (uint64_t)buffer;
            return new CLMemReleaseBuffer(buffer, mBufferPool.get());
        }
        MNN_ASSERT(storageType == STATIC);
#ifdef MNN_LOW_MEMORY
        // for weight quant model's weight
        if ((nativeTensor->getType().code == halide_type_int) &&
            (nativeTensor->getType().bits == 8 || nativeTensor->getType().bits == 4)) {
            // int8 quant
            size_t alloc_size = size;
            if (nativeTensor->getType().bits == 4) {
                // int4 quant
                alloc_size = size / 2;
            }
            auto buffer = mStaticBufferPool->alloc(alloc_size);
            ((Tensor*)nativeTensor)->buffer().device = (uint64_t)buffer;
            return new CLMemReleaseBuffer(buffer, mStaticBufferPool.get());
        }
#endif
        auto buffer = mStaticBufferPool->alloc(size*
                     (dataType == CL_HALF_FLOAT ? sizeof(half_float::half) : sizeof(float)));
        ((Tensor*)nativeTensor)->buffer().device = (uint64_t)buffer; // fix
        return new CLMemReleaseBuffer(buffer, mStaticBufferPool.get());
    }
    else
    #endif /* MNN_OPENCL_BUFFER_CLOSED */
    {
        size_t imageWidth  = (size_t) (UP_DIV(C, 4) * W);//image mode only C pack to 4
        size_t imageHeight = (size_t)N * H;
        cl_channel_type dataType = CL_HALF_FLOAT;
        //when user want high precision, use float datatype
        if (mPrecision == BackendConfig::Precision_High) {
            dataType = CL_FLOAT;
        }

        if (storageType == DYNAMIC_SEPERATE) {
            auto image                               = mImagePool->alloc(imageWidth, imageHeight, dataType, true);
            ((Tensor*)nativeTensor)->buffer().device = (uint64_t)image; // fix
            return new CLMemReleaseImage(image, mImagePool.get());
        }
        if (storageType == DYNAMIC) {
            auto image                               = mImagePool->alloc(imageWidth, imageHeight, dataType);
            ((Tensor*)nativeTensor)->buffer().device = (uint64_t)image; // fix
            return new CLMemReleaseImage(image, mImagePool.get());
        }
        MNN_ASSERT(storageType == STATIC);
        auto image                               = mStaticImagePool->alloc(imageWidth, imageHeight, dataType);
        ((Tensor*)nativeTensor)->buffer().device = (uint64_t)image; // fix
        return new CLMemReleaseImage(image, mStaticImagePool.get());
    }
}
```
`OpenCLBackend::onAcquire` 是openclbackend 对外暴露的接口，主要是做了对Tensor进行计算对其后的空间大小，调用`CLMemReleaseImage`对一个native Tensor分配内存。