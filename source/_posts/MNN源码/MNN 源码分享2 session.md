---
title: MNN源码分析2 Session
index_img: https://s2.loli.net/2024/03/12/AUQ3MRt2ndakyup.png
tags: [模型部署,MNN]
date: 2024-03-012 18:42:30
categories: MNN 源码系列
published: true
comment: 'utterances'
category_bar: true
---
MNN 使用了ScheduleInfo，以及ScheduleConfig两个类来配置session。
有几个变量需要注意。

`allTensors` 是mnn中的信息，也是计算图中，所有的中间结果需要的tensor。

## schedule的创建

在MNN中tensor，是模型有向图里面的上下OP的中间计算结果。
在创建session时，首先会创建一个schedule，在schedule中初始化常量Tensor,创建Tensor，初始化他们的维度信息等。

```Cpp
bool Schedule::schedule(ScheduleInfo& scheduleInfo, const Net* net, const std::vector<ScheduleConfig>& configs, const RuntimeInfo& runtimeInfo) {
    if (nullptr == net->oplists()) {
        MNN_PRINT("Empty net for schedule\n");
        return false;
    }
    if (scheduleInfo.defaultBackend.get() == nullptr && scheduleInfo.allTensors.empty()) {
        // Const not init, init it
        BackendConfig defaultConfig;
        defaultConfig.flags = 4;
        scheduleInfo.defaultBackend.reset(runtimeInfo.second->onCreate(&defaultConfig));  // 通过runtimeInfo.second 创建默认的backend 给scheduleInfo
        ErrorCode code = NO_ERROR;
        initConstTensors(scheduleInfo.allTensors, net, scheduleInfo.defaultBackend.get(), code); //初始化常量Tensor，MNN.fbs 里面的OpType的Const
        if (NO_ERROR != code) {
            MNN_ERROR("Schedule Const init errorcode = %d\n", code);
            return false;
        }
    }
    bool valid = initTensors(scheduleInfo.allTensors, net); //为所有的tensor其分配dim，初始化。MNN中每个tensor都会有一个index，方便操作
    scheduleInfo.validForResize = valid;
    std::vector<std::shared_ptr<Tensor>>& allTensors = scheduleInfo.allTensors;
    std::vector<std::pair<Schedule::BackendCache, std::vector<Schedule::OpCacheInfo>>> result;  //初始化 用来存op

    for (auto& config : configs) {
        Backend::Info compute;
        compute.type      = getApprociateType(config);
        compute.numThread = config.numThread;
        if(config.type == MNN_FORWARD_AUTO) {
            if(compute.type == MNN_FORWARD_OPENCL || compute.type == MNN_FORWARD_METAL) {
                // AUTO set default gpu-mode MNN_GPU_TUNING_FAST
                compute.numThread = 16;
            }
        }
        compute.user      = config.backendConfig;
        auto oplists      = _scheduleUnit(net, config, allTensors);  //这里输出对应的op 执行顺序
        Schedule::BackendCache cache;
        cache.info = std::move(compute);
        result.emplace_back(std::make_pair(cache, std::move(oplists)));   //转入result
    }

    scheduleInfo.pipelineInfo = std::move(result); //存入pipline

    // get all used op's output, drop unused op, won't change op order. always insert all Input Ops
    std::vector<const Op*> oplists;
    {
        for (std::pair<Schedule::BackendCache, vector<Schedule::OpCacheInfo>>& pipeline : scheduleInfo.pipelineInfo) {
            for (auto& info : pipeline.second) {
                oplists.push_back(info.op);
            }
        }
    }
    // set tensors' input/output usage by oplists info
    setInputOutputForOps(allTensors, oplists, net->usage() == Usage_INFERENCE_STATIC);

    // add output index by config info and outputName
    std::unordered_map<std::string, int> tensorNameIndexMap;
    for (int i = 0; i < net->tensorName()->size(); ++i) {
        tensorNameIndexMap[net->tensorName()->Get(i)->str()] = i;
    }
    bool userSetOutput = false;
    //这里时对应着，比如你想打印一些中间tensor， 就需要在tensormap中根据name找到他们，并当成输出tensor。
    for (auto& config : configs) {
        userSetOutput = userSetOutput || (!config.saveTensors.empty());
        for (const auto& name : config.saveTensors) {
            auto iter = tensorNameIndexMap.find(name);
            if (iter != tensorNameIndexMap.end()) {
                auto t = allTensors[iter->second].get();
                if (TensorUtils::getDescribe(t)->usage == Tensor::InsideDescribe::NORMAL) {
                    TensorUtils::getDescribe(t)->usage = Tensor::InsideDescribe::OUTPUT;
                }
                scheduleInfo.outputTensor.insert(
                           std::make_pair(net->tensorName()->GetAsString(iter->second)->c_str(), t));
            } else {
                MNN_PRINT("Bad outputname: %s\n", name.c_str());
            }
        }
    }
    if (net->outputName()) {
        userSetOutput = userSetOutput || net->outputName()->size() >= 1;
        for (int i = 0; i < net->outputName()->size(); ++i) {
            std::string name = net->outputName()->Get(i)->str();
            auto iter = tensorNameIndexMap.find(name);
            if (iter != tensorNameIndexMap.end()) {
                auto t = allTensors[iter->second].get();
                if (TensorUtils::getDescribe(t)->usage == Tensor::InsideDescribe::NORMAL) {
                    TensorUtils::getDescribe(t)->usage = Tensor::InsideDescribe::OUTPUT;
                }
                scheduleInfo.outputTensor.insert(
                               std::make_pair(net->tensorName()->GetAsString(iter->second)->c_str(), t));
            }
        }
    }
    if (scheduleInfo.outputTensor.empty()) {
        userSetOutput = false;
    }
    // add input/output tensor to schedule's input/output
    //配置scheduleInfo 记录模型的输入 输出张量
    for (int index = 0; index < allTensors.size(); index++) {
        auto t = allTensors[index].get();
        auto usage = TensorUtils::getDescribe(t)->usage;  //usage NORMAL ,INPUT 
        if (usage == Tensor::InsideDescribe::INPUT) {
            scheduleInfo.inputTensors.insert(std::make_pair(net->tensorName()->GetAsString(index)->c_str(), t));
        }
        if (usage == Tensor::InsideDescribe::OUTPUT && (!userSetOutput)) {
            scheduleInfo.outputTensor.insert(
                       std::make_pair(net->tensorName()->GetAsString(index)->c_str(), t));
        }
    }
    // 是否是静态推理
    if (net->usage() == Usage_INFERENCE_STATIC) {  
        for (auto& pipInfo : scheduleInfo.pipelineInfo) {
            pipInfo.first.needComputeGeometry = false;   
            pipInfo.first.needComputeShape = false;
        }
    }
    // 设置flag needComputeGeometry需要计算 计算图buildConstantTensors
#ifndef MNN_BUILD_MINI
    for (auto iter = scheduleInfo.pipelineInfo.begin(); iter != scheduleInfo.pipelineInfo.end();) {
        if (!iter->first.needComputeGeometry) {
            // For static model don't need check const
            iter++;
            continue;
        }
        auto breakIndex = GeometryComputerUtils::buildConstantTensors(iter->second);
        if (breakIndex >= 0) {
            scheduleInfo.needInputContentForShape = true;
        }
#ifdef MNN_SEPERTE_SIZE
        if (breakIndex >= 0 && (breakIndex + 1) < iter->second.size()) {
            // Split oplist
            std::vector<Schedule::PipelineInfo> fuse;
            std::vector<Schedule::PipelineInfo> separate;
            fuse.insert(fuse.begin(), iter->second.begin(), iter->second.begin() + breakIndex + 1);
            separate.insert(separate.begin(), iter->second.begin() + breakIndex + 1, iter->second.end());
            oplists.clear();
            iter->second = std::move(separate);
            iter = scheduleInfo.pipelineInfo.insert(iter, std::make_pair(iter->first, fuse));
            iter++;
            iter++;
        } else {
            iter++;
        }
#else
        iter++;
#endif
    }
#endif
    return true;
}
```
配置scheduleInfo 配置模型的输入 输出张量的string，也就是在推理时，我们可以根据输入输出tensor的string来，初始化我们的输入，根据output的名字来读取结果。
从上面代码可以看到，如果输入是静态的化，也就是讲可以确定输入和输出的dim维度信息。那么就不需要在线计算你的中间tensor的大小。
## initTensor
```Cpp
bool initTensors(std::vector<std::shared_ptr<Tensor>>& tensors, const Net* net) {
    bool valid    = true;
    auto describes = net->extraTensorDescribe();
    std::vector<const TensorDescribe*> des(tensors.size());
    for (int i=0; i<tensors.size(); ++i) {
        // Init all tensor except for const
        if (tensors[i].get() == nullptr) {
            tensors[i].reset(new Tensor);  //虽然这里new 了一个tensor 但是 tensor 内部存储数据的内存 仍然没有分配
            TensorUtils::getDescribe(tensors[i].get())->index = i;
            // MNN_PRINT("initTensors create tensor:%p, index:%d, backend:%d\n", tensors[i].get(), i, TensorUtils::getDescribe(tensors[i].get())->backend);
        }
    }
    if (describes) {
        for (int i = 0; i < describes->size(); i++) {
            int index  = describes->GetAs<TensorDescribe>(i)->index();
            des[index] = describes->GetAs<TensorDescribe>(i);
        }
    }
    //设置Tensor的量化信息
    for (int i = 0; i < tensors.size(); ++i) {
        if (des[i] != nullptr && des[i]->quantInfo()) {    
            TensorUtils::getDescribe(tensors[i].get())->quantAttr.reset(new QuantAttr);
            auto quant   = TensorUtils::getDescribe(tensors[i].get())->quantAttr.get();
            quant->scale =  des[i]->quantInfo()->scale();
            quant->zero  =  des[i]->quantInfo()->zero();
            quant->min   =  des[i]->quantInfo()->min();
            quant->max   =  des[i]->quantInfo()->max();
            // Don't copy datatype, it can be set by backend
        }
    }
    // Set Input Tensor, if the type of input is not the same with ExtraTensorDescribe, use input parameter
    // 其实就是设置input 层的tensor，将input op的dim 维度信息 给Tensor
    // 对于输入图片是动态的 dim的后的h,w是-1 （猜想）
    for (int opIndex = 0; opIndex < net->oplists()->size(); ++opIndex) {
        auto op = net->oplists()->GetAs<Op>(opIndex);
        if (OpType_Input == op->type()) {
            MNN_ASSERT(nullptr != op->outputIndexes());
            MNN_ASSERT(op->outputIndexes()->size() == 1);
            auto index      = op->outputIndexes()->data()[0];
            auto tensor     = tensors[index].get();
            auto& tb        = tensor->buffer();
            auto inputParam = op->main_as_Input();
            if (auto idims = inputParam->dims()) {
                for (int i = 0; i < idims->size(); ++i) {
                    int extent = idims->data()[i];
                    // dim-0 is batch(when input batch is -1, set it to be 1, ignore other dim)
                    if (i == 0 && extent == -1) {
                        extent = 1;
                    }
                    if (extent < 0) {
                        valid = false;
                    }
                    tb.dim[i].extent = extent;
                }
                tb.dimensions = idims->size();
            } else {
                tb.dimensions = 0;
            }
            tensor->setType(inputParam->dtype());
            TensorUtils::getDescribe(tensor)->dimensionFormat = inputParam->dformat();
            TensorUtils::setLinearLayout(tensor);
        }
    }
    if (net->usage() != Usage_INFERENCE_STATIC) {
        return valid;
    }
    //如果模型时静态的就直接初始化Tensor的shape
    // static model will set all tensors' shape
    //动态的话 resizeTensor中初始化，因为到那个时候Tensor的shape才能根据input来确定
    for (int i = 0; i < describes->size(); i++) {
        int index  = describes->GetAs<TensorDescribe>(i)->index();
        des[index] = describes->GetAs<TensorDescribe>(i);
    }
    for (int i = 0; i < tensors.size(); ++i) {
        if (TensorUtils::getDescribe(tensors[i].get())->usage != Tensor::InsideDescribe::NORMAL) {
            // Const / Trainable Shape has been inited
            continue;
        }
        auto blob = des[i]->blob();
        auto& tb = tensors[i]->buffer();
        if (auto idims = blob->dims()) {
            for (int d = 0; d < idims->size(); d++) {
                tb.dim[d].extent = idims->Get(d);
            }
            tb.dimensions = idims->size();
        } else {
            tb.dimensions = 0;
        }
        tensors[i]->setType(blob->dataType());
    }
    for (int i = 0; i < tensors.size(); ++i) {
        auto blob                                                   = des[i]->blob();
        TensorUtils::getDescribe(tensors[i].get())->dimensionFormat = blob->dataFormat();
        if (auto regions = des[i]->regions()) {
            auto& regs = TensorUtils::getDescribe(tensors[i].get())->regions;
            TensorUtils::getDescribe(tensors[i].get())->memoryType = Tensor::InsideDescribe::MEMORY_BACKEND;
            regs.reserve(regions->size());
            for (int r = 0; r < regions->size(); r++) {
                auto region = regions->GetAs<Region>(r);
                Tensor::InsideDescribe::Region reg;
                reg.origin     = tensors[region->origin()].get();
                reg.src.offset = region->src()->offset();
                reg.dst.offset = region->dst()->offset();
                for (int d = 0; d < 3; d++) {
                    reg.size[d]       = region->size()->data()[d];
                    reg.src.stride[d] = region->src()->stride()->data()[d];
                    reg.dst.stride[d] = region->dst()->stride()->data()[d];
                }
                regs.emplace_back(std::move(reg));
            }
        }
    }
    return valid;
}

```
## 计算图oplist的构建
```Cpp
static vector<Schedule::OpCacheInfo> _scheduleUnit(const Net* net, const ScheduleConfig& configs,
                                                    const vector<shared_ptr<Tensor>>& allTensors) {
    vector<Schedule::OpCacheInfo> oplists;
    vector<const Op*> ops;
    generateScheduleGraph(ops, net, configs, allTensors); //import 构建计算图
    initPipelineInfosFromOps(oplists, ops, allTensors); //import 初始化pipline 后续计算全部根据pipline
    return oplists;
}
```
schedle的构造函数中的，`_scheduleUnit`函数很重要。`generateScheduleGraph`用来生成`OPlists`,`OPlists`是个有向无环图，记录着`OP`的执行顺序。下面是`generateScheduleGraph`的一部分。
```Cpp
static void generateScheduleGraph(vector<const Op*>& ops, const Net* net, const ScheduleConfig& configs,
                                  const vector<shared_ptr<Tensor>>& allTensors) {
    if (configs.path.inputs.empty() && configs.path.outputs.empty()) {
        // Use Default Linear schedule
        ops.clear();
        ops.reserve(net->oplists()->size());
        for (int i = 0; i < net->oplists()->size(); ++i) {
            auto op = net->oplists()->GetAs<Op>(i);
            ops.emplace_back(op);  //ops[4]->type()  MNN::OpType_ConvolutionDepthwise
        }
        return;
    }
```
其中主要做的是，将mnn的flatbuffer中的net中的oplist，通过`net->oplists()->GetAs<Op>`读到程序里面。这样`OPlist`就基本构建好了。

## 初始化OP的输入，输出地址
```Cpp
static vector<Schedule::OpCacheInfo> _scheduleUnit(const Net* net, const ScheduleConfig& configs,
                                                    const vector<shared_ptr<Tensor>>& allTensors) {
    vector<Schedule::OpCacheInfo> oplists;
    vector<const Op*> ops;
    generateScheduleGraph(ops, net, configs, allTensors); //import 构建计算图
    initPipelineInfosFromOps(oplists, ops, allTensors); //import 初始化pipline 后续计算全部根据pipline
    return oplists;
}

```
在`initPipelineInfosFromOps`中
```Cpp
void initPipelineInfosFromOps(std::vector<Schedule::OpCacheInfo>& infos, std::vector<const Op*>& ops, const std::vector<std::shared_ptr<Tensor>>& allTensors) {
    for (const Op* op : ops) {
        // MNN_PRINT("initPipelineInfosFromOps, op type:%s, op name:%s\n", EnumNameOpType(op->type()), op->name()->c_str());

        Schedule::OpCacheInfo opInfo;
        opInfo.op = op;
        if (nullptr != op->outputIndexes()) {
            auto data = op->outputIndexes()->data();
            for (int j = 0; j < op->outputIndexes()->size(); ++j) {
                opInfo.outputs.push_back(allTensors[data[j]].get());  // 这里设置OP 的输出tensor
            }
        }
        if (nullptr != op->inputIndexes()) {
            auto data = op->inputIndexes()->data();
            for (int j = 0; j < op->inputIndexes()->size(); ++j) {
                opInfo.inputs.push_back(allTensors[data[j]].get());  // 这里设置OP 的输入tensor
            }
        }
        if (needComputeOp(op)) {
            infos.emplace_back(std::move(opInfo));
        }
    }
}

```
从这个函数里面可以看出，`tensor`的`input`和`output`被存入到了`opinfo`里面。这样op就能找到他的输入地址，和输出地址。
## 资源的创建
需要强调的是，如果输入是动态的也就是input比如是图片，有的大有的小，务必在推理时，resizeTensor和resizeSession。
```Cpp
    net->resizeTensor(input_cpu,input_cpu->shape());
    net->resizeSession(session);
```

在`session->resize()`其中最重要的是`iter->allocMemory(firstMalloc, forbidReplace)`,足足有两百行。
我们现在有了`oplist`，和`tensor`。 但是`tenor`的物理内存仍然没有创建，Op对应的实现也没有创建比如卷积类等。
`allocMemory`就是实例化模型中卷积，等等操作，以及`tensor`的内存分配问题。
### Executions的创建
```Cpp
static ErrorCode _createExecutions(Schedule::PipelineInfo &mInfo) {
  auto &mBackend = mInfo.first.cache.first;
  auto &mBackupBackend = mInfo.first.cache.second;
  for (auto &info : mInfo.second) {
    auto &buffer = info.executeBuffer;
    // MNN_PRINT("before resize, mInfo.second size:%lu, command size:%lu,op
    // type:%s, op name:%s\n", mInfo.second.size(), buffer.command.size(),
    // EnumNameOpType(info.op->type()), info.op->name()->c_str());
    for (auto &iterP : buffer.command) {
      auto &iter = *iterP;
      // Create exe
      // Find Cache
      bool cached = false;
      if (nullptr == iter.execution) {
        /** Cache origin execution for fast resize*/
        auto exeIter = info.executionCache.find(iter.op);
        if (exeIter != info.executionCache.end()) {
          iter.execution = exeIter->second;
          cached = true;
        }
      }
      //在这个地方会根据OP类型和benckend的类型去实例化 卷积什么的OP
      if (nullptr == iter.execution) {  
        iter.execution.reset(
            mBackend->onCreate(iter.inputs, iter.outputs, iter.op));
      }
      //如果上面的代码没有成功，也就是说上面的代码不支持这个操作，则使用备用的banckend
      if (nullptr == iter.execution) {
        // Try Backup
        iter.execution.reset(
            mBackupBackend->onCreate(iter.inputs, iter.outputs, iter.op));
        if (nullptr == iter.execution) {
          if (mInfo.first.reportError) {
            MNN_ERROR("Create execution error : %d\n", iter.op->type());
          }
          return NOT_SUPPORT;
        }
      }
      // invalid means memory alloc failed
      if (!iter.execution->valid()) {
        iter.execution = nullptr;
        iter.execution = nullptr;
        return OUT_OF_MEMORY;
      }
      if ((!cached) && iter.buffer == nullptr &&
          (iter.op->type() != OpType_Raster) &&
          (iter.op->type() != OpType_BinaryOp)) {
        info.executionCache.insert(std::make_pair(iter.op, iter.execution));
      }
    }
  }
  return NO_ERROR;
}
```
 在这个地方会根据OP类型和benckend的类型去实例化 卷积什么的OP，如果上面的代码没有成功，也就是说上面的代码不支持这个操作，则使用备用的banckend。`mBackend->onCreate()`是虚函数，根据OP类型来动态创建。
 ### 内存分配

内存分配有三个步骤

```Cpp
      mBackend->onResizeBegin();
       auto curBackend = iter.execution->backend();
      if (mAllocInput) {
        for (auto t : iter.workInputs) {  
          auto allocRes = _allocTensor(t, curBackend, mOutputStatic);
          if (!allocRes) {
            return OUT_OF_MEMORY;
          }
        }
      }
      {
        for (auto t : iter.workOutputs) {
          auto res = _allocTensor(t, curBackend, mOutputStatic);
          if (!res) {
            return OUT_OF_MEMORY;
          }
        }
      }
      mBackend->onResizeEnd();
```
`_allocTensor` 会分别调用backend的 `OnAccquire`来实现。比如`OpenCLBackend::onAcquire` 之后在介绍backend时再详细介绍。