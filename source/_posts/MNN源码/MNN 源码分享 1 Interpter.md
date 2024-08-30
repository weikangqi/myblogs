---
title: MNN源码分析1 Interpreter
index_img: https://s2.loli.net/2024/03/12/AUQ3MRt2ndakyup.png
tags: [模型部署,MNN]
date: 2024-03-15 18:42:30
categories: MNN 源码系列
published: true
comment: 'utterances'
---
# 模型加载
## 从磁盘到内存
首先从Interpreter,出发探究运行前初始化的操作.
`MNN::Interpreter::createFromFile(yourmodel))`
会执行`static Content *loadModelFile(const char *file)` 用来加载mnn 文件, 其中使用`FileLoader` 来封装`fopen`,`fread`等操作.
```Cpp
bool FileLoader::read() {
    auto block = MNNMemoryAllocAlign(gCacheSize, MNN_MEMORY_ALIGN_DEFAULT);
    if (nullptr == block) {
        MNN_PRINT("Memory Alloc Failed\n");
        return false;
    }
    auto size  = fread(block, 1, gCacheSize, mFile);
    mTotalSize = size;
    mBlocks.push_back(std::make_pair(size, block));

    while (size == gCacheSize) {
        block = MNNMemoryAllocAlign(gCacheSize, MNN_MEMORY_ALIGN_DEFAULT);
        if (nullptr == block) {
            MNN_PRINT("Memory Alloc Failed\n");
            return false;
        }
        size = fread(block, 1, gCacheSize, mFile);
        if (size > gCacheSize) {
            MNN_PRINT("Read file Error\n");
            MNNMemoryFreeAlign(block);
            return false;
        }
        mTotalSize += size;
        mBlocks.push_back(std::make_pair(size, block));
    }

    if (ferror(mFile)) {
        return false;
    }
    return true;
}
```
首先申请4096大小的内存块,`mBlocks.push_back(std::make_pair(size, block))` 将内存块大小,当然模型肯定是大于4096 Byte的,申请成功后,将继续以4096 Byte大小向操作系统申请内存。并将申请到的内存大小，以及地址放入mBlocks中。当文件读到末尾时，`size <= gCacheSize` 这样就结束循环。于是模型就从磁盘读入到了内存中。每一次读`gCacheSize`可能是因为不知道文件的大小，无法确定要分配多少内存。
```Cpp
class MNN_PUBLIC FileLoader {
public:
    FileLoader(const char* file);  // mFile = fopen(file, "rb")

    ~FileLoader();

    bool read();
    
    static bool write(const char* filePath, std::pair<const void*, size_t> cacheInfo);

    bool valid() const {
        return mFile != nullptr;
    }
    inline size_t size() const {
        return mTotalSize;
    }

    bool merge(AutoStorage<uint8_t>& buffer); 

    int offset(int64_t offset);  //封装了 fseek

    bool read(char* buffer, int64_t size); //封装了 fread
private:
    std::vector<std::pair<size_t, void*>> mBlocks; // 存放申请的内存块
    FILE* mFile                 = nullptr;
    static const int gCacheSize = 4096;  //一次性 向操作系统索要的内存
    size_t mTotalSize           = 0;
    const char* mFilePath       = nullptr;
};
```
## 从内存块到Content
```Cpp
--- loadModelFile
  auto net = new Content;
  bool success = loader->merge(net->buffer);
---

bool FileLoader::merge(AutoStorage<uint8_t>& buffer) {
    buffer.reset((int)mTotalSize);  // malloc mTotalSize的小的连续内存，并将指针赋予buffer的mData，大小赋予mSize，相当于net的存模型数据的空间。
    if (buffer.get() == nullptr) {
        MNN_PRINT("Memory Alloc Failed\n");
        return false;
    }
    auto dst   = buffer.get();  //dst buffer->mData
    int offset = 0;
    for (auto iter : mBlocks) {
        ::memcpy(dst + offset, iter.second, iter.first);   //拷贝mBlocks中内存，合并到一个net的大内存中。
        offset += iter.first;
    }
    return true;
}

--- loadModelFile
    loader.reset(); //将临时的内存释放
    return net;
---

```
这样模型就从FileLoad 到了Content中
### Content的结构
```Cpp
struct Content {
  AutoStorage<uint8_t> buffer; //模型的物理地址
  const Net *net = nullptr;
  std::vector<std::unique_ptr<Session>> sessions;
  std::map<Tensor *, const Session *> tensorMap;
  Session::ModeGroup modes;
  AutoStorage<uint8_t> cacheBuffer;
  std::string cacheFile;
  std::mutex lock;
  size_t lastCacheSize = 0;
  std::string bizCode;
  std::string uuid;
  std::string externalFile;
#ifdef MNN_INTERNAL_ENABLED
  std::map<std::string, std::string> basicLogginData;
  std::map<const Session *, std::tuple<int, int>> sessionInfo;
#endif
};
```

## 反序列化 解析模型
MNN的mnn文件，实际上是使用Google的FlatBuffers库序列化后的文件，这样模型占具内存的空间就会小，读模型到系统时，内存需求就小一点。
```Cpp
net->net = GetNet(net->buffer.get());  

inline const MNN::Net *GetNet(const void *buf) {
  return flatbuffers::GetRoot<MNN::Net>(buf);
}

```
这样`net->net` 就可以通过 flatbuffers库解析模型，如`net->net->oplists()`。

```Cpp
return new Interpreter(net);
```
这样一个Interpreter 就初始化了。