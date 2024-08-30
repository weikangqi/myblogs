



##  基本数据结构

```Cpp
struct MemNode {
public:
    MemNode(size_t s) : size(s) {}
    ~MemNode() {}
    size_t size = 0, offset = 0;
    void* base = nullptr; // 数据地址
    bool usage = true; // 是否被使用
    MemNode *left = nullptr, *right = nullptr;
    
    std::vector<MemNode*> children;
    std::vector<Tensor*> tensors;
};
```

```Cpp
struct MemChunk {
public:
    MemChunk() = default;
    MemChunk(void* base, size_t offset = 0) : first(base), second(offset) {}
    MemChunk(std::pair<void*, size_t> pointer) : first(pointer.first), second(pointer.second) {}
    MemChunk(MemNode* node) : mNode(node) {}
    ~MemChunk() = default;
    MemChunk operator+ (size_t offset);  // this->second += offset
    void* base() const;
    size_t offset() const;
    bool invalid() const;
    void attach(Tensor* tensor);
    uint8_t* ptr() const {    // 获取数据指针 
        if (mNode) {
            return static_cast<uint8_t*>(mNode->base) + mNode->offset + second;
        }
        return static_cast<uint8_t*>(first) + second;
    }
public:
    void* first = nullptr;   //相当于MemNode的base
    size_t second = 0;  //想当于 MemNode->offset + second
private:
    MemNode* mNode = nullptr;
    friend class DeferBufferAllocator;
    friend class EagerBufferAllocator;
    friend class DefaultAllocator;
};
```



## BufferAllocator

```Cpp
class MNN_PUBLIC BufferAllocator : public NonCopyable {
public:
    class Allocator {
    public:
        Allocator() = default;
        virtual ~ Allocator() = default;
        virtual MemChunk onAlloc(size_t size, size_t align) = 0;
        virtual void onRelease(MemChunk chunk) = 0;
        static std::shared_ptr<Allocator> createDefault();
        static std::shared_ptr<Allocator> createRecurse(BufferAllocator* parent);
    };
    BufferAllocator() = default;
    virtual ~BufferAllocator() = default;
    virtual MemChunk alloc(size_t size, bool separate = false, size_t align = 0) = 0;
    virtual bool free(MemChunk chunk) = 0;
    virtual void release(bool allRelease = true) = 0;
    virtual size_t totalSize() const = 0;
    virtual void barrierBegin() {}
    virtual void barrierEnd() {}
    virtual void beginGroup() {}
    virtual void endGroup() {}
    virtual void reset() {}
    virtual ErrorCode compute();
};
```

