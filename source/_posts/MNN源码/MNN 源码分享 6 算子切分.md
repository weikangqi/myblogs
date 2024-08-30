---
title: MNN源码分析4 mnn的图计算
index_img: https://s2.loli.net/2024/03/12/AUQ3MRt2ndakyup.png
tags: [模型部署,MNN]
date: 2024-03-14 18:42:30
categories: MNN 源码系列
published: true
comment: 'utterances'
---
# MNN源码分析3 mnn的图计算 分析 （slice 切片为例）
为了探究MNN中的Tensor的切分，我定义了下面的pytorch代码，想查看对应到MNN中，是如何实现的。
```python
import torch
import torch.nn as nn

class CustomModel(nn.Module):
    def __init__(self, input_shape):
        super(CustomModel, self).__init__()
        self.input_shape = input_shape
        self.conv1 = nn.Conv2d(in_channels=input_shape[1], out_channels=16, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)


    def forward(self, x,which):
        x = self.conv1(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.conv3(x)
        
        # 计算切分的大小
        h1 = x.size(2) //3

        x1 = x[:, :, :h1, :]
        x2 = x[:, :, h1:, :]
        print(x1.shape)

        x1 = self.conv4(x1)
        x2 = self.conv4(x2)
        x1 = self.conv5(x1)
        x2 = self.conv5(x2)
        
        x = torch.cat((x1, x2), dim=2)
        print(x.shape)
        x = self.conv6(x)
        return x


input_shape = (1,3, 244, 244)
model = CustomModel(input_shape)

dummy_input = torch.randn( 1, 3, 244, 244)
torch.onnx.export(model, dummy_input, "my.onnx", verbose=True)

```

这样就将一个Tensor按照H维度切成了两个部分。分别计算最后合成一个Tensor，继续计算。
## ONNX 计算图
![my.onnx](https://s2.loli.net/2024/03/14/58rP2Hi1NYJhZlU.png)
可以看到ONNX就是按照pytorch的定义导出了整个计算图。并没有进行什么优化之类的。
## MNN 计算图
![my.mnn](https://s2.loli.net/2024/03/14/adJLDSe2CmxKGXz.png)

MNN会converTensor，将NC4HW4的Tensor转成NCHW的实现。在pipline的中oplist中也有这些子算子的op，但是这些op都没有执行，而是在Raster的op里计算的,Raster用来计算所有形变算子如 slice, concat, reshape, broadcast。
需要注意的是，这样虽然也能运行，但是不够简洁。还是需要使用MNNConver的保存静态图来进行优化。
```bash
./MNNConvert -f  ONNX --modelFile ../my.onnx --saveStaticModel    --MNNModel new.mnn
```
这样导出的图如下：
![new.mnn](https://s2.loli.net/2024/03/15/7vz5YxWku2dQgPm.png)
## TODO
- [ ] 分析Raster 实现
