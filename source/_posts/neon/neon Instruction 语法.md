---
title: ARM neon Instruction  语法
index_img: https://s2.loli.net/2024/03/18/JUWMRmv6fEaBHAu.png
tags: [neon,arm,simd]
date: 2024-03-18:42:30
categories: neon 学习系列
published: true
excerpt: 学习arm neon,最终实现一个FFT为目标。
comment: 'utterances'
---
`V{<mod>}<op> {<shape>}{<cond>}{.<dt>} <dest1>{,<dest2>},<src1>{, <src2>}`
## NEON data types
![20240320100728](https://s2.loli.net/2024/03/20/LAVQ9wC6T4RnlHy.png)

## Instruction modifiers
对于某些指令，可以指定一个修饰符，用以改变操作的行为。
| 修饰符 | 行为 | 例子 | 描述 |
| --- | --- | --- | --- | 
| None | Basic opeartion | VADD.16 Q0, Q1,Q2| 结果不会被修改|
| Q | Saturation (饱和)| VQADD.S16 D0, D2,D3 | 结果向量中的每个元素如果超出可表示范围，则设为最大值或最小值。范围取决于元素的类型（位数和符号）。如果任何通道发生饱和，浮点状态与控制寄存器（FPSCR）中的粘性 QC 位将被设置。|
| H | Halved (减半) | VHADD.S16 Q0, Q1,Q4 | 每个元素向右移动一位（实际上是截断后的除以二）。VHADD 可用于计算两个输入的平均值。|
| D | Doubled before saturation | VQDMULL.S16 Q0, D1, D3 | 这在以 Q15 格式相乘时常常需要，需要额外加倍以将结果转换为正确的形式。|
| R | Rounded (四舍五入) |  VRSUBHN.I16 D0, Q1, Q3 | 该指令对结果进行四舍五入，以纠正截断所引起的偏差。这相当于在截断之前向结果加上0.5。|

## Instruction shape
结果向量和操作数向量具有相同数量的元素。然而，结果中元素的数据类型可以与一个或多个操作数中的元素的数据类型不同。因此，结果的寄存器大小也可能与一个或多个操作数的寄存器大小不同。寄存器大小之间的这种关系由形状描述。对于某些指令，您可以指定一个形状。NEON数据处理指令通常有普通、长、宽、窄和饱和变体。
###  None specified
操作数和结果都是相同的宽度。
```Cpp
VADD.I16 Q0, Q1, Q2
```

### Long - L
```Cpp
 VADDL.S16 Q0, D2, D3
```
长指令通常操作双字向量并产生四倍字长向量。结果元素的宽度是操作数元素宽度的两倍。通过在指令后附加 L 来指定长指令。Q是D的两倍。

![20240320094905](https://s2.loli.net/2024/03/20/3qcnPt1CWsS9Jya.png)
### Narrow - N 
```Cpp
VADDHN.I16 D0, Q1, Q2
```
操作数具有相同的宽度。每个结果元素中的位数是每个操作数元素中位数的一半。
![20240320100137](https://s2.loli.net/2024/03/20/JrysBQZhAKGoE39.png)

### Wide - W
```Cpp
VADDW.I16 Q0, Q1, D4
```
结果和操作数的宽度是第二个操作数的两倍。Q0,Q1是D4的两倍

![20240320100339](https://s2.loli.net/2024/03/20/Hn7rY2LcRS9ZViK.png)