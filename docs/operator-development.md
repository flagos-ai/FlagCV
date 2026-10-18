# 算子开发指南

以 `bitwise_and` 为参考实现。新增算子需要改动的位置是固定的。

## 1. 竖切清单

| 步骤 | 文件 | 内容 |
| --- | --- | --- |
| 1 | `include/flagcv/types.h` | 需要新枚举时才改 |
| 2 | `include/flagcv/flagcv.h` | 算子类型条目与典型 setter |
| 3 | `include/flagcv/imgproc.hpp` | 即时模式签名 |
| 4 | `src/imgproc/<family>.cpp` | GpuMat → ImageSpec，组装 OperationSpec |
| 5 | `src/pipeline/lowering/<family>.cpp` | 语义校验 + 生成描述 |
| 6 | `src/pipeline/lowering/dispatch.cpp` | 注册一行分发 |
| 7 | `kernels/common/<family>.py` | Triton kernel |
| 8 | `kernels/registry.json` | 注册 operation → source/functions |
| 9 | `tests/` + `reference/cpu/` | 验证 |

校验逻辑一律放在步骤 5，不要放在步骤 4 —— 否则即时模式与管线模式会在接受什么
操作数上漂移。

## 2. 行为规范

### 2.1 mask 语义是"不写入"，不是"写零"

`mask` 为零处目标像素必须保持原值。正确做法是把 mask 折进 store 谓词：

```python
store_predicate = in_range & (mask_values != 0)
tl.store(destination_pointer + offsets, result, mask=store_predicate)
```

写成 `tl.where(mask, result, 0)` 会把未选中像素写成零，语义不同。因此使用 mask 时
调用方必须传入已初始化的目标图像。

### 2.2 整数算子使用饱和取整

整数路径的算术必须显式 clamp 到目标深度值域，`255 + 1` 得 `255` 而非溢出回绕，
舍入方式为就近取整。不能依赖 C 的隐式转换。

### 2.3 类型不匹配应当报错，不应静默重解释

按位算子只接受整数深度，浮点输入应拒绝而非按位重解释。

### 2.4 并列情况的选择必须固定

极值定位、特征点筛选在数值并列时必须显式规约为确定结果（通常取扫描顺序最先者），
否则并行实现不可复现。测试应包含含并列值的用例。

## 3. pitched 寻址

所有 load/store 必须经由字节行距：

```python
offsets = y_offsets[:, None] * row_pitch + x_offsets[None, :]
```

通道轴折进列轴而非单独循环 —— 像素的各通道在内存中本就连续，折进列轴后一个 tile
天然合并访问，也不必把通道数做成 `constexpr`（否则 1/3/4 通道各编一份）。mask 是
单通道，用 `x_offsets // channels` 换算回像素坐标。

每个操作数持有独立 pitch，源与目标不要求相同。

## 4. 标量操作数走设备缓冲

标量不应编译进 kernel 常量，否则换一个标量值就缓存未命中。做法是作为长度等于通道
数的设备缓冲传入，这样同一编译产物可服务任意标量值。

## 5. 容差

功能测试的容差按算子类别定，不使用统一阈值。详见 `docs/numerical-tolerance.md`。

## 6. 平台覆盖

通用 kernel 应能让所有平台跑通。平台只在有明确性能收益时才在
`backends/<platform>/kernels/` 覆盖并登记。平台登记是排他的：登记后不会在失败时
静默回退通用实现。

## 7. 平台覆盖的具体做法

以 `bitwise_and` 为例。通用 kernel 每 lane 搬一个分量,对 8 位图像浪费四分之三
带宽 —— 一个字节 load 占满一个 lane。NVIDIA 侧
[backends/nvidia/kernels/bitwise.py](../backends/nvidia/kernels/bitwise.py) 把四个
连续 8 位分量重解释成一个 int32 做打包运算。

**按位运算适合打包,算术运算不适合。** 按位运算逐 bit 独立,打包是精确的,不会有
进位或舍入跨越分量边界。加减乘除则会,所以这个技巧不能照搬到 arithmetic kernel。

打包路径需要行首 4 字节对齐且每行分量数是 4 的倍数。这两个约束由
[backends/nvidia/dispatch/bitwise.cpp](../backends/nvidia/dispatch/bitwise.cpp)
在 build 期检查,不满足就回落通用 kernel。**必须在 dispatch 期检查**,因为平台登记
是排他的:一旦 registry 登记了该算子,运行期失败不会静默回退。

带 mask 时打包有个额外问题:一个 int32 跨四个分量,可能横跨 mask 值不同的像素,
不能整字 predicate。平台 kernel 的做法是读回目标原值,按字节混合打包结果与原值。
这比通用 kernel 的整字谓词复杂,但语义一致 —— mask 为零处仍然字节级保持原内容。
