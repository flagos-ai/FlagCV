# 添加算子

两件事，分开做：**加一个对标 OpenCV 的接口**，和**加一个 Triton kernel**。
一个新算子通常两件都要做；给已有算子换平台实现只做第二件。

下面以 `bitwise_and` 为例（仓库里已经没有它了，这里只是示范）。

## 一、添加新接口

### 1. 声明

接口按 OpenCV 的模块归属放：`core` 放逐元素运算与基础变换，`imgproc` 放滤波、
重采样、色彩空间。查 `opencv2/core.hpp` 或 `opencv2/imgproc.hpp` 里该函数的签名，
照抄参数顺序和默认值，再按 FlagCV 的约定改两处。

写进 `include/flagcv/core.hpp`：

```cpp
void bitwise_and(Handle& handle, const GpuMat& src1, const GpuMat& src2,
                 const GpuMat& dst, const GpuMat& mask = GpuMat(),
                 Stream stream = nullptr);
```

两处改动：`handle` 领头，因为它带着后端和 kernel 缓存；`stream` 收尾且有默认值，
让常见调用读起来还像 OpenCV。`dst` 传 const 引用，因为 `GpuMat` 是非拥有视图，
改的是它指向的显存，不是它自己。

OpenCV 支持的形态要一次补齐：`bitwise_and` 有 array⊕array 和 array⊕scalar 两种，
漏掉后者就会出现"kernel 写了但没有入口能走到"的死代码。

### 2. Lowering

`src/lowering/bitwise.cpp`，一个算子族一个文件：

```cpp
LoweredOp lower_bitwise(const GpuMat& src1, const GpuMat& src2,
                        const GpuMat& dst, const GpuMat& mask, int mode) {
  /* to_spec() takes (image, name) in that order and rejects empty operands. */
  const ImageSpec s1 = to_spec(src1, "src1");
  const ImageSpec s2 = to_spec(src2, "src2");
  const ImageSpec d = to_spec(dst, "dst");

  /* require_same_layout() compares a pair, so chain it across the operands. */
  require_same_layout(s1, s2);
  require_same_layout(s1, d);
  if (!mask.empty()) require_mask(to_spec(mask, "mask"), d);

  LoweredOp op;
  op.kernel = "bitwise_binary";
  op.add_image(s1);
  op.add_image(s2);
  op.add_image(d);
  if (!mask.empty()) op.add_image(to_spec(mask, "mask"));
  op.add_param("mode", mode);
  op.add_param("has_mask", mask.empty() ? 0 : 1);
  return op;
}
```

校验放这层，不放入口函数。`bitwise_and/or/xor/not` 是四个入口共一个族，校验写在入口
里早晚会互相漂移。

行距不用手写：`to_spec()` 里 `ImageSpec::row_stride` 已经从 `GpuMat::row_pitch()`
换算过了。**公开 API 的 pitch 是字节，kernel 参数是元素**，换算只在
`GpuMat::row_stride()` 一个地方发生，别在别处再算一次。

### 3. 入口实现

`src/core/bitwise.cpp`：

```cpp
void bitwise_and(Handle& handle, const GpuMat& src1, const GpuMat& src2,
                 const GpuMat& dst, const GpuMat& mask, Stream stream) {
  const LoweredOp op = lower_bitwise(src1, src2, dst, mask, kModeAnd);
  std::vector<void*> pointers{src1.data(), src2.data(), dst.data()};
  if (!mask.empty()) pointers.push_back(mask.data());
  launch(handle, op, pointers, stream);
}
```

没有中央 dispatch，入口自己知道自己属于哪个族，直接调对应 lowering。

### 4. 注册与测试

- `src/CMakeLists.txt` 的 `add_library` 里加 `core/bitwise.cpp` 和
  `lowering/bitwise.cpp`
- `tests/CMakeLists.txt` 里加 `flagcv_add_test(bitwise)`，写 `tests/test_bitwise.cpp`
- 按 `numerical-tolerance.md` 选容差级别。按位运算是 A 级，必须逐字节相同

## 二、添加 Triton kernel

### 1. 通用 kernel 还是平台 kernel

默认写通用的，放 `kernels/common/<family>.py`。只有当某平台需要本地化实现
（厂商库更快、或硬件约束不同）时才在 `backends/<platform>/kernels/` 下再写一份。

两级 registry 按顺序查：先 `backends/<platform>/kernels/registry.json`，
再 `kernels/registry.json`。

**平台注册是排他的。** 平台一旦声明了某算子，那边失败就直接报给调用方，不会悄悄
回落到通用 kernel。所以平台 kernel 的所有约束（depth、通道数、对齐）必须在该平台
`compiler.py` 的 `identify()` 里编译前就查掉，而不是编译中才发现。

### 2. 写 kernel

`kernels/common/bitwise.py`：

```python
@triton.jit
def bitwise_binary_kernel(
    src1_ptr, src2_ptr, dst_ptr, mask_ptr,
    width, height, channels,
    src1_row_stride, src2_row_stride, dst_row_stride, mask_row_stride,
    mode: tl.constexpr, has_mask: tl.constexpr,
    BLOCK_X: tl.constexpr, BLOCK_Y: tl.constexpr,
):
    ...
    store_predicate = in_bounds
    if has_mask:
        store_predicate = store_predicate & (tl.load(mask_ptr + ...) != 0)
    tl.store(dst_ptr + dst_offsets, result, mask=store_predicate)
```

三条硬约束：

**行距是元素，不是字节。** 指针按像素深度定型，指针算术就按元素前进。
`y * row_stride + x` 里的 `row_stride` 必须是元素数。写成字节的话 8 位能跑对，
16 位差 2 倍、32 位差 4 倍 —— 这是老代码里真实踩过的坑。

**mask 折进 store 谓词。** mask 的语义是"不写"，不是"写 0"。被 mask 排除的目标像素
要保持原字节不变，所以用 `mask=store_predicate`，不要用
`tl.where(mask, result, 0)`。带 mask 的调用因此要求 `dst` 已初始化。

**标量当设备 buffer 传。** 标量操作数传成 `channels` 个元素的设备 buffer，不要编译
成常量。否则改一次标量值就触发一次重编译。只有元素个数进 cache key，值永远不进。

`BLOCK_X`/`BLOCK_Y` 这类留给平台 `compiler.py` 填，kernel 里只声明。

### 3. 登记

`kernels/registry.json`：

```json
{ "platform": "common",
  "kernels": [
    { "name": "bitwise_binary",
      "source": "common/bitwise.py",
      "function": "bitwise_binary_kernel" }
  ] }
```

`name` 就是 lowering 里填的 `op.kernel`。

### 4. 平台特化（可选）

在 `backends/<platform>/compiler.py` 里：

- `identify()` 回答"这个 op 我接不接"，并在这里把所有约束查完
- `compile()` 选 block 大小、编译、产出 artifact

然后在 `backends/<platform>/kernels/registry.json` 里登记。调优参数放
`backends/<platform>/tuning/<family>.yaml`。

core 库不链 Python、不链厂商工具链，编译在独立进程里（`compiler/`）完成，
plugin 只负责加载产物和 launch。plugin 不能 include core 的内部头文件 ——
它拿到的是纯数据，这样 core 加字段不会破 ABI。
