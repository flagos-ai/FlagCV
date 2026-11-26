# 添加 Triton 算子及 C++ 调用链

新增 Triton 算子时，除 Python 调用入口外，还需要接入 C++ 接口、lowering、
kernel registry 和编译计划。已有 Python/Triton 实现也按这条路径补齐。
C++ 接口与内存、stream 的约定见 [C++ operator interfaces](cpp-operators.md)。

## 调用链与文件位置

```text
include/flagcv/{core,imgproc}.hpp       声明公开接口
  -> src/{core,imgproc}/<family>.cpp   调用对应 lowering，传设备指针和 stream
  -> src/lowering/<family>.cpp        校验操作数，构造 LoweredOp
  -> src/runtime/runtime.cpp          查缓存、解析 registry、请求编译、加载和 launch
       -> compiler/flagcv_codegen/main.py
       -> backends/<platform>/compiler.py
       -> compiler/flagcv_codegen/providers.py   选择调优参数与 grid
       -> compiler/flagcv_codegen/triton_build.py
       -> kernels/common/<family>.py            编译 @triton.jit 函数
  -> backends/<platform>/backend.cpp  加载编译产物，在调用方 stream 上执行
```

普通算子复用 runtime 和 backend plugin；它们负责缓存、编译通信、二进制加载和
设备启动。只有新算子需要现有协议无法表达的参数或执行能力时，才扩展这些公共层。

## 1. 明确接口语义并声明

先确定输入输出类型、通道与布局、饱和或舍入规则，以及 mask、标量、缩放和重叠
存储的语义。接口声明与实际 kernel 能力一致；尚未实现的重载不要先暴露。

按 OpenCV 模块归属选择 `include/flagcv/core.hpp` 或 `imgproc.hpp`，保持操作数
顺序，以 `Handle&` 开头，以带默认值的 `Stream` 结尾。`GpuMat` 不拥有设备内存，
`dst` 使用 const 引用，调用方负责预分配输出。

以 array-array 按位操作的接口形式为例：

```cpp
void bitwise_and(Handle& handle, const GpuMat& src1, const GpuMat& src2,
                 const GpuMat& dst, Stream stream = nullptr);
```

## 2. Lowering：校验并构造参数

在 `src/lowering/lowering.hpp` 声明 lowering，在 `src/lowering/<family>.cpp`
实现。同族算子可以复用 lowering 和编译期 mode，但应明确每个 mode 的类型限制。
可参考 [pointwise.cpp](../src/lowering/pointwise.cpp) 和
[flip.cpp](../src/lowering/flip.cpp)。

在进入 runtime 前完成校验，包括：

- 空输入、非正尺寸、depth 和通道数限制。
- 输入输出的尺寸、类型与通道是否匹配。
- 基地址按元素大小对齐，pitch 足以容纳一行。
- 原地操作与部分重叠是否安全；重排类操作尤其需要拒绝读写竞争。
- 尺寸、索引和 stride 的整数范围。

使用 `to_spec(image, "name")` 生成 `ImageSpec`，按需要调用
`require_same_layout()` 等校验辅助函数。公开 API 的 pitch 单位是字节；
`GpuMat::row_stride()` 转换后的 `ImageSpec::row_stride` 单位是元素。

`LoweredOp` 保存 kernel 注册名、图像描述和参数，不保存设备指针。例如逐元素
操作可以按下列顺序构造参数（省略前面的校验）：

```cpp
LoweredOp op;
op.kernel = "my_op";
op.add_image(s1);  // to_spec(src1, "src1")
op.add_image(s2);  // to_spec(src2, "src2")
op.add_image(d);   // to_spec(dst, "dst")
op.add_hinted_param("width", s1.width);
op.add_param("height", s1.height);
op.add_hinted_param("src1_row_stride", s1.row_stride);
op.add_hinted_param("src2_row_stride", s2.row_stride);
op.add_hinted_param("dst_row_stride", d.row_stride);
op.add_constexpr("CHANNELS", s1.channels);
```

参数的选择决定编译缓存如何复用：

| 方法 | 传给 kernel 的方式 | 缓存特化规则 |
| --- | --- | --- |
| `add_param` | 运行期 i32/i64 参数 | 按整数类型特化，不按具体数值特化 |
| `add_hinted_param` | 运行期参数，附带对齐提示 | 在类型之外，按是否能被 16 整除特化 |
| `add_constexpr` | 编译期常量 | 具体数值进入缓存 key |

模式和通道数这类小集合适合 constexpr。尺寸与 stride 使用运行期参数；影响
向量化的 width、row stride 可加 divisibility hint。图像内容不进入缓存 key。

## 3. C++ 入口：lowering 后调用 launch

在 `src/core/<family>.cpp` 或 `src/imgproc/<family>.cpp` 实现公开函数，包含
公开头文件、`src/lowering/lowering.hpp` 和 `src/runtime/runtime.hpp`。
入口只负责调用 lowering、组装指针并传递 stream。现有代码的形式如下：

```cpp
void bitwise_and(Handle& handle, const GpuMat& src1, const GpuMat& src2,
                 const GpuMat& dst, Stream stream) {
  const LoweredOp op = lower_pointwise(src1, src2, dst, PointwiseMode::kAnd);
  launch(handle, op, {src1.data(), src2.data(), dst.data()}, stream);
}
```

指针列表必须与 `op.images` 的顺序和数量一致。入口不分配设备内存，不调用
Python wrapper，也不自行同步。操作数校验留在 lowering，平台调优留在 provider。

## 4. Triton kernel：提供可编译的参数签名

Python wrapper 里的 `numel()`、shape 查询、grid 计算和 JIT 启动不能直接作为
C++ 的编译入口。C++ registry 指向 `@triton.jit` 函数；把 wrapper 所需的元数据
显式放进 lowering 参数，并把 grid 与调优参数交给编译计划。

通用实现放在 `kernels/common/<family>.py`。保留原有 Python wrapper，以便 Python
测试与使用方继续调用；如果原 kernel 只支持连续单通道输入，而 C++ 接口需要
独立行距或多通道，应扩展 kernel 或提供单独的 kernel 入口。

参数表必须严格按以下顺序排列：

1. 每个 `op.images` 对应的 `<ImageSpec.name>_ptr`。
2. `op.params` 中的参数，名称和顺序完全一致，constexpr 明确标注。
3. 编译计划提供的调优 constexpr，例如 `BLOCK_X`、`BLOCK_Y`。

与上面的 `my_op` lowering 对应的签名为：

```python
@triton.jit
def my_op_kernel(
    src1_ptr, src2_ptr, dst_ptr,
    width, height,
    src1_row_stride, src2_row_stride, dst_row_stride,
    CHANNELS: tl.constexpr,
    BLOCK_X: tl.constexpr, BLOCK_Y: tl.constexpr,
):
    # 实现操作语义；row_stride 以元素计，行偏移使用足够宽的整数。
    ...
```

编译器 `_check_arguments()` 会核对参数名和顺序。kernel 的访存还需要覆盖 tile
边界、独立行距和通道交错，避免写入行尾 padding。可参考
[pointwise.py](../kernels/common/pointwise.py)。

## 5. Kernel 注册和编译计划

在 `kernels/registry.json` 的 `kernels` 数组中增加：

```json
{
  "name": "my_op",
  "source": "common/my_op.py",
  "function": "my_op_kernel"
}
```

`name` 必须等于 `op.kernel`，`source` 相对该 registry 所在目录，`function`
必须是可直接编译的 JIT 函数。

然后在 `compiler/flagcv_codegen/providers.py` 添加 plan，并登记到 `PLANS`。
例如对按行分块的逐元素 kernel：

```python
def plan_my_op(request, facts):
    # block 和 warp 应按 facts 中的平台约束选择，或复用签名相容的已有 plan。
    block_x = _power_of_two_at_most(max(facts["tile"] // 4, 64))
    tuning = {"BLOCK_X": block_x, "BLOCK_Y": 1}
    grid = [
        {"product": ["width", "CHANNELS"], "block": block_x},
        {"product": ["height"], "block": 1},
        1,
    ]
    return tuning, min(4, facts["max_warps"]), grid


PLANS["my_op"] = plan_my_op
```

这个 plan 示例采用保守的元素块大小；具体选择应考虑操作数 depth 和平台资源
限制。`tuning` 的 key 要与 kernel 尾部 constexpr 完全一致，grid 中引用的名字
必须来自 lowering 参数。运行期尺寸值不会发送给编译进程，runtime 根据 plan
返回的表达式计算实际 grid；不要在 provider 中读取请求里不存在的 width 数值。

`backends/<platform>/compiler.py` 调用公共 plan 和 `compile_kernel()`；
`identify()` 仅提供编译器身份摘要，用于缓存失效。平台能力检查放在
`compile()` 或它调用的 plan 中，不能依赖无操作数信息的 `identify()`。

默认先查 `backends/<platform>/kernels/registry.json`，再查公共 registry。
平台注册具有排他性：命中后失败会直接报错，不会回退公共 kernel。
只有需要平台实现时才增加平台 kernel 和 registry 条目。确认插件报告的 target、
已安装的 Triton 编译后端及二进制格式一致；注册成功本身不代表设备链路已验证。

## 6. 接入构建与验证

将新增入口和 lowering 源文件加入 `src/CMakeLists.txt` 的 `add_library(flagcv ...)`。
如果扩展已有族文件，无需重复添加。头文件由现有安装规则导出，通用 Python kernel
也由现有 `kernels/common/` 安装规则打包。

C++ 精度测试放在 `tests/test_<family>.cpp`，CPU 标量参考放在
`reference/cpu/`，容差规则见 [数值容差策略](numerical-tolerance.md)。在
`tests/CMakeLists.txt` 调用 `flagcv_add_test(<family>)`，并链接测试分配显存所需的
设备 runtime，依赖对应 backend plugin。已有测试配置以 CUDA runtime 为例；
迁移到其他设备时，测试的内存分配、拷贝、stream 和同步也要相应接入。

在隔离设备的环境中，从公开 C++ 接口运行测试，验证首次编译加载、缓存复用和
GPU 输出对比。用例应包含边界尺寸、支持的 depth/通道、不同 pitch、偏移指针、
输入与 padding 保护、重叠限制和非默认 stream。另跑原 Python 用例，确认 wrapper
保留兼容。Python 性能测试默认现场执行 OpenCV CUDA 基线；新增算子需在
`adapters/opencv_cuda/bridge.cpp` 添加真实 OpenCV 调用，并在
`adapters/opencv_cuda.py` 接入参数映射。对应流程见
[OpenCV CUDA reference](opencv-add-reference.md)。其他后端可显式使用
`--baseline none` 测量 candidate。测量数据不写入本文档。

NVIDIA 环境下的构建与调用链检查示例：

```bash
cmake -S . -B build -DBUILD_SHARED_LIBS=ON -DFLAGCV_BUILD_TESTS=ON \
  -DFLAGCV_BACKENDS=nvidia -DFLAGCV_WARNINGS_AS_ERRORS=ON
cmake --build build -j 8
CUDA_VISIBLE_DEVICES=0 ctest --test-dir build --output-on-failure
```

## 扩展 mask 和标量时的约定

mask 表示“不写”，被排除的目标像素保留原值。使用单通道 8U mask，并将 mask
合并到 store 谓词；调用方需初始化 `dst`。可选 mask 的图像描述、指针及参数签名
必须在 lowering、入口和 kernel 之间保持一致。

标量操作数可采用调用方提供的设备 buffer，让通道值在运行期加载；不要把每次
调用的像素值编码进 constexpr。标量 buffer 的描述与索引规则需要明确实现，
不能直接套用会要求图像尺寸相等的 array-array lowering。
