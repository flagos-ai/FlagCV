# FlagCV

FlagCV 是一个面向多种加速器平台的 C/C++ 图像处理运行时。算子以 Triton 实现,
通过统一的 kernel 库支持多种硬件后端。

当前状态:**早期开发**。分层结构与核心契约已确定,算子实现以 `bitwise_and`
作为参考竖切。

## 调用链

```
include/flagcv/imgproc.hpp          bitwise_and() 声明
      |
src/imgproc/bitwise.cpp             GpuMat → ImageSpec,组装 OperationSpec
      |
src/runtime/immediate_cache.cpp     单算子管线构建与缓存(待实现)
      |
src/pipeline/lowering/dispatch.cpp  唯一分发入口
      |
src/pipeline/lowering/bitwise.cpp   语义校验 → LoweredOperation
      |
kernels/registry.json               operation → source/functions
      |
kernels/common/bitwise.py           Triton kernel
```

## 用法

```cpp
#include <flagcv/imgproc.hpp>

flagcv::Handle handle("nvidia", 0);

flagcv::GpuMat src1(device_a, 1920, 1080, FLAGCV_DEPTH_8U, 3);
flagcv::GpuMat src2(device_b, 1920, 1080, FLAGCV_DEPTH_8U, 3);
flagcv::GpuMat dst(device_c, 1920, 1080, FLAGCV_DEPTH_8U, 3);

flagcv::bitwise_and(handle, src1, src2, dst);

// 带 mask:mask 为零处 dst 保持原值
flagcv::GpuMat mask(device_m, 1920, 1080, FLAGCV_DEPTH_8U, 1);
flagcv::bitwise_and(handle, src1, src2, dst, mask);
```

调用异步入队,读结果前需自行同步流。FlagCV 不分配或释放设备内存,`GpuMat`
只是调用方存储与其布局的配对视图。参数不一致时抛 `flagcv::Error`。

## 行为约定

- 只接受整数深度,浮点输入抛异常而非按位重解释
- `mask` 为 8 位单通道,为零处目标像素保持原值
- `mask` 折进 store 谓词而非用 `tl.where` 选值,因此未写入像素字节级保持原内容
- row pitch 以字节计且独立于 `width * channels * depth`,源与目标 pitch 不要求相同

## 编译与测试

```bash
cmake -S . -B build -DFLAGCV_WARNINGS_AS_ERRORS=ON
cmake --build build
cd build && ctest --output-on-failure
```

测试覆盖 lowering 层的算子校验与降级结果,不需要设备。

`src/runtime/immediate_cache.cpp` 尚未实现,调用 `bitwise_and` 会在校验通过后抛
`FLAGCV_STATUS_NOT_SUPPORTED`。需要补的步骤在该文件注释中列出。

## 文档

- [架构边界](docs/architecture.md)
- [算子开发指南](docs/operator-development.md)
- [数值容差策略](docs/numerical-tolerance.md)
