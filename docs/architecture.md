# FlagCV 架构边界

## 1. 总体分层

```text
Public C/C++ API（即时模式 + 管线模式）
      |
      v
Platform-neutral Image + Pipeline + lowering + runtime
      |
      v
External compiler protocol
      |
      +--> platform kernel registry（优先）
      +--> common kernel registry（平台未登记时回退）
      |
      v
Execution engine（Triton JIT / 厂商影像单元）
      |
      v
Platform stream + GPU/NPU
```

生产依赖只能向下。公共 API、Image 模型、Pipeline、lowering 和测试 workload 都不能
包含任何平台 SDK 头文件。

## 2. 图像数据模型

图像描述需同时承载色彩语义和存储布局。关键设计是 **row pitch 以字节计且独立于
`width * channels * depth`** —— 影像硬件普遍要求行对齐填充，因此不能假设紧凑布局，
源与目标的 pitch 也不要求相同。

多平面格式（NV12、YUV420P）用每平面独立的 offset 与 pitch 表达，色度平面的实际
extent 由 `src/image/format.cpp` 统一推导。该推导必须集中在一处，否则每个 lowering
都要重复且容易分叉。

## 3. 双模式 API

即时模式面向单次调用，管线模式显式组织多算子并允许融合。两者**不存在两套实现**：
即时模式内部构造单算子管线，由 `src/runtime/immediate_cache.cpp` 按算子与操作数
布局缓存已构建的 executable，因此第二次起没有 build 开销。

算子校验放在 lowering 而非即时入口，这样两条路径不会在"接受什么"上漂移。

## 4. 融合

图像处理的瓶颈几乎全在显存带宽，而非算力。典型预处理链若每步落盘，带宽浪费数倍。
因此融合不是可选优化而是核心模块，至少覆盖：

- 连续 elementwise 链合并为单 kernel
- 几何变换吸收后续 elementwise
- 可分离滤波两趟合并
- 色彩转换与相邻重采样融合

虚拟图像用于表达融合后的中间结果，其存储归属可执行体工作区，不出现在执行绑定中。

## 5. 执行引擎选择

部分平台配备专用影像硬件（图像编解码单元、影像预处理单元），属于固定功能单元，
Triton 无法触及。这类单元在其擅长的算子上通常显著快于通用 kernel 且不占计算核心。

因此引擎选择是**按算子**的：registry 条目可指定 `engine` 字段，后端在选择 kernel
时一并决定引擎，而不是全局二选一。

## 6. 数据相关输出尺寸

特征点检测、轮廓提取这类算子的输出元素数在执行前未知。为保住"执行期不同步"契约，
采用**有界输出 + 计数输出**：上界由 build 期确定，调用方自行决定何时同步读取计数。
不采用两阶段 execute，因为那会破坏异步契约。

## 7. 目录职责

| 目录 | 职责 |
| --- | --- |
| `include/` | 唯一公开接口；平台 ABI 与内部头文件不得安装 |
| `src/image/` | 图像数据模型、色彩格式推导、pitch 与 ROI |
| `src/imgproc/` | 即时模式入口 |
| `src/pipeline/` | 管线编排、校验、IR、融合 |
| `src/pipeline/lowering/` | 按算子族拆分，`dispatch.cpp` 为唯一分发入口 |
| `src/runtime/` | 编译客户端、产物缓存、executable |
| `kernels/common/` | 平台中立 Triton kernel |
| `backends/<p>/` | 平台插件，结构同构，新增平台不改共享代码 |
| `reference/cpu/` | 零依赖标量参考实现 |

lowering 只检查公开语义并生成整数描述；block 尺寸、shared memory 等 kernel 能力由
平台编译器判断，因此新增平台不修改 lowering。
