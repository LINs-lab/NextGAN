# ImageNet 代码架构

`model.name` 选择算法，`network.name` 选择生成器主干。当前发布范围是
ImageNet；[使用入口](../imgnet/README.md)、[配置字段](../imgnet/configs/README.md)
和[资产约定](../imgnet/ASSETS.md)分别说明运行、参数与数据要求。

| `model.name` | 算法文件 | 监督方式 |
| --- | --- | --- |
| `tbsm` | `methods/tbsm.py` | 原始 vector 监督，归一化 bearing；λ、ρ 可配置 |
| `logistic_gan` | `methods/logistic_gan.py` | 非饱和 logistic GAN |
| `nextgan` | `methods/nextgan.py` | scalar potential 的输入梯度与目标位移做 field MSE |

TBSM 的 λ、ρ 默认均为 `0.9`，`use_potential_grad` 默认 `false`。
算法目标由方法名固定，不再配置第二个 loss 选择开关。

## 目录与职责

```text
imgnet/
├── main.py                    # 训练入口与训练循环
├── reproduce.py               # NextGAN 环境、训练、评估统一入口
├── config.py                  # YAML 读取、环境变量展开、方法名读取
├── builders/
│   ├── __init__.py            # 训练分派与独立评估构建入口
│   ├── common.py              # 共用参数解析、资产构建、初始化与恢复
│   ├── datasets.py            # ImageFolder、latent 与数据索引缓存
│   ├── evaluation.py          # 直接加载 online/EMA 生成器，无训练对象
│   ├── tbsm.py                # 组装 TBSM
│   └── gan.py                 # 组装两个 GAN 方法
├── methods/
│   ├── tbsm.py                # vector 监督、RepresentationScattering 与更新
│   ├── logistic_gan.py
│   ├── nextgan.py
│   └── _gan_base.py           # GAN 共用调度、优化器、采样与 EMA
├── networks/
│   ├── generator.py           # 生成器包装与 UniversalAdapter
│   ├── critics.py             # critic 网络及表征连接
│   ├── backbone/
│   │   ├── jit/               # JiT 模型、适配器与辅助层
│   │   ├── dit/               # DiT 模型、适配器、位置编码与 SwiGLU
│   │   └── pixeldit/          # PixelDiT 模型、适配器与辅助层
│   ├── vaes/                  # SD-VAE 与 VA-VAE
│   └── representors/          # MAE、ResNet、SigLIP2 与 pixel 表征
├── metrics/                   # 单个 checkpoint 的生成与评估
├── configs/                   # 六份通用模板 + reproduced/ 一份 DiT 实验配置
├── scripts/                   # 训练/评估启动及数据准备工具
└── tests/                     # 按功能集中的十一组回归与共享 fixture
```

`builders.build_training_model` 在分配网络前检查方法参数和选定 checkpoint。
子 builder 接收已解析的配置，负责组装网络与方法；算法文件定义监督目标、
detach 规则和更新过程。TBSM 调用 `step(x, y)`；GAN 调用 `step(next_batch)`，
回调每次从数据迭代器取新 batch，负责跨 epoch 和设备传输。

`_gan_base.py` 共用两个 GAN 的更新调度，具体 `_critic_loss` 和
`_generator_loss` 分别留在各方法中。TBSM 的 `RepresentationScattering` 包含三体散射监督，留在算法文件；
critic 的网络结构集中于 `networks/critics.py`。

测试按 builders、checkpoints、data、evaluation、metrics、launchers、
reproduce、GAN、TBSM、networks、runtime 十一个主题组织。
共享配置和数值 fixture 放在 `tests/helpers.py`，测试文件不相互导入。
新回归加入对应主题文件，避免每轮修复都增加一个测试文件。
在 `imgnet/` 下运行 `python -m unittest discover -s tests` 即可执行整套检查。

## Pixel 与 latent 数据流

```text
训练数据 → 生成器输入空间 → 生成器输出 → to_image → 冻结表征 → critic → 算法损失
真实数据 ───────────────────────────→ to_image ────────┘
```

- **Pixel**：生成器直接处理归一化 RGB。`Generator.to_image` 使用
  `x * 0.5 + 0.5` 映射到图片表征的输入尺度；训练期间该映射本身不做裁剪。
  采样结果再裁剪到 `[0, 1]`。
- **Latent**：生成器处理 latent 张量，真实与生成 latent 均通过冻结 VAE 解码
  后进入图片表征。数据通道数、空间分辨率、multiplier、解码器和统计文件
  必须匹配；SD-VAE 与 VA-VAE 的数据不能仅改路径后混用。
- **表征梯度**：MAE、ResNet、SigLIP2 的权重被冻结，生成图片经过这些编码器的
  输入梯度仍按算法需要保留。PixelRepresentor 直接切分像素 patch。

`network.adapter` 配置预测类型转换，统一使用 `x`、`noise`、`velocity`，默认输出数据 `x`。公共时间约定固定为
`data_to_noise`：t=0 是数据，t=1 是噪声；不再配置插值方向。
JiT 自身的适配器仍通过 `1-t` 对接预训练网络的时间约定。
`UniversalAdapter` 将主干预测转换到算法所需的预测类型，不负责 VAE 解码。
`network.params` 配置主干结构，`latent` 配置解码和数据尺度。

DiT 主干通过 `(x, t, y)` 接收输入，只返回主预测；适配器负责接入统一的
`(x, y, t)` 生成器接口。旧辅助时间、辅助输出头和独立 CFG 接口已移除。
JiT 包装保留主干、标签 dropout 与采样能力，训练损失由三种算法实现，
不再携带另一套未调用的 flow-matching 训练入口。

## Critic 与训练状态

三种方法的生成器 EMA 权重均直接维护在 `_eval_generator` 中。训练时原地更新，
保存 checkpoint 和采样直接使用这份状态，无需同步额外的参数副本。

`Critic` 用于 TBSM 的直接 vector 输出或势能梯度；`ScatteringCritic`
及其变体提供 GAN 所需的 `potential`/`field` 输出。
`RepresentationCritic` 连接冻结表征和 GAN critic，维护特征 EMA
统计与归一化。位置编码、时间编码、RMSNorm 和相同空间 block 共用实现，
方法各自的输出头与损失保持独立。

两种 GAN 的一步包含 K 次 critic 更新和一次 G 更新；每次 critic 更新使用
A 个新 microbatch，数据、噪声、生成时间和插值均重新采样。fake 表征只提取和
归一化一次，critic 使用其 detach 值，G 使用原计算图。LogisticGAN/NextGAN
先完成最后一次 critic 更新再计算 G 梯度；A 份 G 损失合并反传，确保 DDP 同步完整。
NextGAN 的 alpha 按样本和表征头独立采样。
G 查询期间冻结 critic 参数与 spectral normalization 状态，绕过 critic DDP reducer。

初始化权重只初始化生成器，优化器从零开始。训练 checkpoint 保存模型、
优化器、更新步数及调度信息；GAN 使用 version 2，只在完整 G 更新后保存，
不接受旧 GAN 调度。TBSM 保留 version 1 和待累积梯度；恢复时更新/累积调度必须匹配。
checkpoint 不恢复 RNG 状态或数据游标，不能据此承诺无缝逐位续训。

训练和评估入口在固定的 Torch 2.5 环境中禁用 cuDNN SDPA，以规避已复现的
完整 JiT-B BF16 执行计划错误；不重新启用调用方已禁用的其他 SDPA 后端。
potential-gradient 路径仍保留 math-only attention，以支持二阶反向。

评估只构建生成器与可选解码器，按 `evaluation.weights` 选择 `online` 或
`ema`，通过 `builders.build_evaluation_generator` 直接读取现有 checkpoint，
不创建训练算法、优化器、训练表征或 critic，也不包装 DDP。
评估读取当前 checkpoint 的 `backbone`（online）或
`model._eval_generator.backbone.*`（EMA），严格匹配网络；缺失 EMA 时明确报错，
不会回退到 online 权重。训练恢复必须包含当前版本的 `training_state`；
不再为旧训练 checkpoint 补默认状态。预训练初始化权重通过独立初始化路径加载。
评估结果保留 checkpoint 步数。
每次评估使用构建时选定的一套权重，只生成一轮样本；不再套空 EMA 切换器或
固定单值的 CFG/EMA 网格。CSV 保留原有列，记录实际 online/EMA 标签与固定
单步采样参数。
P&R 参考目录与缓存状态由 rank 0 判定并广播，所有 rank 共同进入缓存或
特征提取分支，避免并发缓存发布或可见状态差异打乱集体通信。
`reproduce.py` 先检查全部选定实验的资产，再串行训练和评估；输出中保存
生效配置、资产/源码哈希、日志与指标，重复评估使用独立输出目录。

预训练权重和数据不随代码发布；主干及派生代码的许可范围见
[第三方说明](../THIRD_PARTY_NOTICES.md)。
