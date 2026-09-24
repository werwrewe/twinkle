# GPU FlashAttention-3 开箱指南（行内离线源）

本文档面向**办公网内网环境**：GPU 机器不能直连公网 pip，twinkle 只有源码没有 wheel，
FlashAttention-3（下称 FA3）的 wheel 是在特定镜像里针对特定 torch 版本编译出来的。
按本文顺序执行即可完成安装，并能在训练前确认"跑起来的确实是 FA3，而不是被静默替换掉的别的 kernel"。

> 适用范围：CUDA GPU（H800 / H100，Hopper / SM90）。昇腾 NPU 走另一条路径，
> 见 [NPU 开箱指南](docs/source_zh/使用指引/NPU的支持.md)。

## 一、先记住三件事

这三条是 FA3 与 FlashAttention-2（FA2）最容易踩错的地方，后面所有步骤都围绕它们展开。

### 1. FA3 的 wheel、发行版名、模块名是**三个不同的东西**

| 名称 | 值 | 用途 |
| --- | --- | --- |
| wheel 文件名 | `flash_attn_3-<version>-...whl` | `pip install` 时要匹配的文件 |
| 发行版（distribution）名 | `flash-attn-3` | `importlib.metadata` 里的名字，transformers 用它判断"装没装" |
| Python 导入模块名 | `flash_attn_interface` | 代码里 `import flash_attn_interface` |

注意：**FA2 的模块名是 `flash_attn`，FA3 的是 `flash_attn_interface`**。
所以 `python -c "import flash_attn"` 在装了 FA3 的环境里会 **ImportError** —— 这不代表 FA3 没装好。
（`flash_attn_3/setup.py` 中 `PACKAGE_NAME = "flash_attn_3"`、`py_modules = ["flash_attn_interface", ...]`。）

### 2. 必须装 wheel，不能手工拷贝目录

transformers 判断 FA3 可不可用，要**同时**满足：

1. `flash_attn_interface` 这个模块能被找到；**并且**
2. `importlib.metadata.packages_distributions()['flash_attn_interface']` 里含有 `flash-attn-3`；
   **并且**
3. CUDA 可用。

第 2 条意味着：**把编译产物目录手工 `cp` 进 `site-packages` 是不行的**。
手工拷贝的目录没有 METADATA / RECORD 记录，`packages_distributions()` 查不到，
transformers 会认为 FA3 不可用（虽然 `import flash_attn_interface` 明明能成功）。
只有 `pip install <whl>` 才会注册发行版信息。

> 这是 910 NPU 那边"手工拷贝也能用"的经验在这里**不成立**的原因：
> NPU 的 adapter 自己改了 transformers 的兼容性矩阵，绕开了这个判定；CUDA 这条路没有 patch，
> 用的是 transformers 原生判定，所以必须老老实实装 wheel。

### 3. 被静默替换的风险

当"请求了 FA3 但判定为不可用"时，transformers 有两种行为：

- 环境里装了 `kernels` 这个包 → **静默**从 HuggingFace Hub 下载另一个 kernel
  （`kernels-community/vllm-flash-attn3`）顶上，训练照跑，但你跑的不是 FA3 了，日志里也不显眼；
- 没装 `kernels` → 抛出明确的 ImportError。

本文档第 4 节的验证脚本就是为了在第 2 种情况之前就把问题暴露出来。

## 二、环境要求

| 组件 | 要求 | 说明 |
| --- | --- | --- |
| GPU | H800 / H100（Hopper，SM90） | FA3 的主目标架构；transformers 的兼容矩阵要求 SM80+ |
| CUDA | >= 12.3 | FA3 官方要求 |
| 镜像 | `registry.cic.cmbchina.cn/cic/vllm/vllm-openai:dsv4-multilora` | **FA3 的 wheel 是针对该镜像内的 torch 编译的** |
| Python | 3.11 | 与 wheel 的 `cp311` 标签一致 |
| PyTorch | 镜像自带版本，**不要动** | see 红线 |
| transformers | >= 5.x | 由 twinkle 依赖带入；FA3 走原生 CUDA 分支，**无需任何 patch** |

**最重要的一条约束**：FA3 的 wheel 是 C++/CUDA 扩展，与编译时所用的 torch 二进制 ABI 绑定。
**换了 torch 版本或换了镜像，就必须重新编译这个 wheel**，否则会出现诡异的运行时报错
（例如莫名的设备号错误、静默读到错误内存）。

## 三、安装步骤（在 dsv4-multilora 镜像内执行）

### 0. 进入镜像并记录 torch 版本

```bash
docker run -it --rm -v /nas:/nas registry.cic.cmbchina.cn/cic/vllm/vllm-openai:dsv4-multilora bash

# 记下这个版本号，后面每一步都要用它当"锚"
python -c "import torch; print(torch.__version__)"
```

### 1. 装 FA3 wheel

FA3 的 wheel 编译产物在 `/nas/disk6/ljl/wheel/flash-attention/hopper/dist/`。

```bash
# 看一眼实际文件名，不要照抄通配符
ls /nas/disk6/ljl/wheel/flash-attention/hopper/dist/

# 安装（注意文件名是 flash_attn_3-*，不是 flash_attn-*）
pip install /nas/disk6/ljl/wheel/flash-attention/hopper/dist/flash_attn_3-*.whl --no-deps
```

`--no-deps` 是必须的：FA3 依赖的 `torch` 已经在镜像里，加 `--no-deps` 才不会让 pip 去解析、
进而有升级 torch 的机会。

**验证（用正确的模块名）**：

```bash
python -c "import flash_attn_interface as fa; print(fa.__file__)"
# 期望：打印 .../site-packages/flash_attn_interface.py，无异常
```

不要用 `import flash_attn` 来验证 FA3 —— 那是 FA2 的模块名。

### 2. 配置行内 pip 源

```bash
pip config set global.index-url http://central.jaf.cmbchina.cn/artifactory/api/pypi/group-pypi/simple
pip config set install.trusted-host central.jaf.cmbchina.cn
```

行内源是 http，所以 `trusted-host` 必须配，否则 pip 会因 TLS 校验直接拒绝。

### 3. 安装 twinkle（源码、editable）

本文档所在的 twinkle 分支为 **`dev-swift-v5-fa3`**（在 `dev-swift-v5` 基础上加入 FA3 适配，
见第五节）。把该分支的源码放到办公网可访问的目录（下文以 `/nas/disk1/clc/twinkle` 为例），然后：

```bash
cd /nas/disk1/clc/twinkle
pip install -e . --no-deps
```

用 `-e`（editable）而不是普通安装：源码目录直接映射进 `site-packages`，
后续改代码不用重装，也避免普通安装产生的副本与源码不同步。

### 4. 补齐缺失依赖

`--no-deps` 意味着 pip 不会替你装依赖。**缺什么装什么，每条也带 `--no-deps`**：

```bash
# 先看缺什么
python -c "import twinkle; print('twinkle ok')"

# 例如提示 No module named 'xxx'，就单独装那一个
pip install xxx --no-deps
```

逐条安装、逐条带 `--no-deps`，是这套流程能保证"绝不覆盖 torch"的关键。
（全量依赖清单见 `pyproject.toml`；通常镜像里已经有了大部分。）

### 5. 统一核对

```bash
pip list | grep -Ei "torch|flash|twinkle"
```

期望看到的组合（示例）：

```
flash_attn_3        <版本>     # FA3，来自第 1 步的 wheel
flash-attn-3        <版本>     # 同上，发行版记录
twinkle             <版本>     /path/to/src   # editable
torch               <镜像自带>  # 必须与第 0 步记录的一致
```

### 红线（务必遵守）

| 禁止 | 原因 |
| --- | --- |
| ❌ `pip install -U ...` | `-U` 会解析并升级依赖，可能把镜像里的 torch 换掉 |
| ❌ 在不匹配的 torch / 非 dsv4-multilora 镜像里装这个 wheel | ABI 不匹配，行为不可预期 |
| ✅ 所有 `pip install` 都带 `--no-deps`，或显式钉 `torch==$(python -c "import torch; print(torch.__version__)")` | 保证 torch 不被替换 |

补充说明：若某条命令**不得不**解析依赖（例如从行内源装一个带依赖的包），
至少写成 `pip install <pkg> torch==$(python -c "import torch; print(torch.__version__)")`，
把 torch 显式钉在镜像自带版本上。

## 四、验证安装：一条命令看清环境

transformers 自带 FA3 可用性判定（distribution 级），直接问它即可，
**不需要构造模型、不需要 GPU 跑起来**：

```bash
python - <<'EOF'
from transformers.utils.import_utils import is_flash_attn_3_available
print('is_flash_attn_3_available:', is_flash_attn_3_available())
from importlib.metadata import packages_distributions
print('distributions for flash_attn_interface:', packages_distributions().get('flash_attn_interface'))
import torch
print('cuda_available:', torch.cuda.is_available(), 'capability:', torch.cuda.get_device_capability() if torch.cuda.is_available() else None)
EOF
```

判读要点：

- `is_flash_attn_3_available` 必须是 `True`；
- `import flash_attn_interface` 成功但判定为 `False` → **典型的"手工拷贝"症状**，
  说明模块在但没注册发行版（distributions 一行为空或不含 `flash_attn_3`），
  回第三节第 1 步用 `pip install <whl>` 重装；
- 发行版名要对得上：transformers 要求 `flash-attn-3`（大小写、下划线会归一化）。

训练启动后还有两道运行期确认，见第五节末尾和第十节。

## 五、在训练里启用 FA3

### 方式一：kernelize（推荐，和 sdpa 同一个机制）

`kernel/config.py` 的默认映射里有一条：

```python
cfg['flash_attention_3'] = KernelChoice(op='flash_attention_3', backends=('cuda', ))
```

```python
from twinkle.kernel import kernelize
from twinkle.model import TransformersModel

model = TransformersModel(model_id=model_id, device_mesh=...)   # 不需要任何 attention 参数
model = kernelize(model)
```

`install_fa3` 在 kernelize 时做的是**纯叶子替换**：CUDA 上模型的默认选择本来就是
`'sdpa'`（`config._attn_implementation` 不动），installer 把 FA3 叶子挂到
`ALL_ATTENTION_FUNCTIONS['sdpa']` 背后，所有默认模型即走 FA3——与 NPU sdpa op
逐行同构，零翻转、零 kwargs。`'flash_attention_3'` key 也一并填上，显式
`attn_implementation='flash_attention_3'` 的选择和 SP 的 FA wrapper 落在同一片叶子。
**显式选择永远优先**：config 是 `'flash_attention_2'`/`'eager'` 的模型走别的 key，
完全不受影响。

没装 flash-attn-3、或不在 CUDA 上时，这条 entry 的 `available()` 为 False，
kernelize 直接跳过，模型保持 transformers 默认 sdpa，行为与未开启逐位一致。

叶子对 mask 形态自适应：mask 为 None 或 2D padding mask（pack、常规因果）走 FA3
快路径；padding batch 下 sdpa mask 工厂产的 4D 加性 mask 会回退 stock sdpa——
正确性不变，只是这些调用不加速。

### 方式二：kwargs 透传（transformers 公开参数）

```python
model = TransformersModel(model_id=model_id, attn_implementation='flash_attention_3', device_mesh=...)
```

`TransformersModel(**kwargs)` 原样透传给 `from_pretrained`，由 transformers 写入
`config._attn_implementation`——框架侧零改动，选 `flash_attention_2`/`'eager'` 也用这条路。
注意：这条路 registry key 不由 twinkle 拥有；kernelize 之后该 key 的叶子会被
`install_fa3` 换成 twinkle 的（同一片 FA3 叶子，行为一致）。

### 启动时会发生什么

方式一 kernelize 成功安装后**不打专门日志**（与 sdpa op 一致，静默换叶子）；只有
key 已被外来 wrapper（如 SP partial）占用而跳过时才会看到
`[FA3] <key> already holds ...; keeping the existing wrapper` 的 INFO。
判断"跑起来的确实是 FA3"看第十节的 probe（`'sdpa'` key 的叶子是
`cuda_fa3_attention_forward`、`attn_leaf` 落在 `flash_attn_interface`、kernel 调用计数 > 0）。

### 关于 attention dropout

FA3 的 `flash_attn_func` / `flash_attn_varlen_func` **没有 `dropout_p` 参数**。
transformers 按函数签名过滤 kwargs，所以 `attention_dropout > 0` 会被**静默丢弃**，
训练实际上是在 `dropout=0` 下进行的（twinkle 不再就此告警）。

若确实需要 dropout，请改用 `flash_attention_2`；否则建议把 `attention_dropout` 设为 0，
让配置与实际行为一致。

## 六、并行策略支持情况

| 策略 | FA2 | FA3 | 说明 |
| --- | --- | --- | --- |
| DP / FSDP | ✅ | ✅ | 与注意力后端无关 |
| **ulysses（SP）** | ✅ | ✅ | 本分支已支持；`padding_free` 打包输入也可用 |
| ring（derived ring attention） | ✅ | ❌ | 显式抛 `NotImplementedError`，见下 |
| CP（SP + ring 组合） | ✅ | ❌ | 依赖 ring，同上 |

**ring / zigzag 目前只支持 FA2。** 如果 `device_mesh` 配出了 ring（`rp_world_size > 1`）
又选择了 FA3，会直接得到明确报错：

```
NotImplementedError: Derived ring attention only supports flash_attention_2 backend.
```

这是刻意为之 —— 与其静默产生错误结果，不如立刻停下。需要 ring 的话请改用 `flash_attention_2`。

同理，若把 `attn_implementation` 设成 `sdpa`/`eager` 又想用 `padding_free`（打包 / 变长批次），
也会得到明确报错。**需要打包训练时请用 `flash_attention_2` 或 `flash_attention_3`。**

## 七、常见问题

### Q1：`import flash_attn` 报错，是 FA3 没装好吗？

不是。FA2 的模块名是 `flash_attn`，FA3 的是 `flash_attn_interface`。
请用第四节的 `is_flash_attn_3_available()` 判断，而不是 `import flash_attn`。

### Q2：明明装好了，transformers 却说 FA3 不可用

对照第一节第 2 条检查三件事：

1. `python -c "import flash_attn_interface"` 是否成功；
2. `python -c "from importlib.metadata import packages_distributions as p; print(p().get('flash_attn_interface'))"`
   是否含有 `flash_attn_3`；
3. `python -c "import torch; print(torch.cuda.is_available())"` 是否为 True。

若第 1 条成功、第 2 条为空 → 你是手工拷贝的，请用 wheel 重装。
若第 3 条为 False → 在无 GPU 或 CUDA 不可用的机器上，FA3 判定天然为不可用。

> 另注：transformers 对可用性的判定带 `lru_cache`。如果你是在**同一个进程内**
> 先检查、后 pip 安装，缓存不会刷新。重新起进程即可。

### Q3：训练日志里出现了别的 kernel 名字，或者显存/精度与预期不符

大概率是遇到了"静默替换成 Hub kernel"（`kernels-community/vllm-flash-attn3`）。
检查环境里是否装了 `kernels` 包；如果装了而 FA3 又判定不可用，transformers 就会走这条路。
按 Q2 修好 FA3 后，这条替换就不会再发生。

### Q4：换了 torch 版本 / 换了镜像之后怎么办

**必须重新编译 FA3 的 wheel**。wheel 与编译时的 torch ABI 绑定，
跨 torch 版本复用会出现难以定位的运行时错误（如设备号异常、静默内存错乱）。
编译产物放回 `/nas/disk6/ljl/wheel/flash-attention/hopper/dist/` 后，重跑第三节第 1 步。

### Q5：`pip install flash_attn_npu` 之类会不会把我的 FA3 冲掉？

不会直接冲突（FA3 与 NPU 的 `flash_attn_npu` 是不同发行版），但**任何**不带 `--no-deps` 的
`pip install` 都有解析并升级 torch 的风险，进而使 FA3 的 wheel 失效。
保持第三节第 5 步的核对习惯即可。

### Q6：为什么 twinkle 不做 transformers patch？

因为 CUDA 上 FA3 是 transformers **原生支持**的：
兼容性矩阵（`FLASH_ATTENTION_COMPATIBILITY_MATRIX[3]`）和导入分支（`_lazy_imports`）上游都有。
NPU 需要 adapter，是因为 transformers 的 `_lazy_imports` 里 NPU 分支排在 FA3 分支**前面**，
必须改判定；CUDA 没有这个问题。
twinkle 在这条路径上只做两件事：

1. **接线**：让 SequenceParallel 把 FA3 也包进自己的 attention wrapper（见下）；
2. **拥有 registry key**：`kernel/ops/flash_attention3` 把 FA3 注册成 kernelize
   的一等 op（见第五节方式一），将来 CUDA 侧修复和 NPU backend 都挂在这个落点上。

## 八、本分支相对 `dev-swift-v5` 的改动

如果你需要在办公网重建这套改动，本分支 `dev-swift-v5-fa3` 涉及：

| 文件 | 改动 |
| --- | --- |
| `src/twinkle/model/transformers/strategy/sequence_parallel/__init__.py` | 新增 `FLASH_ATTENTION_IMPLS = ('flash_attention_2', 'flash_attention_3')`；把 SP 的 attention wrapper 注册到**每一个** FlashAttention 名字上；causal-mask 与 `padding_free` 判定改为按 `FLASH_ATTENTION_IMPLS` 白名单 |
| `src/twinkle/kernel/ops/flash_attention3/` | **新增**：FA3 op（CUDA 叶子 + `install_fa3` installer），把 FA3 叶子挂到 `'sdpa'`（CUDA 默认选择，生效主路）与 `'flash_attention_3'` 两个注册表 key 背后，不动 `config._attn_implementation`；叶子对 4D mask 回退 stock sdpa |
| `src/twinkle/kernel/config.py`、`src/twinkle/kernel/ops/__init__.py` | 默认映射新增 `cfg['flash_attention_3'] = KernelChoice(op='flash_attention_3', backends=('cuda',))` 并注册 op |
| `tests/transformers/test_sp_flash_attention_registration.py` | **新增**：CPU 上即可运行的 SP 注册单元测试，不需要真实 FA3 |
| `cookbook/transformers/sp_fsdp_dense.py` | `SelfCognitionProcessor` 只对 demo 自带的 `ms://` 数据集 map（自定义 jsonl 可直接喂），attention 侧零改动 |

其中第一行是**修 bug**：transformers 把 `flash_attention_2/3/4` 三个 key 指向**同一个**
`flash_attention_forward` 对象，而 `AttentionInterface.get_interface` 是按 key 精确查找的。
原实现只覆盖了 `flash_attention_2`，于是 `attn_implementation='flash_attention_3'` + SP
会**静默地不做序列并行**（不报错，只是变慢/占显存），很难察觉。现在三个名字都覆盖，
且 `_origin` 别名保证 SP 内部回头调用的是 transformers 的原实现，不会自我递归。

把改动搬到办公网的方式（任选其一）：

```bash
# 方式 A：git bundle（附带完整提交历史）
git bundle create twinkle-fa3.bundle dev-swift-v5-fa3
# 在办公网仓库里：
git fetch /path/to/twinkle-fa3.bundle dev-swift-v5-fa3:dev-swift-v5-fa3
git checkout dev-swift-v5-fa3

# 方式 B：打补丁（先 add，否则新增文件不会进补丁）
git add -A
git diff --cached dev-swift-v5 > fa3.patch
git apply fa3.patch                      # 在办公网仓库里应用
```

## 九、验证清单

安装完成后，逐条打勾：

- [ ] `python -c "import torch; print(torch.__version__)"` 与第 0 步记录一致（torch 没被换掉）
- [ ] `python -c "import flash_attn_interface"` 成功
- [ ] 第四节脚本输出 `is_flash_attn_3_available: True`
- [ ] `pip list | grep -Ei "torch|flash|twinkle"` 输出符合第三节第 5 步
- [ ] kernelize 后 `'sdpa'` key 的叶子已换成 twinkle 的（见第十节 probe 第 2 条）
- [ ] 启动日志中**没有** `[flash-attn] ... is not importable` 之类的 transformers WARNING
- [ ] 若使用 `attention_dropout > 0`，确认已接受"实际 dropout 为 0"或改回 `flash_attention_2`

## 十、跑一遍训练，确认 FA3 真的在跑

安装"成功"不等于训练时真的走了 FA3：transformers 在 FA3 不可用时**会静默换成 Hub kernel**，
loss 照常下降，只有速度变了。所以装完建议用 cookbook 现有的 SP 示例跑一次冒烟。

不需要任何专用脚本，`cookbook/transformers/sp_fsdp_dense.sh` 的所有配置都是 CLI flag，
调用处覆盖即可：

```sh
cd cookbook/transformers
CUDA_VISIBLE_DEVICES=0,1 sh sp_fsdp_dense.sh \
    --model-id /path/to/Qwen3-0.6B \
    --dataset-id /path/to/sft.jsonl \
    --template-cls Qwen3Template \
    --dp-size 1 --fsdp-size 1 --ulysses-size 2 \
    --batch-size 2 --train-samples 100
```

不需要任何 attention 参数：脚本默认 `kernelize(model)`，CUDA 上 `install_fa3` 会把
FA3 叶子挂到 `'sdpa'` key 背后，默认模型即走 FA3（见第五节方式一）。
基线对比用 `TWINKLE_TORCH_BASELINE=1` 关 kernelize 跑同一条命令（stock sdpa）。

**看三个地方判断 FA3 是否真跑起来了：**

1. 启动日志**没有** `does not consider it available` / 静默换 Hub kernel 这类 WARNING；
2. kernelize 后 `'sdpa'` key 的叶子已换成 twinkle 的 FA3 叶子：

   ```bash
   PYTHONPATH=src python -c "
   from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS as A
   from twinkle.kernel.ops.flash_attention3.cuda import cuda_fa3_attention_forward as leaf
   # kernelize 之后（训练进程内）应为 True；standalone 跑这行只是演示判定方式
   print(A['sdpa'] is leaf, A['flash_attention_3'] is leaf)"
   ```

   （standalone 下没跑 kernelize，两个都该是 False；要在训练进程里、kernelize 之后判定。）
3. 想拿到铁板证据，在同环境跑一行 leaf 解析：

   ```bash
   python -c "
   from transformers import modeling_flash_attention_utils as mfu
   print(mfu._lazy_imports('flash_attention_3')[1])"
   # 期望看到 flash_attn_interface；若是 flash_attn 或 kernels 相关，说明被换掉了
   ```

> 说明：若要 fa2 基线（`TransformersModel(..., attn_implementation='flash_attention_2')`）
> 需要环境里有 **FA2**（`flash_attn` 包）。镜像里没有的话基线会报错，用
> `TWINKLE_TORCH_BASELINE=1` 的 stock sdpa 当基线也能完成冒烟。

## 参考资源

- [FlashAttention 仓库](https://github.com/Dao-AILab/flash-attention)
- [transformers FlashAttention 文档](https://huggingface.co/docs/transformers/main/en/perf_infer_gpu_one#flashattention-2)
- [昇腾 NPU 开箱指南](docs/source_zh/使用指引/NPU的支持.md)
- [Twinkle GitHub](https://github.com/modelscope/twinkle)

## 下一步

- 📖 阅读 [快速开始](docs/source_zh/使用指引/快速开始.md) 了解训练示例
- 📖 阅读 [安装指南](docs/source_zh/使用指引/安装.md) 了解其他平台的安装
- 📖 阅读 [NPU 开箱指南](docs/source_zh/使用指引/NPU的支持.md) 了解昇腾路径
