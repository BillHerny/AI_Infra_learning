# Distribution-Training：EXO Gym 模拟 + MiniDDP 复现

这个目录是一个“机制验证”型项目：

- **MiniDDP**：从零实现 DDP 的核心机制（参数广播、梯度 hook、bucket、async all-reduce、`no_sync`），并与 `torch.nn.parallel.DistributedDataParallel` 做数值对照。
- **EXO Gym**：在单张 RTX 4060 上拉起多个虚拟 PyTorch 节点，跑 MNIST 的 DDP / DiLoCo / SPARTA 对比，观察节点数变化对收敛和通信的影响。

> 重要边界：单卡上多个虚拟 rank 共享同一张 GPU。它验证的是**算法行为和通信机制**，不是多卡加速比。EXO Gym 在 `num_nodes > GPU 数` 时会自动用 **Gloo**；不要强制 NCCL，否则会报 `Duplicate GPU detected`。

> 本机已验证环境：`conda activate exogym`，Python 3.11，`torch 2.14.1+cu130`，`torchvision 0.29.1+cu130`；`miniddp` 与 `exogym` 均已 editable 安装。

---

## 0. 目录结构

```text
Distribution-Training/EXO-Gym/
├── miniddp/
│   ├── __init__.py
│   ├── comm.py          # 进程组、端口、参数广播
│   ├── reducer.py       # bucket、hook、async all-reduce、finalize
│   ├── ddp.py           # MiniDDP 包装器
│   └── launch.py        # mp.spawn 辅助
├── examples/
│   └── train_mnist.py   # MiniDDP / torch DDP / 单进程训练入口
├── tests/
│   ├── compare_impls.py # 与 torch DDP 的数值对照
│   └── test_compare.py  # pytest 封装
├── experiments/
│   └── mnist_exogym.py  # 参数化后的 EXO Gym MNIST 实验
├── scripts/
│   ├── run_miniddp_cpu.sh
│   ├── run_miniddp_gpu.sh
│   ├── run_exogym_smoke.sh
│   └── run_exogym_scaling.sh
├── pyproject.toml
├── requirements.txt
└── README.md
```

---

## 1. 环境准备（只做一次）

当前机器：RTX 4060 Laptop 8GB、驱动 610.74、23GB 内存、Linux。建议使用独立的 **Python 3.11** 环境，不要污染 base。

```bash
cd ~/projects/ai-infra/Distribution-Training/EXO-Gym

conda create -n exogym python=3.11 -y
conda activate exogym
python -m pip install -U pip

# 与驱动匹配的 CUDA 13.0 wheel；如果该索引不可用，换成 cu128
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130

# 安装本项目（MiniDDP 本身没有额外依赖）
pip install -e .
pip install numpy pandas matplotlib pytest tqdm

python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

**这一步在干什么：**

- 新建 `exogym` 环境，避免 base 里的 torch 被其他依赖升降级。
- `torch`/`torchvision` 用 CUDA 专用索引安装，保证 `torch.cuda.is_available()` 为真。
- `pip install -e .` 把 `miniddp` 以可编辑模式安装，之后 `import miniddp` 能找到当前目录的代码。

如果 `cu130` 索引不存在：

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

---

## 2. 先跑 MiniDDP：CPU 正确性测试

```bash
bash scripts/run_miniddp_cpu.sh
```

这个脚本做两件事：

1. `python -m pytest tests -q`
2. 用 2 个虚拟 rank 在 CPU 上训练 20 步 MNIST，并保存 `results/miniddp_cpu.pt`

**测试在验证什么：**

`tests/compare_impls.py` 会用完全相同的模型初始化、数据划分、SGD 超参数，分别跑：

- `MiniDDP`
- `torch.nn.parallel.DistributedDataParallel`

然后比较两边 rank 0 的最终 `state_dict`。如果实现正确，最大绝对误差应该在 `1e-6` 到 `1e-5` 量级，终端会打印：

```text
max_abs_diff=... worst=...
```

**为什么用 subprocess 跑测试：**

`torch.multiprocessing.spawn` 要求入口是一个真实的 `.py` 文件。直接在 pytest 进程里 spawn 容易和 CUDA/进程组状态冲突，所以 `test_compare.py` 用 `subprocess.run` 调 `compare_impls.py`。

---

## 3. 在 4060 上跑 MiniDDP

```bash
bash scripts/run_miniddp_gpu.sh
```

关键点是 **Gloo + 单卡 CUDA**：

- 2 个 rank 都调用 `torch.cuda.set_device(0)`，共享 `cuda:0`。
- 进程组后端用 `gloo`，因为 NCCL 不允许两个 rank 落在同一张 GPU 上。
- 我们实测过：Gloo 对 CUDA tensor 做 `all_reduce` 是可行的；`torch DDP + Gloo + 单卡 CUDA` 也能跑。

脚本会依次跑：

```bash
python examples/train_mnist.py --impl miniddp  --world-size 2 --device cuda --backend gloo
python examples/train_mnist.py --impl torchddp --world-size 2 --device cuda --backend gloo
RUN_GPU_TESTS=1 python -m pytest tests -q
```

**注意：**

- 这不是双卡加速，两个 rank 在抢同一张 4060。
- 8GB 显存下，`world_size=2`、`batch_size=64` 是安全起点；不要一上来就 `world_size=8`。
- 如果 OOM，先降 `--batch-size`，再降 `--world-size`。

---

## 4. 安装 EXO Gym

```bash
cd ~/projects/ai-infra/Distribution-Training/EXO-Gym
mkdir -p third_party
git clone https://github.com/exo-explore/gym.git third_party/exogym
cd third_party/exogym
git checkout 9fa323c      # 固定到本 README 对应的版本，可选
pip install -e .
cd ../..
python -c "import exogym; print(exogym.__file__)"
```

> **Linux 本机补丁**：上游 `third_party/exogym/exogym/utils.py` 把
> `GLOO_SOCKET_IFNAME` 硬编码成了 macOS 的 `lo0`，在 Linux 上会报
> `Unable to find address for: lo0`。本项目已把它改成：macOS 用 `lo0`，
> Linux 用 `lo`。如果重新 clone 上游，需要重新打这个补丁。

**这一步在干什么：**

- EXO Gym 的 PyPI 包还在测试阶段，直接从 GitHub 安装最稳。
- `pip install -e .` 会安装 `torch>=2.0`、`torchvision`、`datasets`、`transformers`、`wandb` 等依赖；torch 已经装好，pip 不会重复下载。
- 固定 commit 是为了避免上游更新导致实验不可复现。

如果安装很慢，可以先只装最小依赖：

```bash
pip install -e . --no-deps
pip install numpy tqdm "datasets>=2.0.0,<4.0" transformers boto3 einops torchvision pytest matplotlib psutil
```

---

## 5. EXO Gym 冒烟测试

> `scripts/*.sh` 会通过 `scripts/common.sh` 自动使用
> `/home/fate-whispering/miniconda3/envs/exogym/bin/python`，所以不激活 conda
> 也不会误用 base。需要换解释器时：
> `PYTHON_BIN=/path/to/python bash scripts/run_exogym_smoke.sh`。

```bash
bash scripts/run_exogym_smoke.sh
```

等价于：

```bash
MPLBACKEND=Agg python experiments/mnist_exogym.py \
  --num-nodes 2 \
  --max-steps 10 \
  --batch-size 64 \
  --minibatch-size 64 \
  --val-size 512 \
  --val-interval 10 \
  --strategies ddp \
  --run-tag smoke \
  --plot
```

**为什么先跑 10 步：**

EXO Gym 官方 MNIST 示例默认 4 节点、150 步、3 个策略，还要做验证集评估。第一次应该先确认：

- MNIST 能下载；
- 2 个虚拟节点能正常 `mp.spawn`；
- Gloo 能建立连接；
- CSV 日志能写出；
- 显存不爆。

**日志位置：**

```text
logs/exogym_smoke_n2_ddp_s42_<timestamp>/
├── train.csv
├── validation.csv
└── config.json
```

`validation.csv` 里有 `step`、`local_loss`、`global_loss`；`train.csv` 里有 `step`、`examples_trained`、`train_loss`、`lr`。

---

## 6. N 节点扫描

```bash
bash scripts/run_exogym_scaling.sh
```

脚本固定全局 batch = 1024：

| 节点数 N | 每节点 batch |
|---|---|
| 1 | 1024 |
| 2 | 512 |
| 4 | 256 |

每个 N 跑 `ddp`、`diloco`、`sparta` 三个策略，`max_steps=150`，`val_size=2000`，`val_interval=25`。

**为什么固定全局 batch：**

如果 N 变了但每节点 batch 不变，全局 batch 也会变，收敛曲线就不可比。固定 `batch_size × num_nodes = 1024`，才能把差异主要归因于节点数和通信策略。

**注意 N=1 的后端差异：**

- N=1 时 EXO Gym 会自动走 NCCL；
- N>1 时走 Gloo。

所以跨 N 的墙钟时间对比要谨慎。严谨做法是只比较 N=2/4/8（都是 Gloo），N=1 只作为收敛基线。

**8 节点可选：**

```bash
python experiments/mnist_exogym.py \
  --num-nodes 8 --max-steps 50 --batch-size 128 --minibatch-size 32 \
  --val-size 1024 --val-interval 25 --strategies ddp --run-tag n8 --plot
```

8GB 显存下 8 个进程很容易 OOM 或非常慢；先确保 2/4 节点稳定。

---

## 7. 代码逐模块解释

### 7.1 `miniddp/comm.py`：进程组和参数广播

核心函数：

```python
def init_process_group(rank, world_size, port=None, backend="gloo"):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
```

- `MASTER_ADDR` / `MASTER_PORT` 是所有 rank 的会合地址。
- 端口由父进程用 `find_free_port()` 找空闲端口，避免 `Address already in use`。
- `backend="gloo"` 同时支持 CPU tensor 和单卡 CUDA tensor；NCCL 在单卡多 rank 下会报 duplicate GPU。

参数广播：

```python
for param in model.parameters():
    dist.broadcast(param.data, src=0)
for buffer in model.buffers():
    dist.broadcast(buffer.data, src=0)
```

- 每个 rank 的模型初始化可能不同（尤其是不同进程的随机种子），广播保证所有人从 rank 0 的权重开始。
- buffer 也要广播；否则 BatchNorm 的 running stats 可能不一致。
- 注意：DDP 不会自动同步 BatchNorm 的统计量；它只广播 buffer。多 rank 训练 CNN 时，BN 统计量仍会在各 rank 上独立更新。做严格对照时用 MLP 或 `SyncBatchNorm`。

### 7.2 `miniddp/reducer.py`：bucket + hook + async all-reduce

这是 MiniDDP 的核心。

**第一步：按反向参数顺序分桶**

```python
for param in reversed(self.params):
    groups[(param.device, param.dtype)].append(param)
```

- 反向传播从最后一层开始，后层的梯度先就绪。
- 按 `reversed(model.parameters())` 分桶，能让后面的 bucket 先 all-reduce，和前层仍在进行的反向计算重叠。
- `(device, dtype)` 分组保证同一个 bucket 里的 tensor 可以拼成一个连续 buffer。

**第二步：给每个参数注册 hook**

```python
param.register_post_accumulate_grad_hook(self._make_hook(bucket))
```

- `post_accumulate_grad_hook` 在该参数的梯度累加完成后触发。
- 相比直接 `param.register_hook`，它拿到的 `param.grad` 已经是累加后的结果，适合梯度累积场景。
- hook 里做三件事：把 `param.grad` 拷进 bucket view、标记该参数 ready、当 bucket 内所有参数都 ready 时启动 all-reduce。

**第三步：bucket 扁平化**

```python
view = self.buffer[offset:offset + n].view_as(param)
self.param_to_view[id(param)] = view
```

- 每个 bucket 是一块连续的 `torch.empty(...)`。
- 每个参数的梯度对应 buffer 里的一段 view。
- 这样 all-reduce 一次处理整个 bucket，而不是每个参数发一次集合通信，减少小 tensor 的启动延迟。

**第四步：异步 all-reduce**

```python
bucket.work = dist.all_reduce(
    bucket.buffer, op=dist.ReduceOp.SUM, async_op=True
)
```

- `async_op=True` 让通信在后台进行，反向传播可以继续。
- 不要在 hook 里 `work.wait()`，否则就退化成同步 all-reduce，失去 overlap。
- 用 `SUM` 而不是 `AVG`：Gloo 对 `AVG` 的支持不如 `SUM` 稳，最后统一除以 `world_size`。

**第五步：`finalize()` 等待并写回**

```python
bucket.work.wait()
bucket.buffer.div_(self.world_size)
for param, view in zip(bucket.params, bucket.views):
    param.grad.copy_(view)
```

- 在 `optimizer.step()` 之前必须等待所有 bucket 完成。
- 先除以 `world_size`，再把结果写回每个参数的 `.grad`。
- 写回后 `zero_grad()` 才能安全进入下一步。
- 如果要做梯度裁剪，也必须先 `finalize()` 再裁剪。

**第六步：`no_sync()`**

```python
if not self.sync_enabled:
    return
```

- `no_sync` 期间 hook 直接返回，`.grad` 继续在 PyTorch 的 autograd 里累加。
- 下一次同步 backward 时，hook 把**累积后的** `.grad` 拷进 bucket，再 all-reduce。
- 因此 `no_sync` 只应该用于中间 micro-batch；最后一个 micro-batch 必须在同步上下文里做 backward。

**第七步：未使用参数处理**

```python
if len(bucket.ready_ids) != len(bucket.params):
    raise RuntimeError("bucket is not ready ...")
```

- 如果某个参数没有参与 forward/backward，它的 hook 不会触发，bucket 永远不满。
- 当前 MiniDDP 不支持 `find_unused_parameters`，所以直接报错，避免进程 hang。
- 后续可以扩展：在 backward 结束后遍历参数，把未使用参数的 `.grad` 置零并标记 ready。

### 7.3 `miniddp/ddp.py`：对外 API

```python
ddp = MiniDDP(model, optimizer, bucket_cap_mb=1.0)
loss = criterion(ddp(x), y)
loss.backward()
ddp.step()      # finalize + optimizer.step()
ddp.zero_grad()
```

- `MiniDDP` 不是 `nn.Module` 子类，`__call__` 直接转发给内部 model。
- 构造时调用 `broadcast_model`，对齐所有 rank 的初始权重。
- `step()` 先 `reducer.finalize()`，再 `optimizer.step()`；这样同步点非常明确。
- `no_sync()` 用上下文管理器临时关闭 all-reduce，支持梯度累积。
- `__getattr__` 把 `train()`、`eval()`、`parameters()` 等属性转发给内部 model，用起来更像 DDP。

### 7.4 `examples/train_mnist.py`：训练循环

**数据划分：**

```python
train_sampler = DistributedSampler(
    train_ds, num_replicas=world_size, rank=rank,
    shuffle=True, seed=args.seed, drop_last=True,
)
```

- 每个 rank 只看到数据的一个子集，避免重复样本。
- `drop_last=True` 保证各 rank 的 local batch 大小一致；否则梯度平均会偏向某些 rank。
- `seed` 固定，保证不同实现使用相同的 shard 顺序。

**训练一步：**

```python
optimizer.zero_grad(set_to_none=True)
for micro in range(args.accum_steps):
    x, y = stream.next()
    loss = F.cross_entropy(model(x), y)
    if micro < args.accum_steps - 1:
        with sync_wrapper.no_sync():
            loss.backward()
    else:
        loss.backward()
ddp.step()
```

- `accum_steps=1` 时就是普通 DDP。
- `accum_steps>1` 时，前 `accum_steps-1` 个 micro-batch 在 `no_sync` 里累积梯度；最后一个在同步上下文里触发 all-reduce。
- `ddp.step()` 等待通信完成，再更新参数。

**验证：**

```python
for tensor in (correct, total, loss_sum):
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
```

- 每个 rank 只评估自己的 shard；all-reduce 后得到全局 loss 和 accuracy。

**保存：**

```python
state = unwrap_state_dict(model, args.impl)
torch.save({k: v.detach().cpu() for k, v in state.items()}, output)
```

- `torch DDP` 的 state dict 在 `.module` 下，MiniDDP 在 `.model` 下，统一 unwrap 后保存。
- 保存到 CPU，避免 CUDA tensor 序列化问题。

### 7.5 `tests/compare_impls.py`：为什么能证明实现正确

它做的是**端到端对照**，不是只看 all-reduce 的数值：

1. 父进程先跑一遍 MiniDDP：`mp.spawn` 出 `world_size` 个 rank，训练 `steps` 步，rank 0 保存 state dict。
2. 父进程再用完全相同的 seed、数据、优化器跑一遍 torch DDP。
3. 比较两个 state dict 的每个参数，打印最大绝对误差。

如果 MiniDDP 的 hook 时机、bucket 顺序、除以 `world_size`、`no_sync` 累积逻辑有任何一个错了，最终参数都会明显偏离。

### 7.6 `experiments/mnist_exogym.py`：对上游示例做了什么改造

- 把硬编码的 `num_nodes=4`、`max_steps=150`、`batch_size=256` 变成 CLI 参数。
- 只跑指定策略，方便先做 `ddp` 冒烟测试。
- `Trainer(model, train_ds, val_ds, device=device)`：设备在构造器里传，`fit()` 里不传。
- 显式传 `minibatch_size`，跳过 EXO Gym 的自动显存探测，避免第一次运行卡很久。
- `val_size` 降到 2000、`val_interval` 改成 25，减少验证集 all-reduce 的开销。
- 默认使用更小的 CNN，降低 8GB 显存下 4 节点的 OOM 风险；`--model official` 可以切回上游 CNN。
- 最后把 run 配置写到 `results/*.json`，并可选画 `validation.csv` 的曲线。

---

## 8. MiniDDP 实验建议

### 8.1 正确性

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest tests -q
```

通过标准：所有 case 都打印 `max_abs_diff`，且不超过 `1e-5`。

### 8.2 bucket 大小对通信的影响

```bash
for mb in 0.001 0.01 0.1 1.0 10.0; do
  python examples/train_mnist.py \
    --impl miniddp --world-size 2 --device cpu \
    --max-steps 50 --batch-size 64 --bucket-mb "$mb" \
    --log-interval 10
done
```

观察：

- 终端会打印 `allreduce_calls`、`payload`、`wait`。
- `bucket-mb` 很小时，bucket 数量多、all-reduce 调用次数多，CPU 上的启动延迟占比高。
- `bucket-mb` 很大时，bucket 数量少、单次通信量大，overlap 可能变差。
- 中间某个值通常是折中；具体数值受模型大小、CPU 线程、Gloo 实现影响。

### 8.3 `no_sync` 梯度累积

```bash
python examples/train_mnist.py \
  --impl miniddp --world-size 2 --device cpu \
  --max-steps 20 --batch-size 32 --accum-steps 4
```

注意：有效全局 batch 变成 `batch_size × accum_steps × world_size`。做收敛对比时要保持一致。

### 8.4 与 torch DDP 对比

```bash
python examples/train_mnist.py --impl torchddp --world-size 2 --device cpu \
  --max-steps 50 --batch-size 64 --bucket-mb 1.0
```

MiniDDP 和 torch DDP 的最终 loss/accuracy 应该接近；如果差异大，先用 `tests/compare_impls.py` 做逐步数值对照。

---

## 9. 常见问题

| 现象 | 原因 | 处理 |
|---|---|---|
| `Duplicate GPU detected` | 单卡上强制 NCCL 多 rank | 用 `--backend gloo`；EXO Gym 的 N>1 会自动选 Gloo |
| `bucket is not ready` | 有参数没参与 forward/backward | 确保所有参数都被使用；暂不支持 `find_unused_parameters` |
| `finalize() cannot be called inside no_sync()` | 在 `no_sync` 上下文里调了 `step()` | 最后一个 micro-batch 要在同步上下文里 backward |
| CUDA OOM | 虚拟节点太多 / batch 太大 | 降 `--world-size`、`--batch-size`、`--minibatch-size` |
| `Address already in use` | 端口冲突 | 代码会自动找空闲端口；仍有问题就杀掉残留进程 |
| `EXO Gym is not installed` | 环境不对 | `conda activate exogym`，确认 `python -c "import exogym"` |
| `plt.show()` 卡住 | 无显示环境 | 用 `MPLBACKEND=Agg`；脚本里已改成 `savefig` |
| `URLError` / `ConnectionResetError` | MNIST 下载被网络重置 | 父进程已预下载；若仍失败，按下面“MNIST 手动修复”补齐文件 |
| 测试很慢 | 每个 case 都 spawn 多个进程 | 先只跑单个 case：`python tests/compare_impls.py --world-size 2 --steps 1` |
| 结果每次略有不同 | 浮点归约顺序不同 | 固定 seed；比较时用 `atol=1e-5`，不要要求 bitwise 相等 |

### MNIST 手动修复

`train_mnist.py` 现在会在 `mp.spawn` 之前由父进程下载一次，worker 使用 `download=False`，避免多个 rank 并发写同一个压缩包。如果网络仍然中断，手动补齐四个文件：

```bash
cd ~/projects/ai-infra/Distribution-Training/EXO-Gym
mkdir -p data/MNIST/raw
base=https://ossci-datasets.s3.amazonaws.com/mnist
for f in train-images-idx3-ubyte.gz train-labels-idx1-ubyte.gz \
         t10k-images-idx3-ubyte.gz t10k-labels-idx1-ubyte.gz; do
  curl -L --retry 8 --retry-delay 2 --connect-timeout 15 \
    -o "data/MNIST/raw/$f" "$base/$f"
done
md5sum data/MNIST/raw/*.gz
```

四个文件的期望 MD5：

```text
f68b3c2dcbeaaa9fbdd348bbdeb94873  train-images-idx3-ubyte.gz
d53e105ee54ea40749a09fcbcd1e9432  train-labels-idx1-ubyte.gz
9fb629c4189551a2d022fa330f9573f3  t10k-images-idx3-ubyte.gz
ec29112dd5afa0611ce80d1b7f02629c  t10k-labels-idx1-ubyte.gz
```

补齐后重新运行训练命令即可；torchvision 会校验 MD5 并自动解压。

---

## 10. 这个项目可以写什么

1. **DDP 到底同步了什么**：参数初始化广播、梯度 all-reduce、除以 world_size、optimizer step 的先后顺序。
2. **bucket 为什么有用**：小 tensor 多次集合通信的启动延迟 vs 大 bucket 的带宽/重叠权衡。
3. **hook 的时机**：为什么用 `post_accumulate_grad_hook`，为什么不在 `optimizer.step()` 里统一 all-reduce。
4. **async 的价值**：`async_op=True` + `finalize()` 如何让通信和反向计算重叠。
5. **no_sync 的语义**：梯度累积时为什么可以跳过通信，最后一次 backward 如何补上。
6. **单卡模拟的边界**：Gloo + 共享 GPU 能验证机制，但不能代表 NCCL/多卡性能；EXO Gym 的 AllReduce 策略是逐参数同步，和 PyTorch DDP 的 bucket 实现不同。
7. **算法层对比**：DDP vs DiLoCo vs SPARTA 在固定全局 batch 下的 loss/step、loss/wall-clock 和最终精度。

---

## 11. 推荐执行顺序

```text
1. 建环境 + 安装 torch/torchvision
2. pip install -e .
3. bash scripts/run_miniddp_cpu.sh
4. bash scripts/run_miniddp_gpu.sh
5. 安装 EXO Gym
6. bash scripts/run_exogym_smoke.sh
7. bash scripts/run_exogym_scaling.sh
8. 分析 logs/ 和 results/，写实验记录
```

如果任何一步报错，先把完整报错贴出来，再按“常见问题”表定位。
