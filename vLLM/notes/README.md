# vLLM 学习资料索引

```
vLLM/notes/
├── papers/   # 论文 PDF
└── repos/    # 论文对应的参考实现源码（tarball 快照）
```

## 论文（papers/）

| 文件 | 论文 | 发表 | 为什么读 |
|---|---|---|---|
| `pagedattention_sosp23.pdf` | Efficient Memory Management for Large Language Model Serving with PagedAttention | SOSP 2023 | vLLM 的核心：分页 KV cache、block table、COW |
| `orca_osdi22.pdf` | Orca: A Distributed Serving System for Transformer-Based Generative Models | OSDI 2022 | continuous batching / iteration-level scheduling 的源头 |
| `flashattention_neurips22.pdf` | FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness | NeurIPS 2022 | IO-aware tiling + online softmax，推理 attention 的基础 |
| `flashattention2_2023.pdf` | FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning | 2023 | 现代推理后端实际使用的版本 |
| `sglang_radixattention_2023.pdf` | SGLang: Efficient Execution of Structured Language Model Programs | 2023 | RadixAttention / 前缀树复用，对照 vLLM 的 prefix caching |

推荐阅读顺序：**Orca → PagedAttention → FlashAttention-2 → SGLang**（先调度、再显存、再算子、再前缀复用）。

## 源码（repos/）

| 目录 | 仓库 | 对应论文 |
|---|---|---|
| `vllm/` | https://github.com/vllm-project/vllm | PagedAttention |
| `flash-attention/` | https://github.com/Dao-AILab/flash-attention | FlashAttention 1/2 |
| `sglang/` | https://github.com/sgl-project/sglang | SGLang / RadixAttention |

说明：

- **Orca 没有官方开源实现**。它是 OSDI'22 的研究系统，社区仓库均为非官方复现；其 iteration-level scheduling 思想在 vLLM 的 `Scheduler` 中延续，读 vLLM 调度器即可。
- `repos/` 下是 GitHub `main` 分支的 tarball 快照（无 `.git` 历史），下载时间与 commit 记录见 `repos/MANIFEST.md`。
- 阅读 vLLM 源码时以 `vllm/v1/`（V1 引擎）为主，网上大量博客描述的是已废弃的 v0 路径。
