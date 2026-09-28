# 基础设施选择：verl + nanoAstra 任务扩展

日期：2026-09-28。用户要求代码驱动教学、复用公开框架，随后授权 `ssh vp`：1×H200。独立执行目录 `/home/user/nanoAstra`；环境与结果见实验 002。

## 选择与范围

| 框架 | 本项目判断 | 核验来源 |
| --- | --- | --- |
| verl | 当前主线；可以直接追踪多轮采样到 advantage 和 policy loss，适合已有模型训练经验的学习者 | 固定源码与官方多轮文档 |
| rLLM | 保留为 Agent 接口层的比较对象；本轮未核验其运行接口，不作为已验证依赖 | 官方仓库 README |
| Agent Lightning | 保留为执行与训练解耦的后续阅读；当前先减少需要跨越的抽象层 | 官方仓库与教程 |

这是基于教学目标的选择，不是吞吐或效果排名。未开展框架性能比较。

- [verl 官方仓库](https://github.com/verl-project/verl)
- [verl 多轮文档](https://verl.readthedocs.io/en/latest/sglang_multiturn/multiturn.html)
- [rLLM 官方仓库](https://github.com/agentica-project/rllm)
- [Agent Lightning 官方仓库](https://github.com/microsoft/agent-lightning)

## 版本与职责

固定 verl commit：`00094bd9cd3fef9cf8903daf60ea4a2bcf832efc`，记录在 `upstream.lock.json`。这是源码阅读与适配锚点，不是已经完成 GPU 验证的训练发行版。

上游原样保存在 Git 忽略的 `.upstream/verl/`，保留原始许可证；不把整个上游复制进 nanoAstra 提交。另一台机器可通过 `scripts/fetch_upstream.py` 获取同一 commit。取回上游不自动安装或执行它。

- verl：Agent loop、推理后端、token 与 mask、GRPO/PPO、分布式更新。
- nanoAstra：病例的可见信息边界、工具、任务数据、奖励、评估与实验。

不另外实现 rollout 调度器、GRPO 优化器或一个新的通用 Agent 框架。

## 当前验证边界

| 项目 | 状态 |
| --- | --- |
| 固定上游源码 | 已下载并阅读；未修改 |
| 领域查询、评分和数据行生成 | 标准库脚本已运行；6 项测试通过 |
| BaseTool 适配器与 YAML | 在远端通过真实 registry/lifecycle probe |
| CPU GRPO 集成 probe | 远端通过 |
| 真实模型与单步 GPU 更新 | Qwen3-0.6B、受限 A/B 判断已通过 |
| 完整 ToolAgentLoop、分布式 trainer | 未运行 |
| 版本记录 | 上游 SHA、模型 snapshot、远端包版本 freeze 已保存；没有 uv.lock |

本地依赖安装请求被用户拒绝，因此没有换途径安装依赖。`pyproject.toml` 的 CPU extra 是后续候选环境，不是已验证安装方案。现有标准库示例支持本机 Python 3.9；verl 集成目标为 Python 3.11/3.12。

## 远端下一步

已完成工具/GRPO probe 和受限动作的模型单步更新。接下来让模型自主查询，验证完整多轮 tokenization、mask、终止与评分，再接入 trainer。不要把 probe 或单步更新写成完整多轮训练已通过。
