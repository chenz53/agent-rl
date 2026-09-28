# 2026-09-28：远端示范

## 用户原话

> 可以连接到remote server做示范: ssh vp

## 本轮操作与教学摘要（非逐字聊天记录）

- 通过指定 SSH alias 连接，确认 1 张 H200 和可用资源。
- 在 `/home/user/nanoAstra` 建立执行镜像与独立虚拟环境，不修改既有项目环境。
- 安装兼容驱动的 torch 2.9.1+cu126 与工具/算法依赖，运行真实 verl tool registry 与 CPU GRPO probe，通过。
- 用缓存的 Qwen3-0.6B 执行一次真实参数更新。查询由脚本固定，模型按 A/B 限制分布采样；使用原生 verl advantage、policy loss 和 PyTorch SGD。
- 受限策略 P(B)：26.0893%→26.9492%；loss 约 1.49e-8，但梯度范数约 142.12。
- 教学重点：loss 的当前数值为 0 不代表梯度为 0；old_log_probs 是常量，current_log_probs 保留梯度；采样与训练必须用同一个受限策略。
- 已取回结果和依赖版本；没有保存或覆盖模型权重，任务已退出。后续仍需实现自主工具调用和完整多轮训练。

教材：[第四讲](../lessons/004-first-real-gpu-update.md)。实测记录：[实验 002](../../experiments/002-remote-h200.md)。
