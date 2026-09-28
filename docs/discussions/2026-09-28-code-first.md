# 2026-09-28：转向代码驱动教学

## 用户原话

> 结合代码讲解是最好的 这样对实践更有帮助 我们可以选取一些public repo/framework来作为infra基础这样我们就不用重复造轮子

对算力问题的用户回答：

> 已有可使用的远程 GPU

## 本轮落实（摘要，非聊天逐字记录）

- 教学以真实框架源码与实验为主线，选择 verl 作为当前底座。
- 固定上游 SHA 并下载原始源码；实现 nanoAstra 的合成病例、记录查询、数据行与终局评分。
- 编写真实 BaseTool 子类、工具 YAML、CPU 集成 probe，但尚未运行 verl runtime。
- 标准库领域 demo 已运行，6 项测试通过。
- 本地 CPU 依赖安装被用户拒绝，没有继续安装；代码与文档保留未验证标记。
- 发现当前 ToolAgentLoop 的 BaseTool 生命周期是逐调用 create/execute/release；不能据名称假设其状态跨轮保存。
- 远程 GPU 的型号、数量、显存和环境尚未提供；未访问远端或启动训练。

教材全文：[第三讲](../lessons/003-verl-code-walkthrough.md)。设计决定：[基础设施](../design/002-infrastructure.md)。
