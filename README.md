# nanoAstra

从 0 到 1 搭建并训练一个简易临床推理 Agent，同时系统学习 Agentic Reinforcement Learning。

学习者具有多年基础模型训练经验；重点是序贯决策、训练环境、奖励、信用分配、训练系统与科学评估。

## 从这里开始

- [学习路线](docs/roadmap.md)
- [第一课：把临床推理变成可训练的问题](docs/lessons/001-clinical-reasoning-as-an-environment.md)
- [第二讲：一条临床推理轨迹如何变成训练信号](docs/lessons/002-one-clinical-trajectory.md)
- [最小 Agent 设计](docs/design/001-minimal-agent.md)
- [讨论记录与索引](docs/discussions/README.md)
- [参考资料](docs/references/README.md)
- [进度与决策](PROGRESS.md)

## 目录约定

| 路径 | 内容 |
| --- | --- |
| `docs/lessons/` | 教学材料、推导、练习 |
| `docs/discussions/` | 讨论归档、问题与决定 |
| `docs/design/` | 环境、训练和评估设计 |
| `docs/references/` | 论文链接与阅读笔记 |
| `src/nanoastra/` | 后续 Agent、环境与训练实现 |
| `tests/` | 后续有实际验证价值的测试 |
| `experiments/` | 实验配置、结果摘要和失败分析 |

## 当前状态

已建立本地 Git 仓库与课程框架；尚未实现 Agent、接入数据、运行训练或创建远程仓库。目录骨架不代表已有可运行的软件。

项目用于研究与学习。起步实验使用合成或明确允许使用的去标识化病例，不将真实患者可识别信息、凭据或模型权重提交到 Git。
