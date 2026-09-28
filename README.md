# nanoAstra

从 0 到 1 搭建并训练一个简易临床推理 Agent，同时系统学习 Agentic Reinforcement Learning。

学习者具有多年基础模型训练经验；重点是序贯决策、训练环境、奖励、信用分配、训练系统与科学评估。

## 从这里开始

- [学习路线](docs/roadmap.md)
- [第一课：把临床推理变成可训练的问题](docs/lessons/001-clinical-reasoning-as-an-environment.md)
- [第二讲：一条临床推理轨迹如何变成训练信号](docs/lessons/002-one-clinical-trajectory.md)
- [第三讲：沿代码接入 verl](docs/lessons/003-verl-code-walkthrough.md)
- [基础设施选择与验证边界](docs/design/002-infrastructure.md)
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

已实现可运行的合成病例查询与终局评分，6 项标准库测试通过。已固定 verl 源码并编写工具适配器；尚未运行 verl 集成、模型推理或训练，也未创建远程仓库。

从仓库根目录运行，不需要安装新依赖：

```sh
python3 scripts/demo_case.py
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

这是预设动作的领域 demo，不能当成模型 rollout。CPU 依赖安装未获批准，候选依赖尚未解析锁定；GPU 环境等待规格确认。

项目用于研究与学习。起步实验使用合成或明确允许使用的去标识化病例，不将真实患者可识别信息、凭据或模型权重提交到 Git。
