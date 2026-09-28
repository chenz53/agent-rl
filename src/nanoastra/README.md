# 实现目录

- `clinical.py`：病例信息边界、查询与 verl 数据行。
- `reward.py`：自定义终局评分。
- `verl_tool.py`：真实 BaseTool 子类，已在远端通过 registry/lifecycle smoke test。
- `upstream.py`：检查并载入固定上游源码。

领域 demo 和工具/GRPO 集成已验证；受限 A/B 的真实 Qwen 参数更新见实验 002。完整多轮 Agent/训练尚未实现。
