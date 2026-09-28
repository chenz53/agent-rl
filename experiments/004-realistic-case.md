# 实验 004：分阶段临床记录回放

日期：2026-09-28。病例：`ed-0001`，完全合成。目的：验证较丰富病历能复用现有查询工具，并隔离未来结果和教师答案。没有运行模型推理或参数更新。

## 验证结果

- 本地及 vp 标准库测试共 11 项通过，含新增的 5 项阶段投影测试。
- vp 现有固定版本 verl registry/BaseTool 成功加载新 YAML，执行两个阶段的查询并清理实例。
- 初诊 CTPA 返回尚未完成；教师字段返回 `invalid_field`；后续阶段返回 CTPA 阳性报告，下肢超声仍未完成。
- 原 A/B 工具及真实 CPU GRPO probe 回归通过。
- 第五讲全部 3 个 Python 片段按标明的上下文顺序执行通过；这不改变实验 003 前四课 42/29 的统计。

证据：[远端完整输出](results/004-realistic-case.txt)、[教材片段输出](results/004-lesson-snippets.json)、[测试代码](../tests/test_case_replay.py)。

## 复现

本地仓库根目录：

```sh
PYTHONPATH=src python3 -m unittest discover -s tests -v
python3 scripts/demo_realistic_case.py --stage initial
python3 scripts/demo_realistic_case.py --stage followup --fields ctpa_report leg_ultrasound
```

vp 的 `/home/user/nanoAstra` 内，使用现有 `.venv/bin/python`，给 demo 命令加 `--verl`；原接口回归为 `.venv/bin/python scripts/probe_verl.py`。

## 解释边界

这些检查证明程序的数据边界和接口行为，不证明病例经过临床验证，也不证明模型具有临床能力。病例是固定记录的两个时间切片；没有模拟检查下单、耗时、患者状态变化或治疗效果。未将原 A/B 评分器复用成医学 reward。参考诊断与教学解释尚需独立临床专家审核后，才能考虑用于严肃评测。
