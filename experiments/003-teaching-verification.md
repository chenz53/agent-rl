# 实验 003：教材示例的数值与代码验证

日期：2026-09-28。目的：验证重写后的讲解与代码/数值一致，不进行新的模型训练。

## 数值复核

在 vp 原有 nanoAstra 环境运行 `scripts/verify_lesson_math.py`，直接导入固定 verl 的 advantage 与 policy loss，不复制替代实现。

结果保存到 [003-lesson-math.json](results/003-lesson-math.json)。所有断言通过，覆盖：

- 本次 16 个实际 reward 的均值 0.125、样本标准差 0.341565、advantage −0.365961/+2.561730。
- 等价两 logit 参数化下，loss≈0、梯度约 [+0.160108,−0.160108]。
- 错误地让 old/current log-probability 共用计算图，梯度为 [0,0]。
- 5 个正/负 advantage 与 clipping/dual-clip 例子。
- 工具位置 mask=0、组内全对/全错的 outcome advantage=0。
- 同 prompt 分组与错误跨 prompt 分组的差别。
- 温度、二项采样概率、梯度范数裁剪比例和查询收益的计算。

等价 logits 是从已记录策略概率重建的，不是读取 Qwen 原始 logits；这里的 logit 梯度不是全模型梯度范数。

## 文档片段执行

`scripts/check_lesson_snippets.py` 从 4 课中提取 Python 代码。所有 42 个片段通过 AST 解析；明确选择了 29 个可执行片段，在每课独立 namespace 中按顺序运行，其中第三课保留 registry→tool 调用需要的上下文。

输出保存到 [003-lesson-snippets.json](results/003-lesson-snippets.json)，已核对文中预期结果。其余是依赖完整训练上下文的源码摘录、方法片段或故意展示的错误代码，未声称这些能独立运行。

片段选择采用显式编号。编辑文档增加/移动代码块后，需要审查选择列表；语法和运行通过本身不能证明 snippet 的教学语义没有变化。

## 复现

在远端仓库根目录：

```sh
.venv/bin/python scripts/verify_lesson_math.py
.venv/bin/python scripts/check_lesson_snippets.py
```

没有加载 Qwen、没有 GPU 参数更新、没有更改实验 002 的原始结果。这里只验证教材计算、工具调用和解释；不支持任何临床有效性结论。
