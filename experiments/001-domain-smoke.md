# 实验 001：领域函数与评分 smoke test

日期：2026-09-28。环境：macOS arm64，系统 Python 3.9.6。没有模型、采样或优化器。

执行：`python3 scripts/demo_case.py` 与 `PYTHONPATH=src python3 -m unittest discover -s tests -v`。

结果：6 项测试通过。允许查询的字段正常返回；label 查询被拒绝；缺失信息返回 unavailable；独立 record 不共享修改；正确答案得 1，错误答案与非法格式得 0；不适用的数据源和标签会显式报错。

另对 9 个 Python 文件进行 AST 语法解析并通过。曾尝试 compileall，但系统 Python 将缓存定向到不可写目录，失败；之后使用不写缓存的语法解析，不把它计为 runtime 集成验证。

未运行：verl import、BaseTool/YAML registry、真实 tokenizer、完整 ToolAgentLoop、GRPO CPU probe、LLM 推理与 GPU 训练。本地依赖安装请求被拒绝，因此停止安装。

实验结论仅限领域代码的约定，不支持任何模型能力结论。
