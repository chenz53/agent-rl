# 实验 002：vp 上的 verl 集成与真实 Qwen 更新

日期：2026-09-28。用户明确提供 `ssh vp` 用于远端示范。

## 环境

- 远端执行目录：`/home/user/nanoAstra`；是本地仓库的执行镜像，不是已发布的远程 Git 仓库。
- 独立 Python 环境：`/home/user/nanoAstra/.venv`，Python 3.12.3。
- GPU：1× NVIDIA H200，nvidia-smi 总显存 143771 MiB；驱动 565.57.01。
- PyTorch：2.9.1+cu126；Transformers：5.5.3；tensordict：0.10.0。
- verl：`00094bd9cd3fef9cf8903daf60ea4a2bcf832efc`，没有修改上游源码。
- 模型：Qwen/Qwen3-0.6B，使用已有缓存，snapshot `c1899de289a04d12100db370d81485cdf75e47ca`。
- 完整环境版本：[freeze 文件](results/002-remote-requirements.txt)。这是已安装版本快照，不含包哈希，也不是 GPU 分布式训练全栈认证。

未修改其他项目虚拟环境。此前拒绝的是本地依赖安装，本轮根据远端示范授权在远端独立环境安装。

## 验证结果

1. BaseTool adapter 经真实 verl registry 从 YAML 加载，允许字段正常返回，label 查询被拒绝，调用后实例释放。
2. 真实 CPU GRPO probe：`[1,1,0,0]` 得到约 ±0.866 的 advantage；示意环境 token 为 0；全失败组的 outcome advantage 为 0。
3. 6 项领域测试在远端 Python 3.12 环境通过。
4. Qwen3-0.6B 在 H200 上采样 16 个 A/B 动作，计算真实 verl advantage 与 policy loss，反向传播并执行一个 SGD 步。

原始结果：[JSON](results/002-gpu-update.json)。14 次 A、2 次 B；更新后受限策略 P(B) 从 0.260893 增至 0.269492。梯度范数 142.1208，裁剪至 1；learning rate=1e-4，float32。峰值 allocated 显存 5.676 GiB。

执行后退出进程，没有保存 checkpoint，没有常驻训练任务。检查 nvidia-smi 时显存占用为 4 MiB。完整日志在远端 `outputs/gpu-update.log`。

## 在已准备的 vp 上复现

```sh
ssh vp
cd /home/user/nanoAstra
.venv/bin/python scripts/probe_verl.py
.venv/bin/python scripts/demo_gpu_update.py \
  --model /home/user/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca \
  --seed 42 --samples 16 --temperature 2 --learning-rate 0.0001
```

固定 snapshot 路径避免缓存 main 更新后悄然更换模型。固定 seed 不承诺不同硬件/软件版本的逐 bit 重现。

## 重建依赖的说明

本次先从 PyTorch cu126 官方 index 安装 `torch==2.9.1`，随后安装其余依赖并保留该 torch 版本。使用 freeze 文件重建时，需要提供 cu126 wheel 来源；仅从默认 PyPI 安装 `torch==2.9.1+cu126` 可能找不到包。

可在新的、明确用于 nanoAstra 的环境中使用以下命令；不要覆盖已有环境：

```sh
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python torch==2.9.1 \
  --index-url https://download.pytorch.org/whl/cu126
uv pip install --python .venv/bin/python -r experiments/results/002-remote-requirements.txt
python3 scripts/fetch_upstream.py
```

模型文件需要预先缓存；脚本使用 local_files_only，不会自动下载模型。

## 限制

查询固定，只优化受限 A/B 判断；格式由脚本包装；单病例；单更新；未使用分布式 trainer。尚未验证模型自主工具调用、完整多轮 token mask、vLLM/SGLang 或临床评估。当前结果不能作为临床能力或训练收敛的证据。
