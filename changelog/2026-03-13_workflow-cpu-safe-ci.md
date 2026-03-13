# Workflow CPU-safe CI 调整

日期：2026-03-13

## 变更内容

- 删除主线 CI 中无效的 CUDA 构建 job，仅保留 Hosted Runner 可执行的 Python lint 与 CPU-safe smoke tests
- 移除 `pytest ... || true` 的伪成功逻辑，改为仅在 `not cuda` 无用例可收集时接受退出码 5
- 在测试前加入 `python -m compileall`，确保 Python 源码至少通过语法层面的快速校验

## 背景

该仓库的测试体系主要依赖 CUDA 与自定义扩展，GitHub Hosted Runner 无法承担真实 GPU 验证。本次调整把主线 CI 收敛到仍有信号价值的 CPU-safe 检查，同时避免继续隐藏真实失败。
