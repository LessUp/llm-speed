# OpenSpec Change Archive

本目录记录已完成的变更规格。

## 归档列表

| 变更 | 状态 | 归档日期 | 说明 |
|------|------|----------|------|
| 2026-04-23-bf16-support | ⏸️ Deferred | 2026-04-23 | BF16 精度支持，待后续版本实施 |
| 2026-04-23-final-verification | ✅ Completed | 2026-04-23 | 最终验证与确认 |
| 2026-04-23-flashattention-backward | ⏸️ Deferred | 2026-04-23 | FlashAttention 反向传播，待后续版本实施 |
| 2026-04-23-project-closeout | ✅ Completed | 2026-04-23 | 项目收尾与清理 |
| 2026-04-27-final-cleanup | ✅ Completed | 2026-04-27 | 最终清理残留问题 |

## 状态说明

- ✅ **Completed**: 变更已实施完成
- ⏸️ **Deferred**: 变更已延迟，待后续版本实施

## 延迟功能

延迟功能的代码补丁存储在 `../../../deferred/` 目录中：

```bash
# 查看 BF16 支持补丁
cat ../../../deferred/bf16-support.patch

# 应用补丁（如需实施）
git apply ../../../deferred/bf16-support.patch
```

> 详见 `AGENTS.md` 中的延迟积压声明。
