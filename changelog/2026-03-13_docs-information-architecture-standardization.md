# 2026-03-13 文档与 Pages 信息架构规范化

## 变更背景

- 按批次 A 的统一要求，收敛仓库入口与文档入口职责。
- 此前 `README.md`、`README.zh-CN.md` 与 `index.md` 都在重复介绍项目能力、架构与使用方式，仓库首页和文档首页边界不清。
- Pages workflow 只监听 `main`，而仓库当前实际分支为 `master`，会导致文档更新无法自动触发部署。

## 导航与目录调整

- 保持现有 Jekyll 目录结构，继续使用根目录 `index.md` 作为文档首页。
- 将 `README.md` / `README.zh-CN.md` 收敛为仓库入口，仅保留项目定位、最小构建命令与文档链接。
- 将 `docs/deepwiki.md` 明确作为“使用指南”页面，负责承载更完整的 CUDA 内核与优化路线说明。
- 继续使用 `CONTRIBUTING.md` 作为开发指南，`changelog/` 作为归档入口。

## 首页调整

- `index.md` 改写为文档导读页，新增项目定位、适合谁、从哪里开始、推荐阅读路径与核心入口表。
- 首页不再重复完整功能列表和大段项目结构说明，而是把读者引导到 `README`、`docs/deepwiki.md`、`CONTRIBUTING.md` 与 `changelog/`。
- 站点首页保留 CI / Pages / License 等核心徽章，方便在文档入口直接判断工程状态。

## Pages / Workflow 调整

- `.github/workflows/pages.yml` 的推送触发分支从仅 `main` 扩展为 `master, main`。
- 保持现有 `actions/configure-pages`、`actions/upload-pages-artifact` 与 `actions/deploy-pages` 链路不变，只修正分支触发与信息架构相关内容。

## 验证结果

- 已人工确认仓库远程地址为 `https://github.com/LessUp/llm-speed.git`，与站点链接一致。
- 已人工检查 `README`、`index.md`、`docs/deepwiki.md`、`CONTRIBUTING.md` 与 `changelog/` 的链接关系均对应现有文件。
- 已确认 `docs/` 当前只有 `docs/deepwiki.md`，因此本次首页将其明确收敛为核心使用指南入口。
- 本次未运行本地 Jekyll 构建；后续可在具备 Ruby / Jekyll 环境时补充静态构建验证。

## 后续待办

- 可后续将 `docs/deepwiki.md` 拆分为更细的 `guide/`、`reference/` 页面，但当前先保持最小变更。
- `changelog/CHANGELOG.md` 目前仍带有转义换行文本，后续可单独整理为标准 Markdown 页面。
