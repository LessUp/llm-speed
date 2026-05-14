# Component Library

This document describes the Vue components available in the VitePress documentation site.

## Usage

Components are auto-registered globally. Use them directly in markdown:

```markdown
<KernelCard
  title="FlashAttention"
  icon="⚡"
  description="Memory-efficient attention computation"
  memory="O(N)"
  :speedup="128"
/>
```

## Components

### KernelCard

Display kernel information with performance metrics.

| Prop | Type | Required | Description |
|------|------|----------|-------------|
| `title` | `string` | Yes | Kernel name |
| `icon` | `string` | Yes | Emoji or icon |
| `description` | `string` | Yes | Brief description |
| `memory` | `string` | No | Memory complexity |
| `speedup` | `number` | No | Speedup factor |
| `features` | `string[]` | No | Feature tags |

---

### PerfChart

Performance visualization using Chart.js.

| Prop | Type | Required | Description |
|------|------|----------|-------------|
| `type` | `'bar' \| 'line'` | No | Chart type (default: `'bar'`) |
| `title` | `string` | No | Chart title |
| `labels` | `string[]` | Yes | X-axis labels |
| `datasets` | `Dataset[]` | Yes | Data datasets |

---

### MemoryCompare

FlashAttention memory comparison table. **No props required** - data loaded from `benchmarks.json`.

---

### GPUSupport

GPU architecture support matrix. **No props required** - data loaded from `gpu-support.json`.

---

### CodeDemo

Code snippet with filename header and copy button.

| Prop | Type | Required | Description |
|------|------|----------|-------------|
| `filename` | `string` | Yes | Filename to display |
| `code` | `string` | Yes | Code content |
| `language` | `string` | No | Syntax highlighting (default: `'python'`) |
| `colabUrl` | `string` | No | Google Colab link |

---

### CodeDiff

Side-by-side code comparison for optimization visualization.

| Prop | Type | Required | Description |
|------|------|----------|-------------|
| `title` | `string` | No | Diff title |
| `leftLabel` | `string` | Yes | Left pane label |
| `rightLabel` | `string` | Yes | Right pane label |
| `leftCode` | `string` | Yes | Left pane code |
| `rightCode` | `string` | Yes | Right pane code |
| `lang` | `string` | No | Syntax highlighting (default: `'cuda'`) |
| `highlight` | `'left' \| 'right' \| 'both'` | No | Highlight side (default: `'right'`) |

---

### AlgorithmCard

Algorithm details with complexity badges.

| Prop | Type | Required | Description |
|------|------|----------|-------------|
| `title` | `string` | Yes | Algorithm name |
| `description` | `string` | No | Description |
| `timeComplexity` | `string` | Yes | Time complexity |
| `spaceComplexity` | `string` | Yes | Space complexity |
| `code` | `string` | No | Code snippet |
| `lang` | `string` | No | Syntax highlighting (default: `'cuda'`) |

---

### RoadmapTimeline

Optimization progression timeline.

| Prop | Type | Required | Description |
|------|------|----------|-------------|
| `stages` | `Stage[]` | No | Timeline stages (uses defaults if omitted) |

## Data Sources

| Component | Data File | Loading Method |
|-----------|-----------|----------------|
| MemoryCompare | `data/benchmarks.json` | `useFlashAttentionMemory()` |
| GPUSupport | `data/gpu-support.json` | `useGpuSupport()` |

To modify data for these components, edit the corresponding JSON files.

## Adding New Components

1. Create the component in `.vitepress/components/`
2. Define Props interface in the component's `<script setup>`
3. Add type definition to `types/components.ts`
4. Document in this file
