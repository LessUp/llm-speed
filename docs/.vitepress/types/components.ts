/**
 * Component Type Definitions
 *
 * This file defines the interfaces for all Vue components in the docs site.
 * Components are organized by category.
 */

// ============================================================================
// Kernel & Performance Components
// ============================================================================

/**
 * KernelCard - Display kernel information with metrics
 * @example
 * <KernelCard
 *   title="FlashAttention"
 *   icon="⚡"
 *   description="Memory-efficient attention"
 *   memory="O(N)"
 *   :speedup="128"
 *   :features="['FP16', 'BF16']"
 * />
 */
export interface KernelCardProps {
  /** Kernel name */
  title: string
  /** Emoji or icon */
  icon: string
  /** Brief description */
  description: string
  /** Memory complexity (optional) */
  memory?: string
  /** Speedup factor (optional) */
  speedup?: number
  /** Feature tags (optional) */
  features?: string[]
}

/**
 * PerfChart - Performance visualization chart
 * @example
 * <PerfChart
 *   type="bar"
 *   title="Memory Usage"
 *   :labels="['1024', '4096', '8192']"
 *   :datasets="[{ label: 'Standard', data: [4, 64, 256] }]"
 * />
 */
export interface PerfChartProps {
  /** Chart type */
  type?: 'bar' | 'line'
  /** Chart title (optional) */
  title?: string
  /** X-axis labels */
  labels: string[]
  /** Data datasets */
  datasets: Array<{
    label: string
    data: number[]
    backgroundColor?: string
    borderColor?: string
  }>
}

/**
 * MemoryCompare - FlashAttention memory comparison table
 *
 * Data source: .vitepress/data/benchmarks.json
 * No props required - data loaded from JSON
 */
export interface MemoryCompareProps {
  // No props - uses useFlashAttentionMemory() composable
}

/**
 * GPUSupport - GPU architecture support matrix
 *
 * Data source: .vitepress/data/gpu-support.json
 * No props required - data loaded from JSON
 */
export interface GPUSupportProps {
  // No props - uses useGpuSupport() composable
}

// ============================================================================
// Code Display Components
// ============================================================================

/**
 * CodeDemo - Code snippet with filename and copy button
 * @example
 * <CodeDemo
 *   filename="example.py"
 *   :code="pythonCode"
 *   language="python"
 *   colabUrl="https://colab.research.google.com/..."
 * />
 */
export interface CodeDemoProps {
  /** Filename to display */
  filename: string
  /** Code content */
  code: string
  /** Language for syntax highlighting */
  language?: string
  /** Google Colab URL (optional) */
  colabUrl?: string
}

/**
 * CodeDiff - Side-by-side code comparison
 * @example
 * <CodeDiff
 *   title="Optimization"
 *   leftLabel="Before"
 *   rightLabel="After"
 *   :leftCode="beforeCode"
 *   :rightCode="afterCode"
 *   highlight="right"
 * />
 */
export interface CodeDiffProps {
  /** Title for the diff (optional) */
  title?: string
  /** Label for left pane */
  leftLabel: string
  /** Label for right pane */
  rightLabel: string
  /** Code in left pane */
  leftCode: string
  /** Code in right pane */
  rightCode: string
  /** Language for syntax highlighting */
  lang?: string
  /** Which side to highlight */
  highlight?: 'left' | 'right' | 'both'
}

/**
 * AlgorithmCard - Algorithm details with complexity
 * @example
 * <AlgorithmCard
 *   title="FlashAttention"
 *   description="Memory-efficient attention"
 *   timeComplexity="O(N²)"
 *   spaceComplexity="O(N)"
 *   :code="kernelCode"
 * />
 */
export interface AlgorithmCardProps {
  /** Algorithm name */
  title: string
  /** Description (optional) */
  description?: string
  /** Time complexity notation */
  timeComplexity: string
  /** Space complexity notation */
  spaceComplexity: string
  /** Code snippet (optional) */
  code?: string
  /** Language for syntax highlighting */
  lang?: string
}

// ============================================================================
// Roadmap Components
// ============================================================================

/**
 * RoadmapTimeline - Optimization progression timeline
 * @example
 * <RoadmapTimeline
 *   :stages="[
 *     { name: 'Naive', description: 'Baseline', memory: 'O(N²)', status: 'complete' },
 *     { name: 'FlashAttention', description: 'Optimized', memory: 'O(N)', status: 'current' }
 *   ]"
 * />
 */
export interface RoadmapTimelineProps {
  /** Timeline stages (uses defaults if not provided) */
  stages?: Array<{
    name: string
    description: string
    memory: string
    status: 'complete' | 'current' | 'planned'
  }>
}

// ============================================================================
// Data Source Map
// ============================================================================

/**
 * Components and their data sources
 *
 * | Component      | Data Source                    | Props Required |
 * |----------------|--------------------------------|----------------|
 * | KernelCard     | Props only                     | Yes            |
 * | PerfChart      | Props only                     | Yes            |
 * | MemoryCompare  | benchmarks.json (flashAttention) | No           |
 * | GPUSupport     | gpu-support.json               | No             |
 * | CodeDemo       | Props only                     | Yes            |
 * | CodeDiff       | Props only                     | Yes            |
 * | AlgorithmCard  | Props only                     | Yes            |
 * | RoadmapTimeline| Props or defaults              | Optional       |
 */
