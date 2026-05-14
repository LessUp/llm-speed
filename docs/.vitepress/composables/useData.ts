/**
 * Composables for loading JSON data files
 * These provide a clean interface for components to access data
 */

import gpuSupportData from '../data/gpu-support.json'
import benchmarksData from '../data/benchmarks.json'

// Types derived from JSON structure
export interface GPUArchitecture {
  name: string
  computeCapability: string
  examples: string[]
  tensorCore: {
    fp16: boolean
    bf16: boolean
    tf32: boolean
    int8: boolean
    fp8: boolean
  }
  status: 'primary' | 'supported' | 'limited'
  notes: string
}

export interface FlashAttentionMetric {
  seqLength: number
  standardMemoryMB: number
  flashMemoryMB: number
  speedup: number
}

/**
 * Load GPU support data
 */
export function useGpuSupport() {
  return {
    architectures: gpuSupportData.architectures as GPUArchitecture[],
    precisionSupport: gpuSupportData.precisionSupport,
  }
}

/**
 * Load FlashAttention memory comparison data
 */
export function useFlashAttentionMemory() {
  const metrics = benchmarksData.flashAttention.metrics as FlashAttentionMetric[]

  return {
    metrics,
    description: benchmarksData.flashAttention.description,
    notes: benchmarksData.flashAttention.notes,
    // Derived data for table display
    tableData: metrics.map(m => ({
      seqLength: m.seqLength,
      standard: `${m.standardMemoryMB} MB`,
      flash: `${m.flashMemoryMB} MB`,
      savings: `${m.speedup}×`,
    })),
  }
}

/**
 * Load Tensor Core GEMM performance data
 */
export function useTensorCoreGemm() {
  return {
    metrics: benchmarksData.tensorCoreGemm.metrics,
    description: benchmarksData.tensorCoreGemm.description,
    notes: benchmarksData.tensorCoreGemm.notes,
  }
}

/**
 * Load kernel comparison data
 */
export function useKernelComparison() {
  return {
    kernels: benchmarksData.comparison.kernels,
  }
}
