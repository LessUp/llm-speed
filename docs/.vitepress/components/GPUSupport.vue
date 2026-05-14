<script setup lang="ts">
import { useGpuSupport, type GPUArchitecture } from '../composables/useData'

const { architectures } = useGpuSupport()

const statusLabels: Record<string, string> = {
  primary: '✅ Primary target',
  supported: '✅ Supported',
  limited: '⚠️ Limited',
}

const statusClasses: Record<string, string> = {
  primary: 'status-primary',
  supported: 'status-supported',
  limited: 'status-limited',
}

function formatExamples(gpu: GPUArchitecture): string {
  return gpu.examples.join(', ')
}

function formatTensorCore(gpu: GPUArchitecture): string {
  const types: string[] = []
  if (gpu.tensorCore.fp16) types.push('FP16')
  if (gpu.tensorCore.bf16) types.push('BF16')
  if (gpu.tensorCore.tf32) types.push('TF32')
  if (gpu.tensorCore.int8) types.push('INT8')
  if (gpu.tensorCore.fp8) types.push('FP8')
  return types.join(', ')
}
</script>

<template>
  <div class="gpu-support">
    <table class="gpu-table">
      <thead>
        <tr>
          <th>Architecture</th>
          <th>Examples</th>
          <th>Tensor Core</th>
          <th>Status</th>
        </tr>
      </thead>
      <tbody>
        <tr v-for="gpu in architectures" :key="gpu.name">
          <td><strong>{{ gpu.name }}</strong></td>
          <td>{{ formatExamples(gpu) }}</td>
          <td>{{ formatTensorCore(gpu) }}</td>
          <td :class="statusClasses[gpu.status]">
            {{ statusLabels[gpu.status] }}
          </td>
        </tr>
      </tbody>
    </table>
  </div>
</template>

<style scoped>
.gpu-support {
  margin: 1.5rem 0;
}

.gpu-table {
  width: 100%;
  border-collapse: collapse;
  border-radius: var(--radius-md);
  overflow: hidden;
  border: 1px solid var(--vp-c-border);
}

.gpu-table th,
.gpu-table td {
  padding: var(--spacing-sm) var(--spacing-md);
  text-align: left;
  border-bottom: 1px solid var(--vp-c-divider);
}

.gpu-table th {
  background: var(--surface-primary);
  font-weight: 600;
  color: var(--vp-c-text-1);
}

.gpu-table tr:last-child td {
  border-bottom: none;
}

.gpu-table tr:hover td {
  background: var(--surface-secondary);
}

.status-primary {
  color: var(--perf-excellent);
  font-weight: 500;
}

.status-supported {
  color: var(--perf-good);
  font-weight: 500;
}

.status-limited {
  color: var(--perf-warning);
  font-weight: 500;
}
</style>
