<script setup lang="ts">
interface GPU {
  architecture: string
  examples: string
  tensorCore: string
  status: 'primary' | 'supported' | 'limited'
}

const gpus: GPU[] = [
  {
    architecture: 'Ampere',
    examples: 'A100, RTX 30/40',
    tensorCore: 'FP16, BF16, TF32',
    status: 'primary',
  },
  {
    architecture: 'Hopper',
    examples: 'H100',
    tensorCore: 'FP16, BF16, FP8',
    status: 'supported',
  },
  {
    architecture: 'Volta',
    examples: 'V100',
    tensorCore: 'FP16',
    status: 'limited',
  },
  {
    architecture: 'Turing',
    examples: 'T4, RTX 20',
    tensorCore: 'FP16, INT8',
    status: 'limited',
  },
]

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
        <tr v-for="gpu in gpus" :key="gpu.architecture">
          <td><strong>{{ gpu.architecture }}</strong></td>
          <td>{{ gpu.examples }}</td>
          <td>{{ gpu.tensorCore }}</td>
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
