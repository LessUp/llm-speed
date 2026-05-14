<script setup lang="ts">
import { useFlashAttentionMemory } from '../composables/useData'

const { tableData, notes } = useFlashAttentionMemory()
</script>

<template>
  <div class="memory-compare">
    <table class="memory-table">
      <thead>
        <tr>
          <th>Sequence Length</th>
          <th>Standard Attention</th>
          <th>FlashAttention</th>
          <th>Memory Savings</th>
        </tr>
      </thead>
      <tbody>
        <tr v-for="row in tableData" :key="row.seqLength">
          <td><strong>{{ row.seqLength }}</strong></td>
          <td class="standard-col">{{ row.standard }}</td>
          <td class="flash-col">{{ row.flash }}</td>
          <td class="savings-col">{{ row.savings }}</td>
        </tr>
      </tbody>
    </table>

    <div class="memory-bars">
      <div v-for="row in tableData" :key="row.seqLength" class="bar-row">
        <span class="bar-label">Seq {{ row.seqLength }}</span>
        <div class="bar-container">
          <div class="bar standard" :style="{ width: '100%' }">
            <span class="bar-value">{{ row.standard }}</span>
          </div>
        </div>
        <div class="bar-container">
          <div class="bar flash" :style="{ width: (100 / parseInt(row.savings)) + '%' }">
            <span class="bar-value">{{ row.flash }}</span>
          </div>
        </div>
        <span class="bar-savings">{{ row.savings }}</span>
      </div>
    </div>

    <p class="footnote">
      {{ notes }}
    </p>
  </div>
</template>

<style scoped>
.memory-compare {
  margin: 1.5rem 0;
}

.memory-table {
  width: 100%;
  border-collapse: collapse;
  border-radius: var(--radius-md);
  overflow: hidden;
  border: 1px solid var(--vp-c-border);
  margin-bottom: var(--spacing-lg);
}

.memory-table th,
.memory-table td {
  padding: var(--spacing-sm) var(--spacing-md);
  text-align: left;
  border-bottom: 1px solid var(--vp-c-divider);
}

.memory-table th {
  background: var(--surface-primary);
  font-weight: 600;
  color: var(--vp-c-text-1);
}

.standard-col {
  color: var(--perf-warning);
}

.flash-col {
  color: var(--vp-c-brand-1);
  font-weight: 500;
}

.savings-col {
  color: var(--perf-excellent);
  font-weight: 600;
}

.memory-bars {
  display: flex;
  flex-direction: column;
  gap: var(--spacing-md);
  padding: var(--spacing-md);
  background: var(--surface-primary);
  border-radius: var(--radius-md);
  border: 1px solid var(--vp-c-border);
}

.bar-row {
  display: grid;
  grid-template-columns: 80px 1fr 1fr 50px;
  align-items: center;
  gap: var(--spacing-sm);
}

.bar-label {
  font-size: 0.875rem;
  color: var(--vp-c-text-2);
  font-family: var(--vp-font-family-mono);
}

.bar-container {
  height: 20px;
  background: var(--surface-tertiary);
  border-radius: var(--radius-sm);
  overflow: hidden;
}

.bar {
  height: 100%;
  display: flex;
  align-items: center;
  padding: 0 var(--spacing-xs);
  transition: width var(--transition-slow);
}

.bar.standard {
  background: linear-gradient(90deg, var(--perf-warning), var(--perf-critical));
}

.bar.flash {
  background: linear-gradient(90deg, var(--vp-c-brand-1), var(--vp-c-brand-3));
}

.bar-value {
  font-size: 0.7rem;
  color: #000;
  font-weight: 600;
  white-space: nowrap;
}

.bar-savings {
  font-size: 0.875rem;
  font-weight: 600;
  color: var(--perf-excellent);
  text-align: right;
}

.footnote {
  margin-top: var(--spacing-md);
  font-size: 0.8rem;
  color: var(--vp-c-text-3);
  font-style: italic;
}

@media (max-width: 768px) {
  .bar-row {
    grid-template-columns: 1fr;
    gap: var(--spacing-xs);
  }

  .bar-label,
  .bar-savings {
    text-align: left;
  }
}
</style>
