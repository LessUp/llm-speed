<script setup lang="ts">
import { ref, onMounted, watch } from 'vue'
import { Chart, registerables } from 'chart.js'

// Register Chart.js components
Chart.register(...registerables)

interface Props {
  type?: 'bar' | 'line'
  title?: string
  labels: string[]
  datasets: Array<{
    label: string
    data: number[]
    backgroundColor?: string
    borderColor?: string
  }>
}

const props = withDefaults(defineProps<Props>(), {
  type: 'bar',
})

const chartRef = ref<HTMLCanvasElement | null>(null)
const chartInstance = ref<Chart | null>(null)

const createChart = () => {
  if (!chartRef.value) return

  // Destroy existing chart
  if (chartInstance.value) {
    chartInstance.value.destroy()
  }

  const ctx = chartRef.value.getContext('2d')
  if (!ctx) return

  chartInstance.value = new Chart(ctx, {
    type: props.type,
    data: {
      labels: props.labels,
      datasets: props.datasets.map((ds, i) => ({
        ...ds,
        backgroundColor: ds.backgroundColor || (i === 0 ? 'rgba(118, 185, 0, 0.8)' : 'rgba(0, 160, 224, 0.8)'),
        borderColor: ds.borderColor || (i === 0 ? '#76b900' : '#00A0E0'),
        borderWidth: 1,
      })),
    },
    options: {
      responsive: true,
      maintainAspectRatio: true,
      plugins: {
        legend: {
          position: 'top',
          labels: {
            color: 'var(--vp-c-text-1)',
            font: {
              family: 'var(--vp-font-family-base)',
            },
          },
        },
        title: props.title ? {
          display: true,
          text: props.title,
          color: 'var(--vp-c-text-1)',
          font: {
            family: 'var(--vp-font-family-base)',
            size: 16,
            weight: 'bold',
          },
        } : undefined,
      },
      scales: {
        x: {
          ticks: {
            color: 'var(--vp-c-text-2)',
          },
          grid: {
            color: 'var(--vp-c-border)',
          },
        },
        y: {
          ticks: {
            color: 'var(--vp-c-text-2)',
          },
          grid: {
            color: 'var(--vp-c-border)',
          },
        },
      },
    },
  })
}

onMounted(() => {
  createChart()
})

watch(() => [props.labels, props.datasets], () => {
  createChart()
}, { deep: true })
</script>

<template>
  <div class="perf-chart">
    <canvas ref="chartRef"></canvas>
  </div>
</template>

<style scoped>
.perf-chart {
  background: var(--surface-primary);
  border: 1px solid var(--vp-c-border);
  border-radius: var(--radius-lg);
  padding: var(--spacing-lg);
  margin: 1.5rem 0;
}

canvas {
  max-height: 400px;
}
</style>
