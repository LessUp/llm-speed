<script setup lang="ts">
interface Props {
  title: string
  icon: string
  description: string
  memory?: string
  speedup?: number
  features?: string[]
}

const props = defineProps<Props>()
</script>

<template>
  <div class="kernel-card">
    <div class="icon">{{ icon }}</div>
    <h3 class="title">{{ title }}</h3>
    <p class="description">{{ description }}</p>

    <div v-if="memory || speedup" class="metrics">
      <div v-if="memory" class="metric">
        <span class="metric-label">Memory</span>
        <span class="metric-value memory-value">{{ memory }}</span>
      </div>
      <div v-if="speedup" class="metric">
        <span class="metric-label">Speedup</span>
        <span class="metric-value speedup-value">{{ speedup }}×</span>
      </div>
    </div>

    <ul v-if="features && features.length" class="features-list">
      <li v-for="feature in features" :key="feature">{{ feature }}</li>
    </ul>
  </div>
</template>

<style scoped>
.kernel-card {
  background: var(--surface-primary);
  border: 1px solid var(--vp-c-border);
  border-radius: var(--radius-lg);
  padding: var(--spacing-lg);
  transition: all var(--transition-normal);
}

.kernel-card:hover {
  border-color: var(--vp-c-brand-1);
  box-shadow: var(--shadow-glow);
  transform: translateY(-2px);
}

.icon {
  font-size: 2.5rem;
  margin-bottom: var(--spacing-sm);
}

.title {
  font-size: 1.25rem;
  font-weight: 600;
  color: var(--vp-c-text-1);
  margin-bottom: var(--spacing-xs);
}

.description {
  color: var(--vp-c-text-2);
  line-height: 1.5;
  font-size: 0.9rem;
}

.metrics {
  display: flex;
  gap: var(--spacing-md);
  margin-top: var(--spacing-md);
  padding-top: var(--spacing-md);
  border-top: 1px solid var(--vp-c-divider);
}

.metric {
  display: flex;
  flex-direction: column;
  gap: 2px;
}

.metric-label {
  font-size: 0.75rem;
  color: var(--vp-c-text-3);
  text-transform: uppercase;
  letter-spacing: 0.5px;
}

.metric-value {
  font-size: 1.1rem;
  font-weight: 600;
}

.memory-value {
  color: var(--cuda-accent);
}

.speedup-value {
  color: var(--vp-c-brand-1);
}

.features-list {
  list-style: none;
  padding: 0;
  margin: var(--spacing-md) 0 0 0;
  display: flex;
  flex-wrap: wrap;
  gap: var(--spacing-xs);
}

.features-list li {
  font-size: 0.75rem;
  padding: 4px 8px;
  background: var(--tag-bg);
  color: var(--tag-text);
  border-radius: var(--radius-sm);
}
</style>
