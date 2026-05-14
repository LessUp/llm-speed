<script setup lang="ts">
import { ref } from 'vue'

/**
 * AlgorithmCard Component
 * Displays algorithm details with complexity badges and code snippet
 */
interface Props {
  title: string
  description?: string
  timeComplexity: string
  spaceComplexity: string
  code?: string
  lang?: string
}

const props = withDefaults(defineProps<Props>(), {
  description: '',
  code: '',
  lang: 'cuda',
})

const expanded = ref(false)
</script>

<template>
  <div class="algorithm-card">
    <div class="algorithm-header">
      <h4 class="algorithm-title">{{ title }}</h4>
      <div class="complexity-badges">
        <span class="badge time">
          <span class="badge-label">Time:</span>
          <code>{{ timeComplexity }}</code>
        </span>
        <span class="badge space">
          <span class="badge-label">Space:</span>
          <code>{{ spaceComplexity }}</code>
        </span>
      </div>
    </div>

    <p v-if="description" class="algorithm-description">{{ description }}</p>

    <div v-if="code" class="code-section">
      <button class="expand-btn" @click="expanded = !expanded">
        {{ expanded ? '▼ Hide Code' : '▶ Show Code' }}
      </button>
      <div v-show="expanded" class="code-block">
        <code class="language-{{ lang }}">{{ code }}</code>
      </div>
    </div>
  </div>
</template>

<style scoped>
.algorithm-card {
  border: 1px solid var(--vp-c-border);
  border-radius: 12px;
  padding: 1.25rem;
  margin: 1rem 0;
  background: var(--vp-c-bg-soft);
}

.algorithm-header {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  flex-wrap: wrap;
  gap: 0.75rem;
}

.algorithm-title {
  margin: 0;
  font-size: 1.1rem;
  font-weight: 600;
  color: var(--vp-c-text-1);
}

.complexity-badges {
  display: flex;
  gap: 0.5rem;
  flex-wrap: wrap;
}

.badge {
  display: inline-flex;
  align-items: center;
  gap: 0.25rem;
  padding: 0.25rem 0.5rem;
  border-radius: 6px;
  font-size: 0.85rem;
}

.badge.time {
  background: rgba(118, 185, 0, 0.15);
  color: #76b900;
}

.badge.space {
  background: rgba(0, 160, 224, 0.15);
  color: #00a0e0;
}

.badge-label {
  opacity: 0.8;
}

.badge code {
  font-size: 0.9rem;
}

.algorithm-description {
  margin: 0.75rem 0 0 0;
  color: var(--vp-c-text-2);
  line-height: 1.6;
}

.code-section {
  margin-top: 1rem;
}

.expand-btn {
  padding: 0.35rem 0.75rem;
  border: 1px solid var(--vp-c-border);
  border-radius: 6px;
  background: var(--vp-c-bg);
  color: var(--vp-c-text-2);
  cursor: pointer;
  font-size: 0.85rem;
  transition: all 0.2s;
}

.expand-btn:hover {
  border-color: var(--vp-c-brand-1);
  color: var(--vp-c-brand-1);
}

.code-block {
  margin-top: 0.75rem;
  padding: 1rem;
  border-radius: 8px;
  background: var(--vp-code-block-bg);
  overflow-x: auto;
}

.code-block code {
  font-family: var(--vp-font-family-mono);
  font-size: 0.875rem;
}
</style>
