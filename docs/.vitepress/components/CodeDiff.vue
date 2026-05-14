<script setup lang="ts">
import { ref } from 'vue'

/**
 * CodeDiff Component
 * Side-by-side code comparison for optimization visualization
 */
interface Props {
  title?: string
  leftLabel: string
  rightLabel: string
  leftCode: string
  rightCode: string
  lang?: string
  highlight?: 'left' | 'right' | 'both'
}

const props = withDefaults(defineProps<Props>(), {
  title: '',
  lang: 'cuda',
  highlight: 'right',
})

const showDiff = ref(true)
</script>

<template>
  <div class="code-diff">
    <div v-if="title" class="diff-title">{{ title }}</div>
    <div class="diff-headers">
      <div class="diff-header" :class="{ dimmed: highlight === 'right' }">
        <span class="diff-indicator old">−</span>
        {{ leftLabel }}
      </div>
      <div class="diff-header" :class="{ dimmed: highlight === 'left' }">
        <span class="diff-indicator new">+</span>
        {{ rightLabel }}
      </div>
    </div>
    <div class="diff-content">
      <div class="diff-pane" :class="{ dimmed: highlight === 'right' }">
        <pre><code>{{ leftCode }}</code></pre>
      </div>
      <div class="diff-pane" :class="{ dimmed: highlight === 'left' }">
        <pre><code>{{ rightCode }}</code></pre>
      </div>
    </div>
  </div>
</template>

<style scoped>
.code-diff {
  border: 1px solid var(--vp-c-border);
  border-radius: 12px;
  overflow: hidden;
  margin: 1.5rem 0;
}

.diff-title {
  padding: 0.75rem 1rem;
  font-weight: 600;
  font-size: 0.95rem;
  border-bottom: 1px solid var(--vp-c-border);
  background: var(--vp-c-bg-soft);
}

.diff-headers {
  display: grid;
  grid-template-columns: 1fr 1fr;
  border-bottom: 1px solid var(--vp-c-border);
}

.diff-header {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  padding: 0.5rem 1rem;
  font-size: 0.85rem;
  font-weight: 500;
  color: var(--vp-c-text-2);
}

.diff-header.dimmed {
  opacity: 0.6;
}

.diff-indicator {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 20px;
  height: 20px;
  border-radius: 4px;
  font-weight: 700;
  font-size: 0.9rem;
}

.diff-indicator.old {
  background: rgba(248, 81, 73, 0.15);
  color: #f85149;
}

.diff-indicator.new {
  background: rgba(63, 185, 80, 0.15);
  color: #3fb950;
}

.diff-content {
  display: grid;
  grid-template-columns: 1fr 1fr;
}

.diff-pane {
  padding: 1rem;
  background: var(--vp-code-block-bg);
  overflow-x: auto;
}

.diff-pane.dimmed {
  opacity: 0.7;
}

.diff-pane:first-child {
  border-right: 1px solid var(--vp-c-border);
}

.diff-pane pre {
  margin: 0;
  padding: 0;
}

.diff-pane code {
  font-family: var(--vp-font-family-mono);
  font-size: 0.85rem;
  line-height: 1.6;
  white-space: pre;
}

@media (max-width: 768px) {
  .diff-content {
    grid-template-columns: 1fr;
  }

  .diff-pane:first-child {
    border-right: none;
    border-bottom: 1px solid var(--vp-c-border);
  }

  .diff-headers {
    grid-template-columns: 1fr;
  }

  .diff-header:first-child {
    border-bottom: 1px solid var(--vp-c-border);
  }
}
</style>
