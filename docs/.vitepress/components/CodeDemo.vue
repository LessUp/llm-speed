<script setup lang="ts">
interface Props {
  filename: string
  code: string
  language?: string
  colabUrl?: string
}

const props = withDefaults(defineProps<Props>(), {
  language: 'python',
})

const copyCode = async () => {
  await navigator.clipboard.writeText(props.code)
}
</script>

<template>
  <div class="code-demo">
    <div class="code-demo-header">
      <span class="code-demo-dot red"></span>
      <span class="code-demo-dot yellow"></span>
      <span class="code-demo-dot green"></span>
      <span class="filename">{{ filename }}</span>
      <div class="actions">
        <button @click="copyCode" class="copy-btn" title="Copy code">
          📋
        </button>
        <a
          v-if="colabUrl"
          :href="colabUrl"
          target="_blank"
          rel="noopener"
          class="colab-link"
        >
          <img
            src="https://colab.research.google.com/assets/colab-badge.svg"
            alt="Open in Colab"
          />
        </a>
      </div>
    </div>
    <div class="code-demo-content">
      <pre><code :class="`language-${language}`">{{ code }}</code></pre>
    </div>
  </div>
</template>

<style scoped>
.code-demo {
  background: var(--surface-primary);
  border: 1px solid var(--vp-c-border);
  border-radius: var(--radius-lg);
  overflow: hidden;
  margin: 1rem 0;
}

.code-demo-header {
  background: var(--surface-secondary);
  padding: var(--spacing-sm) var(--spacing-md);
  display: flex;
  align-items: center;
  gap: var(--spacing-xs);
  border-bottom: 1px solid var(--vp-c-border);
}

.code-demo-dot {
  width: 12px;
  height: 12px;
  border-radius: 50%;
}

.code-demo-dot.red { background: var(--perf-critical); }
.code-demo-dot.yellow { background: var(--perf-good); }
.code-demo-dot.green { background: var(--perf-excellent); }

.filename {
  flex: 1;
  margin-left: var(--spacing-sm);
  color: var(--vp-c-text-2);
  font-size: 0.875rem;
  font-family: var(--vp-font-family-mono);
}

.actions {
  display: flex;
  align-items: center;
  gap: var(--spacing-sm);
}

.copy-btn {
  background: transparent;
  border: none;
  cursor: pointer;
  padding: 4px;
  opacity: 0.7;
  transition: opacity var(--transition-fast);
}

.copy-btn:hover {
  opacity: 1;
}

.colab-link img {
  height: 20px;
}

.code-demo-content {
  padding: 0;
  overflow-x: auto;
}

.code-demo-content pre {
  margin: 0;
  padding: var(--spacing-md);
  background: transparent;
}

.code-demo-content code {
  font-size: 0.875rem;
  line-height: 1.6;
  font-family: var(--vp-font-family-mono);
}
</style>
