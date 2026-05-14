---
layout: home
---

<script setup>
import { onMounted } from 'vue'
onMounted(() => {
  // Auto-redirect based on browser language
  if (navigator.language.startsWith('zh')) {
    window.location.href = '/llm-speed/zh/'
  } else {
    window.location.href = '/llm-speed/en/'
  }
})
</script>

# LLM-Speed

## Select Language / 选择语言

<div class="language-selector">
  <a href="/llm-speed/en/" class="lang-btn">
    <span class="lang-icon">🇬🇧</span>
    <span class="lang-text">
      <strong>English</strong>
      <small>Documentation in English</small>
    </span>
  </a>
  <a href="/llm-speed/zh/" class="lang-btn">
    <span class="lang-icon">🇨🇳</span>
    <span class="lang-text">
      <strong>简体中文</strong>
      <small>中文文档</small>
    </span>
  </a>
</div>

<style>
.language-selector {
  display: flex;
  gap: 1.5rem;
  justify-content: center;
  margin: 3rem 0;
  flex-wrap: wrap;
}

.lang-btn {
  display: flex;
  align-items: center;
  gap: 1rem;
  padding: 1.5rem 2rem;
  border: 1px solid var(--vp-c-border);
  border-radius: 12px;
  text-decoration: none;
  transition: all 0.2s ease;
  min-width: 200px;
}

.lang-btn:hover {
  border-color: var(--vp-c-brand-1);
  transform: translateY(-2px);
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
}

.lang-icon {
  font-size: 2rem;
}

.lang-text {
  display: flex;
  flex-direction: column;
  gap: 0.25rem;
}

.lang-text strong {
  font-size: 1.1rem;
  color: var(--vp-c-text-1);
}

.lang-text small {
  font-size: 0.85rem;
  color: var(--vp-c-text-2);
}

/* Dark mode adjustments */
html:not(.dark) .lang-btn {
  background: var(--vp-c-bg-soft);
}

html.dark .lang-btn:hover {
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
}
</style>
