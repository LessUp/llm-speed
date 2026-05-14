/**
 * Vite and plugin configuration
 */

import type { UserConfig } from 'vitepress'
import llmstxt from 'vitepress-plugin-llms'

export const viteConfig: UserConfig = {
  plugins: [llmstxt()],
}

export const mermaidConfig = {
  theme: 'dark' as const,
}

export const markdownConfig = {
  lineNumbers: false,
}
