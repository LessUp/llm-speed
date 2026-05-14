/**
 * Site metadata configuration
 */

export const siteConfig = {
  title: 'LLM-Speed',
  description: 'CUDA kernels for LLM inference: FlashAttention, Tensor Core GEMM, and PyTorch bindings',
  github: 'https://github.com/LessUp/llm-speed',
  githubPages: 'https://lessup.github.io/llm-speed/',
}

/**
 * Resolve dynamic base path for GitHub Pages deployment
 */
export function resolveBase(): string {
  const rawBase = process.env.VITEPRESS_BASE
  if (!rawBase) return '/llm-speed/'

  if (rawBase.startsWith('/')) {
    return rawBase.endsWith('/') ? rawBase : `${rawBase}/`
  }
  return `/${rawBase}/`
}
