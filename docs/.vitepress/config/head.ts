/**
 * SEO and head metadata configuration
 */

import type { HeadConfig } from 'vitepress'

export const headTags: HeadConfig[] = [
  ['link', { rel: 'icon', type: 'image/svg+xml', href: '/assets/images/favicon.svg' }],
  ['meta', { property: 'og:type', content: 'website' }],
  ['meta', { property: 'og:title', content: 'LLM-Speed | CUDA Kernels for LLM Inference' }],
  ['meta', { property: 'og:description', content: 'A focused CUDA kernel library implementing FlashAttention forward, Tensor Core GEMM acceleration, and seamless PyTorch integration.' }],
  ['meta', { property: 'og:image', content: '/assets/images/og-image.svg' }],
  ['meta', { property: 'og:url', content: 'https://lessup.github.io/llm-speed/' }],
  ['meta', { name: 'twitter:card', content: 'summary_large_image' }],
  ['meta', { name: 'twitter:title', content: 'LLM-Speed | CUDA Kernels for LLM Inference' }],
  ['meta', { name: 'twitter:description', content: 'A focused CUDA kernel library implementing FlashAttention forward, Tensor Core GEMM acceleration.' }],
  ['meta', { name: 'twitter:image', content: '/assets/images/og-image.svg' }],
  ['meta', { name: 'keywords', content: 'CUDA, LLM, FlashAttention, Tensor Core, GEMM, GPU, Deep Learning, Inference' }],
]
