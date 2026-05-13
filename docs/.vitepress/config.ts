import { defineConfig } from 'vitepress'
import { withMermaid } from 'vitepress-plugin-mermaid'
import llmstxt from 'vitepress-plugin-llms'

// Dynamic base path for GitHub Pages deployment
const rawBase = process.env.VITEPRESS_BASE
const base = rawBase
  ? rawBase.startsWith('/')
    ? rawBase.endsWith('/') ? rawBase : `${rawBase}/`
    : `/${rawBase}/`
  : '/llm-speed/'

export default withMermaid(defineConfig({
  base,
  title: 'LLM-Speed',
  description: 'CUDA kernels for LLM inference: FlashAttention, Tensor Core GEMM, and PyTorch bindings',

  // Ignore dead links during migration
  ignoreDeadLinks: true,

  // Head elements for SEO and branding
  head: [
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
  ],

  // Internationalization: Chinese and English
  locales: {
    root: {
      label: 'English',
      lang: 'en-US',
      title: 'LLM-Speed',
      description: 'CUDA kernels for LLM inference: FlashAttention, Tensor Core GEMM, and PyTorch bindings',
      themeConfig: {
        nav: [
          { text: 'Home', link: '/' },
          { text: 'Quick Start', link: '/docs/setup/quickstart-en/', activeMatch: '/docs/setup/' },
          { text: 'API', link: '/docs/api/api-en/', activeMatch: '/docs/api/' },
          { text: 'Architecture', link: '/docs/architecture/architecture-en/', activeMatch: '/docs/architecture/' },
          { text: 'Performance', link: '/docs/tutorials/performance-en/', activeMatch: '/docs/tutorials/' },
          { text: 'GitHub', link: 'https://github.com/LessUp/llm-speed' },
        ],
        sidebar: {
          '/docs/setup/': [
            {
              text: 'Getting Started',
              items: [
                { text: 'Quick Start', link: '/docs/setup/quickstart-en/' },
              ],
            },
          ],
          '/docs/api/': [
            {
              text: 'API Reference',
              items: [
                { text: 'API Reference', link: '/docs/api/api-en/' },
              ],
            },
          ],
          '/docs/architecture/': [
            {
              text: 'Architecture',
              items: [
                { text: 'Architecture Overview', link: '/docs/architecture/architecture-en/' },
              ],
            },
          ],
          '/docs/tutorials/': [
            {
              text: 'Tutorials',
              items: [
                { text: 'Performance Guide', link: '/docs/tutorials/performance-en/' },
                { text: 'Troubleshooting', link: '/docs/tutorials/troubleshooting-en/' },
              ],
            },
          ],
        },
      },
    },
    zh: {
      label: '简体中文',
      lang: 'zh-CN',
      link: '/zh/',
      title: 'LLM-Speed',
      description: 'LLM推理CUDA内核库：FlashAttention、Tensor Core GEMM和PyTorch绑定',
      themeConfig: {
        nav: [
          { text: '首页', link: '/zh/' },
          { text: '快速开始', link: '/zh/docs/setup/quickstart-zh/', activeMatch: '/zh/docs/setup/' },
          { text: 'API', link: '/zh/docs/api/api-zh/', activeMatch: '/zh/docs/api/' },
          { text: '架构', link: '/zh/docs/architecture/architecture-zh/', activeMatch: '/zh/docs/architecture/' },
          { text: '性能', link: '/zh/docs/tutorials/performance-zh/', activeMatch: '/zh/docs/tutorials/' },
          { text: 'GitHub', link: 'https://github.com/LessUp/llm-speed' },
        ],
        sidebar: {
          '/zh/docs/setup/': [
            {
              text: '开始使用',
              items: [
                { text: '快速开始', link: '/zh/docs/setup/quickstart-zh/' },
              ],
            },
          ],
          '/zh/docs/api/': [
            {
              text: 'API 参考',
              items: [
                { text: 'API 参考', link: '/zh/docs/api/api-zh/' },
              ],
            },
          ],
          '/zh/docs/architecture/': [
            {
              text: '架构设计',
              items: [
                { text: '架构概述', link: '/zh/docs/architecture/architecture-zh/' },
              ],
            },
          ],
          '/zh/docs/tutorials/': [
            {
              text: '教程',
              items: [
                { text: '性能指南', link: '/zh/docs/tutorials/performance-zh/' },
                { text: '故障排除', link: '/zh/docs/tutorials/troubleshooting-zh/' },
              ],
            },
          ],
        },
      },
    },
  },

  // Global theme configuration
  themeConfig: {
    logo: '/assets/images/logo.svg',
    siteTitle: 'LLM-Speed',
    outline: [2, 3],
    search: {
      provider: 'local',
    },
    socialLinks: [
      { icon: 'github', link: 'https://github.com/LessUp/llm-speed' },
    ],
    footer: {
      message: 'Released under the MIT License.',
      copyright: 'Copyright © 2024-present LessUp',
    },
  },

  // Mermaid configuration
  mermaid: {
    theme: 'dark',
  },

  // Vite plugins
  vite: {
    plugins: [llmstxt()],
  },

  // Markdown configuration
  markdown: {
    lineNumbers: false,
  },
}))
