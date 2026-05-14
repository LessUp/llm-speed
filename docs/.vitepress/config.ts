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

  // Sitemap configuration
  sitemap: {
    hostname: 'https://lessup.github.io/llm-speed/'
  },

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

  // Internationalization: English and Chinese (kimi-cli pattern)
  locales: {
    root: {
      label: 'English',
      lang: 'en-US',
      link: '/en/',
      title: 'LLM-Speed',
      description: 'CUDA kernels for LLM inference: FlashAttention, Tensor Core GEMM, and PyTorch bindings',
      themeConfig: {
        nav: [
          { text: 'Home', link: '/en/' },
          { text: 'Quick Start', link: '/en/setup/', activeMatch: '/en/setup/' },
          { text: 'API', link: '/en/api/', activeMatch: '/en/api/' },
          { text: 'Architecture', link: '/en/architecture/', activeMatch: '/en/architecture/' },
          { text: 'Tutorials', link: '/en/tutorials/performance/', activeMatch: '/en/tutorials/' },
          { text: 'Release Notes', link: '/en/release-notes/changelog/', activeMatch: '/en/release-notes/' },
          { text: 'GitHub', link: 'https://github.com/LessUp/llm-speed' },
        ],
        sidebar: {
          '/en/setup/': [
            {
              text: 'Getting Started',
              items: [
                { text: 'Quick Start', link: '/en/setup/' },
              ],
            },
          ],
          '/en/api/': [
            {
              text: 'API Reference',
              items: [
                { text: 'API Reference', link: '/en/api/' },
              ],
            },
          ],
          '/en/architecture/': [
            {
              text: 'Architecture',
              items: [
                { text: 'Architecture Overview', link: '/en/architecture/' },
                { text: 'FlashAttention Deep Dive', link: '/en/architecture/flash-attention' },
                { text: 'Tensor Core GEMM Deep Dive', link: '/en/architecture/tensor-core-gemm' },
              ],
            },
          ],
          '/en/tutorials/': [
            {
              text: 'Tutorials',
              items: [
                { text: 'Performance Guide', link: '/en/tutorials/performance' },
                { text: 'Troubleshooting', link: '/en/tutorials/troubleshooting' },
              ],
            },
          ],
          '/en/release-notes/': [
            {
              text: 'Release Notes',
              items: [
                { text: 'Changelog', link: '/en/release-notes/changelog' },
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
          { text: '快速开始', link: '/zh/setup/', activeMatch: '/zh/setup/' },
          { text: 'API', link: '/zh/api/', activeMatch: '/zh/api/' },
          { text: '架构', link: '/zh/architecture/', activeMatch: '/zh/architecture/' },
          { text: '教程', link: '/zh/tutorials/performance/', activeMatch: '/zh/tutorials/' },
          { text: '更新日志', link: '/zh/release-notes/changelog/', activeMatch: '/zh/release-notes/' },
          { text: 'GitHub', link: 'https://github.com/LessUp/llm-speed' },
        ],
        sidebar: {
          '/zh/setup/': [
            {
              text: '开始使用',
              items: [
                { text: '快速开始', link: '/zh/setup/' },
              ],
            },
          ],
          '/zh/api/': [
            {
              text: 'API 参考',
              items: [
                { text: 'API 参考', link: '/zh/api/' },
              ],
            },
          ],
          '/zh/architecture/': [
            {
              text: '架构设计',
              items: [
                { text: '架构概述', link: '/zh/architecture/' },
                { text: 'FlashAttention 深度解析', link: '/zh/architecture/flash-attention' },
                { text: 'Tensor Core GEMM 深度解析', link: '/zh/architecture/tensor-core-gemm' },
              ],
            },
          ],
          '/zh/tutorials/': [
            {
              text: '教程',
              items: [
                { text: '性能指南', link: '/zh/tutorials/performance' },
                { text: '故障排除', link: '/zh/tutorials/troubleshooting' },
              ],
            },
          ],
          '/zh/release-notes/': [
            {
              text: '更新日志',
              items: [
                { text: 'Changelog', link: '/zh/release-notes/changelog' },
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
