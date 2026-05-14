/**
 * Internationalization mappings for navigation and sidebar labels
 */

export type Locale = 'en' | 'zh'

// Navigation labels
const navLabels: Record<Locale, Record<string, string>> = {
  en: {
    home: 'Home',
    setup: 'Quick Start',
    api: 'API',
    architecture: 'Architecture',
    tutorials: 'Tutorials',
    releaseNotes: 'Release Notes',
    github: 'GitHub',
  },
  zh: {
    home: '首页',
    setup: '快速开始',
    api: 'API',
    architecture: '架构',
    tutorials: '教程',
    releaseNotes: '更新日志',
    github: 'GitHub',
  },
}

// Sidebar group titles
const sidebarGroupLabels: Record<Locale, Record<string, string>> = {
  en: {
    gettingStarted: 'Getting Started',
    apiReference: 'API Reference',
    architecture: 'Architecture',
    tutorials: 'Tutorials',
    releaseNotes: 'Release Notes',
  },
  zh: {
    gettingStarted: '开始使用',
    apiReference: 'API 参考',
    architecture: '架构设计',
    tutorials: '教程',
    releaseNotes: '更新日志',
  },
}

// Sidebar item labels
const sidebarItemLabels: Record<Locale, Record<string, string>> = {
  en: {
    quickStart: 'Quick Start',
    apiReference: 'API Reference',
    architectureOverview: 'Architecture Overview',
    flashAttention: 'FlashAttention Deep Dive',
    tensorCoreGemm: 'Tensor Core GEMM Deep Dive',
    performanceGuide: 'Performance Guide',
    troubleshooting: 'Troubleshooting',
    changelog: 'Changelog',
  },
  zh: {
    quickStart: '快速开始',
    apiReference: 'API 参考',
    architectureOverview: '架构概述',
    flashAttention: 'FlashAttention 深度解析',
    tensorCoreGemm: 'Tensor Core GEMM 深度解析',
    performanceGuide: '性能指南',
    troubleshooting: '故障排除',
    changelog: 'Changelog',
  },
}

// Site metadata
export const siteMeta: Record<Locale, { title: string; description: string }> = {
  en: {
    title: 'LLM-Speed',
    description: 'CUDA kernels for LLM inference: FlashAttention, Tensor Core GEMM, and PyTorch bindings',
  },
  zh: {
    title: 'LLM-Speed',
    description: 'LLM推理CUDA内核库：FlashAttention、Tensor Core GEMM和PyTorch绑定',
  },
}

export function getNavLabel(key: string, locale: Locale): string {
  return navLabels[locale][key] || key
}

export function getSidebarGroupLabel(key: string, locale: Locale): string {
  return sidebarGroupLabels[locale][key] || key
}

export function getSidebarItemLabel(key: string, locale: Locale): string {
  return sidebarItemLabels[locale][key] || key
}
