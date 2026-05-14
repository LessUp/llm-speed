import { defineConfig } from 'vitepress'
import { withMermaid } from 'vitepress-plugin-mermaid'
import {
  buildLocaleConfig,
  headTags,
  resolveBase,
  siteConfig,
  themeConfig,
  viteConfig,
  mermaidConfig,
  markdownConfig,
} from './config/builder'

export default withMermaid(defineConfig({
  base: resolveBase(),
  title: siteConfig.title,
  description: siteConfig.description,

  // Sitemap configuration
  sitemap: {
    hostname: siteConfig.githubPages,
  },

  // Head elements for SEO and branding
  head: headTags,

  // Internationalization: English and Chinese
  locales: {
    root: buildLocaleConfig('en', '/en/'),
    zh: buildLocaleConfig('zh', '/zh/'),
  },

  // Global theme configuration
  themeConfig,

  // Mermaid configuration
  mermaid: mermaidConfig,

  // Vite plugins
  vite: viteConfig,

  // Markdown configuration
  markdown: markdownConfig,
}))
