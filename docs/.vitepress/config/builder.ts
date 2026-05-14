/**
 * Build VitePress configuration from modular definitions
 *
 * This module exports all configuration utilities and re-exports
 * from other config modules for convenient importing.
 */

import type { DefaultTheme } from 'vitepress'
import { navItems, sidebarGroups, type NavItem, type SidebarGroup } from './navigation'
import { getNavLabel, getSidebarGroupLabel, getSidebarItemLabel, siteMeta, type Locale } from './i18n'

// Re-export all config modules
export { headTags } from './head'
export { siteConfig, resolveBase } from './site'
export { themeConfig } from './theme'
export { viteConfig, mermaidConfig, markdownConfig } from './plugins'
export type { Locale, NavItem, SidebarGroup }

/**
 * Build navigation array for a locale
 */
export function buildNav(locale: Locale, baseLink: string): DefaultTheme.NavItem[] {
  return navItems.map((item) => {
    const label = getNavLabel(item.key, locale)

    if (item.external) {
      return { text: label, link: item.link }
    }

    return {
      text: label,
      link: `${baseLink}${item.link}`,
      ...(item.match && { activeMatch: `${baseLink}${item.match}` }),
    }
  })
}

/**
 * Build sidebar configuration for a locale
 */
export function buildSidebar(locale: Locale, baseLink: string): DefaultTheme.Sidebar {
  const result: DefaultTheme.Sidebar = {}

  for (const [path, groups] of Object.entries(sidebarGroups)) {
    result[`${baseLink}${path}`] = groups.map((group) => ({
      text: getSidebarGroupLabel(group.key, locale),
      items: group.items.map((item) => ({
        text: getSidebarItemLabel(item.key, locale),
        link: `${baseLink}${item.link}`,
      })),
    }))
  }

  return result
}

/**
 * Build locale theme config
 */
export function buildLocaleThemeConfig(locale: Locale, baseLink: string): DefaultTheme.Config {
  return {
    nav: buildNav(locale, baseLink),
    sidebar: buildSidebar(locale, baseLink),
  }
}

/**
 * Build complete locale configuration
 */
export function buildLocaleConfig(locale: Locale, baseLink: string) {
  const meta = siteMeta[locale]
  const langMap: Record<Locale, string> = {
    en: 'en-US',
    zh: 'zh-CN',
  }
  const labelMap: Record<Locale, string> = {
    en: 'English',
    zh: '简体中文',
  }

  return {
    label: labelMap[locale],
    lang: langMap[locale],
    link: baseLink,
    title: meta.title,
    description: meta.description,
    themeConfig: buildLocaleThemeConfig(locale, baseLink),
  }
}
