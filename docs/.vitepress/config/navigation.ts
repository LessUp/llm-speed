/**
 * Navigation structure definitions
 *
 * Single source of truth for nav/sidebar structure.
 * Translations are provided via i18n mappings.
 */

export interface NavItem {
  key: string
  link: string
  match?: string
  external?: boolean
}

export interface SidebarGroup {
  key: string
  items: NavItem[]
}

// Navigation structure - language-agnostic
export const navItems: NavItem[] = [
  { key: 'home', link: '/' },
  { key: 'setup', link: '/setup/', match: '/setup/' },
  { key: 'api', link: '/api/', match: '/api/' },
  { key: 'architecture', link: '/architecture/', match: '/architecture/' },
  { key: 'tutorials', link: '/tutorials/performance/', match: '/tutorials/' },
  { key: 'releaseNotes', link: '/release-notes/changelog/', match: '/release-notes/' },
  { key: 'github', link: 'https://github.com/LessUp/llm-speed', external: true },
]

// Sidebar structure - language-agnostic
export const sidebarGroups: Record<string, SidebarGroup[]> = {
  '/setup/': [
    {
      key: 'gettingStarted',
      items: [{ key: 'quickStart', link: '/setup/' }],
    },
  ],
  '/api/': [
    {
      key: 'apiReference',
      items: [{ key: 'apiReference', link: '/api/' }],
    },
  ],
  '/architecture/': [
    {
      key: 'architecture',
      items: [
        { key: 'architectureOverview', link: '/architecture/' },
        { key: 'flashAttention', link: '/architecture/flash-attention' },
        { key: 'tensorCoreGemm', link: '/architecture/tensor-core-gemm' },
      ],
    },
  ],
  '/tutorials/': [
    {
      key: 'tutorials',
      items: [
        { key: 'performanceGuide', link: '/tutorials/performance' },
        { key: 'troubleshooting', link: '/tutorials/troubleshooting' },
      ],
    },
  ],
  '/release-notes/': [
    {
      key: 'releaseNotes',
      items: [{ key: 'changelog', link: '/release-notes/changelog' }],
    },
  ],
}
