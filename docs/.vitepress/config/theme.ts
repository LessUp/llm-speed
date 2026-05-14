/**
 * VitePress theme configuration
 */

import type { DefaultTheme } from 'vitepress'
import { siteConfig } from './site'

export const themeConfig: DefaultTheme.Config = {
  logo: '/assets/images/logo.svg',
  siteTitle: siteConfig.title,
  outline: [2, 3],
  search: {
    provider: 'local',
  },
  socialLinks: [
    { icon: 'github', link: siteConfig.github },
  ],
  footer: {
    message: 'Released under the MIT License.',
    copyright: 'Copyright © 2024-present LessUp',
  },
}
