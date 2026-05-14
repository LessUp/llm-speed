/**
 * Sync CHANGELOG.md from root to docs release notes page.
 * Transforms Keep a Changelog format for VitePress display.
 *
 * Usage: node scripts/sync-changelog.mjs
 */

import { readFileSync, writeFileSync, existsSync, mkdirSync } from 'node:fs'
import { resolve, dirname } from 'node:path'
import { fileURLToPath } from 'node:url'

const __dirname = dirname(fileURLToPath(import.meta.url))
const rootDir = resolve(__dirname, '../..')
const docsDir = resolve(__dirname, '..')

const sourcePath = resolve(rootDir, 'CHANGELOG.md')
const targetEnPath = resolve(docsDir, 'en/release-notes/changelog.md')
const targetZhPath = resolve(docsDir, 'zh/release-notes/changelog.md')

function transformChangelog(content, locale = 'en') {
  // Add frontmatter
  const title = locale === 'zh' ? '更新日志' : 'Changelog'
  const description = locale === 'zh'
    ? 'LLM-Speed 版本历史与变更记录'
    : 'Version history and release notes for LLM-Speed'

  let frontmatter = `---\ntitle: ${title}\ndescription: ${description}\n---\n\n`

  // Transform version headers: ## [1.0.0] - 2026-04-30 → ## 1.0.0 (2026-04-30)
  let transformed = content
    .replace(/^## \[([\d.]+)\]\s*-\s*(\d{4}-\d{2}-\d{2})/gm, '## $1 ($2)')
    // Remove subsection headers (### Added, ### Changed, etc.) - flatten to bold labels
    .replace(/^### (Added|Changed|Fixed|Removed|Deprecated|Security|Technical Details|Archived Changes|Deferred Features)$/gm, (match, section) => {
      const sectionLabels = {
        en: {
          Added: 'Added',
          Changed: 'Changed',
          Fixed: 'Fixed',
          Removed: 'Removed',
          Deprecated: 'Deprecated',
          Security: 'Security',
          'Technical Details': 'Technical Details',
          'Archived Changes': 'Archived Changes',
          'Deferred Features': 'Deferred Features',
        },
        zh: {
          Added: '新增',
          Changed: '变更',
          Fixed: '修复',
          Removed: '移除',
          Deprecated: '弃用',
          Security: '安全',
          'Technical Details': '技术细节',
          'Archived Changes': '归档变更',
          'Deferred Features': '延迟功能',
        },
      }
      const labels = locale === 'zh' ? sectionLabels.zh : sectionLabels.en
      return `**${labels[section] || section}}**`
    })
    // Strip HTML comments
    .replace(/<!--[\s\S]*?-->/g, '')

  return frontmatter + transformed
}

function ensureDir(dirPath) {
  if (!existsSync(dirPath)) {
    mkdirSync(dirPath, { recursive: true })
  }
}

function main() {
  if (!existsSync(sourcePath)) {
    console.error(`Source file not found: ${sourcePath}`)
    process.exit(1)
  }

  const content = readFileSync(sourcePath, 'utf-8')

  // Generate English version
  const enContent = transformChangelog(content, 'en')
  ensureDir(resolve(targetEnPath, '..'))
  writeFileSync(targetEnPath, enContent, 'utf-8')
  console.log(`✓ Synced English changelog to ${targetEnPath}`)

  // Generate Chinese version
  const zhContent = transformChangelog(content, 'zh')
  ensureDir(resolve(targetZhPath, '..'))
  writeFileSync(targetZhPath, zhContent, 'utf-8')
  console.log(`✓ Synced Chinese changelog to ${targetZhPath}`)
}

main()