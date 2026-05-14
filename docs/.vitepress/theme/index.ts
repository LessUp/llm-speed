/**
 * LLM-Speed Docs - VitePress Custom Theme
 * "CUDA Green + Dark Metal" Design System
 *
 * Features:
 * - NVIDIA CUDA Green brand color (#76b900)
 * - GPU control panel visual aesthetic
 * - Performance dashboard feel
 * - Optimized for technical documentation
 */

import DefaultTheme from 'vitepress/theme'
import { onMounted, watch, nextTick } from 'vue'
import { useRoute } from 'vitepress'

import './style.css'

// Import custom components
import KernelCard from '../components/KernelCard.vue'
import CodeDemo from '../components/CodeDemo.vue'
import GPUSupport from '../components/GPUSupport.vue'
import MemoryCompare from '../components/MemoryCompare.vue'
import PerfChart from '../components/PerfChart.vue'
import AlgorithmCard from '../components/AlgorithmCard.vue'
import CodeDiff from '../components/CodeDiff.vue'
import RoadmapTimeline from '../components/RoadmapTimeline.vue'

export default {
  extends: DefaultTheme,

  setup() {
    const route = useRoute()

    // Handle mermaid diagrams on route change
    onMounted(async () => {
      const { initMermaid } = await import('vitepress-plugin-mermaid')
      initMermaid()
    })

    watch(
      () => route.path,
      () => {
        nextTick(async () => {
          const { initMermaid } = await import('vitepress-plugin-mermaid')
          initMermaid()
        })
      }
    )
  },

  enhanceApp({ app }) {
    // Register custom components globally
    app.component('KernelCard', KernelCard)
    app.component('CodeDemo', CodeDemo)
    app.component('GPUSupport', GPUSupport)
    app.component('MemoryCompare', MemoryCompare)
    app.component('PerfChart', PerfChart)
    app.component('AlgorithmCard', AlgorithmCard)
    app.component('CodeDiff', CodeDiff)
    app.component('RoadmapTimeline', RoadmapTimeline)
  },
}
