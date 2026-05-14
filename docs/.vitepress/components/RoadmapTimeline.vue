<script setup lang="ts">
/**
 * RoadmapTimeline Component
 * Visualizes the optimization progression from Naive to Optimized
 */
interface Stage {
  name: string
  description: string
  memory: string
  status: 'complete' | 'current' | 'planned'
}

interface Props {
  stages?: Stage[]
}

const props = defineProps<Props>()

const stages = props.stages ?? [
  {
    name: 'Naive',
    description: 'Baseline implementation',
    memory: 'O(N²)',
    status: 'complete' as const,
  },
  {
    name: 'Tiled',
    description: 'Shared memory tiling',
    memory: 'Reduced global access',
    status: 'complete' as const,
  },
  {
    name: 'FlashAttention',
    description: 'Online softmax',
    memory: 'O(N)',
    status: 'complete' as const,
  },
  {
    name: 'Optimized',
    description: 'Double buffering',
    memory: 'Compute/memory overlap',
    status: 'current' as const,
  },
]
</script>

<template>
  <div class="roadmap-timeline">
    <div class="timeline-track">
      <div
        v-for="(stage, index) in stages"
        :key="stage.name"
        class="timeline-stage"
        :class="[stage.status, { last: index === stages.length - 1 }]"
      >
        <div class="stage-node">
          <div class="node-circle">
            <span v-if="stage.status === 'complete'" class="check">✓</span>
            <span v-else-if="stage.status === 'current'" class="current">●</span>
            <span v-else class="planned">○</span>
          </div>
          <div v-if="index < stages.length - 1" class="node-line"></div>
        </div>

        <div class="stage-content">
          <h5 class="stage-name">{{ stage.name }}</h5>
          <p class="stage-description">{{ stage.description }}</p>
          <span class="stage-memory">{{ stage.memory }}</span>
        </div>
      </div>
    </div>
  </div>
</template>

<style scoped>
.roadmap-timeline {
  padding: 1.5rem 0;
  overflow-x: auto;
}

.timeline-track {
  display: flex;
  gap: 0;
  min-width: max-content;
}

.timeline-stage {
  display: flex;
  flex-direction: column;
  align-items: center;
  min-width: 160px;
  position: relative;
}

.stage-node {
  display: flex;
  align-items: center;
  position: relative;
}

.node-circle {
  width: 36px;
  height: 36px;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 1rem;
  font-weight: 600;
  position: relative;
  z-index: 2;
}

.timeline-stage.complete .node-circle {
  background: rgba(118, 185, 0, 0.2);
  border: 2px solid #76b900;
  color: #76b900;
}

.timeline-stage.current .node-circle {
  background: rgba(0, 160, 224, 0.2);
  border: 2px solid #00a0e0;
  color: #00a0e0;
}

.timeline-stage.planned .node-circle {
  background: var(--vp-c-bg-soft);
  border: 2px solid var(--vp-c-border);
  color: var(--vp-c-text-3);
}

.node-line {
  position: absolute;
  left: 100%;
  width: 124px;
  height: 2px;
  top: 50%;
  transform: translateY(-50%);
}

.timeline-stage.complete .node-line {
  background: linear-gradient(90deg, #76b900 0%, var(--vp-c-border) 100%);
}

.timeline-stage.current .node-line {
  background: linear-gradient(90deg, #00a0e0 0%, var(--vp-c-border) 100%);
}

.timeline-stage.planned .node-line,
.timeline-stage.last .node-line {
  display: none;
}

.stage-content {
  text-align: center;
  margin-top: 0.75rem;
  padding: 0 0.5rem;
}

.stage-name {
  margin: 0;
  font-size: 0.95rem;
  font-weight: 600;
  color: var(--vp-c-text-1);
}

.timeline-stage.current .stage-name {
  color: #00a0e0;
}

.stage-description {
  margin: 0.25rem 0;
  font-size: 0.8rem;
  color: var(--vp-c-text-2);
}

.stage-memory {
  display: inline-block;
  padding: 0.15rem 0.5rem;
  border-radius: 4px;
  font-size: 0.75rem;
  font-family: var(--vp-font-family-mono);
}

.timeline-stage.complete .stage-memory {
  background: rgba(118, 185, 0, 0.1);
  color: #76b900;
}

.timeline-stage.current .stage-memory {
  background: rgba(0, 160, 224, 0.1);
  color: #00a0e0;
}

.timeline-stage.planned .stage-memory {
  background: var(--vp-c-bg-soft);
  color: var(--vp-c-text-3);
}
</style>
