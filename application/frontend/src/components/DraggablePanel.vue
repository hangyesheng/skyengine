<template>
  <div
    class="draggable-panel"
    :class="{ collapsed: isCollapsed, dragging: isDragging, resizing: isResizing }"
    :style="panelStyle"
    @mouseenter="hovered = true"
    @mouseleave="hovered = false"
  >
    <!-- 标题栏 (拖拽手柄) -->
    <div class="dp-header" @mousedown.prevent="startDrag" @touchstart.prevent="startDragTouch">
      <div class="dp-header-left">
        <span v-if="icon" class="dp-icon">{{ icon }}</span>
        <span class="dp-title">{{ title }}</span>
      </div>
      <div class="dp-header-actions">
        <button class="dp-action-btn" @click.stop="isCollapsed = !isCollapsed" :title="isCollapsed ? '展开' : '折叠'">
          <span class="dp-chevron" :class="{ rotated: isCollapsed }">&#9662;</span>
        </button>
        <button class="dp-action-btn dp-close-btn" @click.stop="$emit('close')" title="关闭">
          &times;
        </button>
      </div>
    </div>

    <!-- 内容区 -->
    <div class="dp-body" v-show="!isCollapsed">
      <slot></slot>
    </div>

    <!-- 折叠态摘要 (可选) -->
    <div class="dp-collapsed-hint" v-if="isCollapsed && $slots.hint">
      <slot name="hint"></slot>
    </div>

    <!-- 缩放手柄 -->
    <div
      v-if="resizable && !isCollapsed"
      class="dp-resize-handle"
      @mousedown.prevent.stop="startResize"
      @touchstart.prevent.stop="startResizeTouch"
    ></div>
  </div>
</template>

<script setup>
import { ref, computed, onMounted, onUnmounted } from 'vue'

const props = defineProps({
  title: { type: String, default: '面板' },
  icon: { type: String, default: '' },
  width: { type: Number, default: 300 },
  /** 初始位置 { x, y }，不传则自动定位 */
  initialPos: { type: Object, default: null },
  /** 是否默认折叠 */
  defaultCollapsed: { type: Boolean, default: false },
  /** 最小宽度 */
  minWidth: { type: Number, default: 200 },
  /** 最小高度 */
  minHeight: { type: Number, default: 120 },
  /** 固定高度 (px)，0 = 自适应 */
  height: { type: Number, default: 0 },
  /** 最大高度 (px)，超出内容滚动 */
  maxHeight: { type: Number, default: 0 },
  /** 是否可缩放 */
  resizable: { type: Boolean, default: false },
})

const emit = defineEmits(['close', 'collapse', 'move', 'resize'])

// ==================== 拖拽状态 ====================

const posX = ref(0)
const posY = ref(0)
const isDragging = ref(false)
const isCollapsed = ref(props.defaultCollapsed)
const hovered = ref(false)

let dragOffsetX = 0
let dragOffsetY = 0

// ==================== 缩放状态 ====================

const panelWidth = ref(props.width)
const panelHeight = ref(props.height)
const isResizing = ref(false)

let resizeStartX = 0
let resizeStartY = 0
let resizeStartWidth = 0
let resizeStartHeight = 0

const panelStyle = computed(() => {
  const style = {
    width: `${panelWidth.value}px`,
    minWidth: `${props.minWidth}px`,
    transform: `translate(${posX.value}px, ${posY.value}px)`,
  }
  if (panelHeight.value > 0) {
    style.height = `${panelHeight.value}px`
    style.minHeight = `${props.minHeight}px`
  }
  if (props.maxHeight > 0) {
    style.maxHeight = `${props.maxHeight}px`
  }
  return style
})

// ==================== 拖拽逻辑 ====================

function startDrag(e) {
  isDragging.value = true
  dragOffsetX = e.clientX - posX.value
  dragOffsetY = e.clientY - posY.value
  document.addEventListener('mousemove', onDrag)
  document.addEventListener('mouseup', stopDrag)
}

function startDragTouch(e) {
  const touch = e.touches[0]
  isDragging.value = true
  dragOffsetX = touch.clientX - posX.value
  dragOffsetY = touch.clientY - posY.value
  document.addEventListener('touchmove', onDragTouch, { passive: false })
  document.addEventListener('touchend', stopDragTouch)
}

function onDrag(e) {
  let nx = e.clientX - dragOffsetX
  let ny = e.clientY - dragOffsetY
  // 边界约束
  nx = Math.max(-panelWidth.value + 60, Math.min(window.innerWidth - 60, nx))
  ny = Math.max(0, Math.min(window.innerHeight - 40, ny))
  posX.value = nx
  posY.value = ny
}

function onDragTouch(e) {
  e.preventDefault()
  const touch = e.touches[0]
  let nx = touch.clientX - dragOffsetX
  let ny = touch.clientY - dragOffsetY
  nx = Math.max(-panelWidth.value + 60, Math.min(window.innerWidth - 60, nx))
  ny = Math.max(0, Math.min(window.innerHeight - 40, ny))
  posX.value = nx
  posY.value = ny
}

function stopDrag() {
  isDragging.value = false
  document.removeEventListener('mousemove', onDrag)
  document.removeEventListener('mouseup', stopDrag)
  emit('move', { x: posX.value, y: posY.value })
}

function stopDragTouch() {
  isDragging.value = false
  document.removeEventListener('touchmove', onDragTouch)
  document.removeEventListener('touchend', stopDragTouch)
  emit('move', { x: posX.value, y: posY.value })
}

// ==================== 缩放逻辑 ====================

function startResize(e) {
  isResizing.value = true
  resizeStartX = e.clientX
  resizeStartY = e.clientY
  resizeStartWidth = panelWidth.value
  resizeStartHeight = panelHeight.value
  document.addEventListener('mousemove', onResize)
  document.addEventListener('mouseup', stopResize)
}

function startResizeTouch(e) {
  const touch = e.touches[0]
  isResizing.value = true
  resizeStartX = touch.clientX
  resizeStartY = touch.clientY
  resizeStartWidth = panelWidth.value
  resizeStartHeight = panelHeight.value
  document.addEventListener('touchmove', onResizeTouch, { passive: false })
  document.addEventListener('touchend', stopResizeTouch)
}

function onResize(e) {
  const newWidth = Math.max(props.minWidth, resizeStartWidth + (e.clientX - resizeStartX))
  const newHeight = Math.max(props.minHeight, resizeStartHeight + (e.clientY - resizeStartY))

  panelWidth.value = Math.min(newWidth, window.innerWidth - posX.value - 20)
  panelHeight.value = props.maxHeight > 0
    ? Math.min(newHeight, props.maxHeight)
    : Math.min(newHeight, window.innerHeight - posY.value - 20)
}

function onResizeTouch(e) {
  e.preventDefault()
  const touch = e.touches[0]
  const newWidth = Math.max(props.minWidth, resizeStartWidth + (touch.clientX - resizeStartX))
  const newHeight = Math.max(props.minHeight, resizeStartHeight + (touch.clientY - resizeStartY))

  panelWidth.value = Math.min(newWidth, window.innerWidth - posX.value - 20)
  panelHeight.value = props.maxHeight > 0
    ? Math.min(newHeight, props.maxHeight)
    : Math.min(newHeight, window.innerHeight - posY.value - 20)
}

function stopResize() {
  isResizing.value = false
  document.removeEventListener('mousemove', onResize)
  document.removeEventListener('mouseup', stopResize)
  emit('resize', { width: panelWidth.value, height: panelHeight.value })
}

function stopResizeTouch() {
  isResizing.value = false
  document.removeEventListener('touchmove', onResizeTouch)
  document.removeEventListener('touchend', stopResizeTouch)
  emit('resize', { width: panelWidth.value, height: panelHeight.value })
}

onUnmounted(() => {
  document.removeEventListener('mousemove', onDrag)
  document.removeEventListener('mouseup', stopDrag)
  document.removeEventListener('touchmove', onDragTouch)
  document.removeEventListener('touchend', stopDragTouch)
  document.removeEventListener('mousemove', onResize)
  document.removeEventListener('mouseup', stopResize)
  document.removeEventListener('touchmove', onResizeTouch)
  document.removeEventListener('touchend', stopResizeTouch)
})

// ==================== 初始化位置 ====================

onMounted(() => {
  if (props.initialPos) {
    posX.value = props.initialPos.x
    posY.value = props.initialPos.y
  }
})
</script>

<style scoped>
@import './styles/DraggablePanel.css';
</style>
