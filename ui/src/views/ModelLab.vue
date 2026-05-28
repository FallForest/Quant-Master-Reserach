<script setup>
import { ref, computed, onMounted } from 'vue'
import { api } from '../utils/api'
import { archLayers, defaultLayers } from '../data/modelArchLayers'

const SPEED_LABELS = ['', '慢', '较慢', '中等', '较快', '快']
const SPEED_COLORS = ['', 'text-danger', 'text-warn', 'text-slate-500', 'text-brand-600', 'text-success']

const CATEGORY_COLORS = {
  '树模型': 'bg-emerald-50 text-emerald-600',
  'RNN': 'bg-blue-50 text-blue-600',
  '注意力': 'bg-purple-50 text-purple-600',
  'CNN': 'bg-amber-50 text-amber-600',
  'DNN': 'bg-slate-100 text-slate-600',
  '集成': 'bg-rose-50 text-rose-600',
  '线性': 'bg-cyan-50 text-cyan-600',
  '频域': 'bg-indigo-50 text-indigo-600',
  'RL混合': 'bg-orange-50 text-orange-600',
  '高级集成': 'bg-pink-50 text-pink-600',
  '信号': 'bg-teal-50 text-teal-600',
  'Transcendence': 'bg-violet-50 text-violet-600',
}

const LAYER_COLORS = {
  input: 'bg-blue-500',
  hidden: 'bg-violet-500',
  output: 'bg-emerald-500',
}

const loading = ref(true)
const error = ref(null)
const models = ref([])
const categories = ref([])
const filterCategory = ref('all')
const expandedId = ref(null)

const filtered = computed(() => {
  if (filterCategory.value === 'all') return models.value
  return models.value.filter(m => m.category === filterCategory.value)
})

const selectedModel = computed(() => {
  if (!expandedId.value) return null
  return models.value.find(m => m.id === expandedId.value) || null
})

const expandedLayers = computed(() => {
  if (!selectedModel.value) return []
  const m = selectedModel.value
  return archLayers[m.id] || defaultLayers(m.name)
})

onMounted(async () => {
  loading.value = true
  error.value = null
  try {
    const data = await api('/api/model-catalog')
    models.value = data?.models || []
    categories.value = data?.categories || []
  } catch (e) {
    error.value = e?.message || '加载模型列表失败'
  } finally {
    loading.value = false
  }
})

function toggleExpand(m) {
  expandedId.value = expandedId.value === m.id ? null : m.id
}

function onCardKeydown(e, m) {
  if (e.key === 'Enter' || e.key === ' ') {
    e.preventDefault()
    toggleExpand(m)
  }
}
</script>

<template>
  <div class="p-4 sm:p-6 space-y-5 animate-slide-in">

    <!-- 标题 + 筛选 -->
    <div class="flex flex-wrap items-center gap-3">
      <div class="flex items-center gap-2">
        <svg class="w-5 h-5 text-brand-500" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" d="M9.75 3.104v5.714a2.25 2.25 0 01-.659 1.591L5 14.5M9.75 3.104c-.251.023-.501.05-.75.082m.75-.082a24.301 24.301 0 014.5 0m0 0v5.714c0 .597.237 1.17.659 1.591L19.8 15.3M14.25 3.104c.251.023.501.05.75.082M19.8 15.3l-1.57.393A9.065 9.065 0 0112 15a9.065 9.065 0 00-6.23.693L5 14.5m14.8.8l1.402 1.402c1.232 1.232.65 3.318-1.067 3.611A48.309 48.309 0 0112 21c-2.773 0-5.491-.235-8.135-.687-1.718-.293-2.3-2.379-1.067-3.61L5 14.5"/>
        </svg>
        <h2 class="text-base font-semibold text-slate-700">模型工坊</h2>
        <span class="text-xs text-slate-400 ml-1">{{ models.length }} 个模型</span>
      </div>
      <div class="flex-1"></div>
      <div class="flex items-center bg-surface-2 rounded-xl p-0.5 gap-0.5 overflow-x-auto">
        <button v-for="c in ['all', ...categories]" :key="c" @click="filterCategory = c"
          :aria-pressed="filterCategory === c"
          :class="['px-3 py-1.5 text-xs font-semibold rounded-lg cursor-pointer transition-all duration-200 whitespace-nowrap',
                   filterCategory === c ? 'bg-white text-brand-600 shadow-sm' : 'text-slate-500 hover:text-slate-700']">
          {{ c === 'all' ? '全部' : c }}
        </button>
      </div>
    </div>

    <!-- 错误状态 -->
    <div v-if="error" class="flex flex-col items-center justify-center py-16 text-center">
      <svg class="w-10 h-10 text-slate-300 mb-3" fill="none" stroke="currentColor" stroke-width="1.5" viewBox="0 0 24 24">
        <path stroke-linecap="round" stroke-linejoin="round" d="M12 9v3.75m9-.75a9 9 0 11-18 0 9 9 0 0118 0zm-9 3.75h.008v.008H12v-.008z"/>
      </svg>
      <p class="text-sm text-slate-500">{{ error }}</p>
    </div>

    <!-- 骨架屏 -->
    <div v-else-if="loading" class="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-3">
      <div v-for="i in 6" :key="i" class="bg-white rounded-xl border border-surface-3 p-4">
        <div class="flex items-center gap-3 mb-3">
          <div class="skeleton h-5 w-5 rounded"></div>
          <div class="skeleton h-4 w-24"></div>
          <div class="skeleton h-5 w-12 rounded-full"></div>
        </div>
        <div class="skeleton h-3 w-full mb-1"></div>
        <div class="skeleton h-3 w-3/4"></div>
      </div>
    </div>

    <!-- 模型卡片网格 -->
    <div v-else class="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-3">
      <div v-for="m in filtered" :key="m.id"
           role="button" tabindex="0"
           :aria-expanded="expandedId === m.id"
           @click="toggleExpand(m)"
           @keydown="onCardKeydown($event, m)"
           :class="['bg-white rounded-xl border border-surface-3 overflow-hidden cursor-pointer transition-all duration-200 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-brand-400',
                    expandedId === m.id ? 'ring-2 ring-brand-400 shadow-sm md:col-span-2 xl:col-span-3' : 'hover:shadow-sm hover:border-brand-200']">
        <!-- 卡片头部 -->
        <div class="p-4">
          <div class="flex items-center gap-2 mb-2">
            <svg class="w-5 h-5 text-brand-500 flex-shrink-0" fill="none" stroke="currentColor" stroke-width="1.5" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" d="M21 7.5l-9-5.25L3 7.5m18 0l-9 5.25m9-5.25v9l-9 5.25M3 7.5l9 5.25M3 7.5v9l9 5.25m0-9v9"/>
            </svg>
            <span class="font-semibold text-sm text-slate-700">{{ m.name }}</span>
            <span :class="['text-[10px] px-2 py-0.5 rounded-full font-medium', CATEGORY_COLORS[m.category] || 'bg-surface-2 text-slate-500']">
              {{ m.category }}
            </span>
            <div class="flex-1"></div>
            <div class="flex items-center gap-1">
              <span class="text-[10px] text-slate-400">速度</span>
              <span :class="['text-xs font-semibold', SPEED_COLORS[m.speed] || 'text-slate-400']">{{ SPEED_LABELS[m.speed] || '-' }}</span>
            </div>
          </div>
          <p class="text-xs text-slate-500 leading-relaxed line-clamp-2">{{ m.desc }}</p>
          <!-- 复杂度条 -->
          <div class="mt-2.5 flex items-center gap-2">
            <span class="text-[10px] text-slate-400">复杂度</span>
            <div class="flex-1 h-1.5 bg-surface-2 rounded-full overflow-hidden">
              <div class="h-full rounded-full bg-brand-400 transition-all duration-300"
                   :style="{ width: (m.complexity / 5 * 100) + '%' }"></div>
            </div>
            <span class="text-[10px] font-mono text-slate-400">{{ m.complexity }}/5</span>
          </div>
        </div>

        <!-- 展开详情 -->
        <div v-if="expandedId === m.id" class="px-4 pb-4 border-t border-surface-3">
          <div class="grid grid-cols-1 lg:grid-cols-2 gap-4 mt-3">
            <!-- 参数 -->
            <div>
              <h4 class="text-xs font-semibold text-slate-500 uppercase tracking-wide mb-2">超参数配置</h4>
              <div class="bg-surface-2/50 rounded-lg p-3 space-y-2">
                <div v-for="(p, k) in m.params" :key="k" class="flex items-center justify-between text-xs">
                  <span class="text-slate-500">{{ p.label }}</span>
                  <span class="text-slate-700 font-mono">
                    {{ p.default }}{{ p.min != null ? ` [${p.min}~${p.max}]` : '' }}
                  </span>
                </div>
              </div>
            </div>
            <!-- 架构图 -->
            <div>
              <h4 class="text-xs font-semibold text-slate-500 uppercase tracking-wide mb-2">模型架构</h4>
              <div role="img" :aria-label="(selectedModel?.name || '') + ' 模型架构'"
                   class="w-full min-h-[80px] bg-surface-2/50 rounded-lg p-3 flex items-center justify-center gap-1 overflow-x-auto">
                <template v-for="(layer, i) in expandedLayers" :key="layer.name">
                  <svg v-if="i > 0" class="w-4 h-4 text-slate-300 flex-shrink-0" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24">
                    <path stroke-linecap="round" stroke-linejoin="round" d="M13.5 4.5L21 12m0 0l-7.5 7.5M21 12H3"/>
                  </svg>
                  <div :class="[LAYER_COLORS[layer.type] || 'bg-slate-400', 'flex-shrink-0 rounded-md px-3 py-1.5 text-[10px] text-white font-medium text-center']"
                       :style="{ minWidth: Math.min(layer.w, 160) + 'px' }">
                    {{ layer.name }}
                  </div>
                </template>
              </div>
            </div>
          </div>
          <div class="mt-3 flex items-center gap-2">
            <router-link :to="{ path: '/backtest', query: { model: m.id } }"
              class="flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium text-white bg-brand-600 rounded-lg hover:bg-brand-700 transition cursor-pointer">
              <svg class="w-3.5 h-3.5" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"/>
              </svg>
              用此模型回测
            </router-link>
            <router-link to="/experiments"
              class="flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium text-brand-600 bg-brand-50 rounded-lg hover:bg-brand-100 transition cursor-pointer">
              <svg class="w-3.5 h-3.5" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" d="M19.5 14.25v-2.625a3.375 3.375 0 00-3.375-3.375h-1.5A1.125 1.125 0 0113.5 7.125v-1.5a3.375 3.375 0 00-3.375-3.375H8.25m0 12.75h7.5m-7.5 3H12M10.5 2.25H5.625c-.621 0-1.125.504-1.125 1.125v17.25c0 .621.504 1.125 1.125 1.125h12.75c.621 0 1.125-.504 1.125-1.125V11.25a9 9 0 00-9-9z"/>
              </svg>
              查看实验
            </router-link>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>
