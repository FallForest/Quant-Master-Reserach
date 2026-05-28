<script setup>
import { ref, computed, onMounted } from 'vue'
import { api } from '../utils/api'

const experiments = ref([])
const loading = ref(true)
const selectedExp = ref(null)
const filterStatus = ref('all')

const filteredExps = computed(() => {
  if (filterStatus.value === 'all') return experiments.value
  return experiments.value.filter(e => e.status === filterStatus.value)
})

onMounted(async () => {
  loading.value = true
  const data = await api('/api/experiments')
  experiments.value = data?.experiments || []
  loading.value = false
})

function selectExp(exp) {
  selectedExp.value = selectedExp.value?.id === exp.id ? null : exp
}

function statusBadge(status) {
  const map = {
    finished: { text: '已完成', cls: 'text-success bg-success/10' },
    running:  { text: '运行中', cls: 'text-brand-600 bg-brand-50' },
    failed:   { text: '失败', cls: 'text-danger bg-danger/10' },
  }
  return map[status] || { text: status, cls: 'text-slate-500 bg-surface-2' }
}
</script>

<template>
  <div class="p-4 sm:p-6 space-y-5 animate-slide-in">

    <!-- 标题 + 筛选 -->
    <div class="flex flex-wrap items-center gap-3">
      <div class="flex items-center gap-2">
        <svg class="w-5 h-5 text-brand-500" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" d="M19.5 14.25v-2.625a3.375 3.375 0 00-3.375-3.375h-1.5A1.125 1.125 0 0113.5 7.125v-1.5a3.375 3.375 0 00-3.375-3.375H8.25m0 12.75h7.5m-7.5 3H12M10.5 2.25H5.625c-.621 0-1.125.504-1.125 1.125v17.25c0 .621.504 1.125 1.125 1.125h12.75c.621 0 1.125-.504 1.125-1.125V11.25a9 9 0 00-9-9z"/>
        </svg>
        <h2 class="text-base font-semibold text-slate-700">实验管理</h2>
      </div>
      <div class="flex-1"></div>
      <div class="flex items-center bg-surface-2 rounded-xl p-0.5 gap-0.5">
        <button v-for="f in [{v:'all',l:'全部'},{v:'finished',l:'已完成'},{v:'running',l:'运行中'},{v:'failed',l:'失败'}]" :key="f.v"
                @click="filterStatus = f.v"
          :class="['px-3 py-1.5 text-xs font-semibold rounded-lg cursor-pointer transition-all duration-200',
                   filterStatus === f.v ? 'bg-white text-brand-600 shadow-sm' : 'text-slate-500 hover:text-slate-700']">
          {{ f.l }}
        </button>
      </div>
      <span class="text-xs text-slate-400">{{ filteredExps.length }} 个实验</span>
    </div>

    <!-- 骨架屏 -->
    <div v-if="loading" class="space-y-3">
      <div v-for="i in 4" :key="i" class="bg-white rounded-xl border border-surface-3 p-5">
        <div class="flex items-center gap-3">
          <div class="skeleton h-4 w-40"></div>
          <div class="skeleton h-5 w-16 rounded-full"></div>
          <div class="flex-1"></div>
          <div class="skeleton h-3 w-24"></div>
        </div>
      </div>
    </div>

    <!-- 实验列表 -->
    <div v-else class="space-y-3">
      <div v-for="exp in filteredExps" :key="exp.id"
           @click="selectExp(exp)"
           :class="['bg-white rounded-xl border border-surface-3 overflow-hidden cursor-pointer transition-all duration-200',
                    selectedExp?.id === exp.id ? 'ring-2 ring-brand-400 shadow-sm' : 'hover:shadow-sm hover:border-brand-200']">
        <!-- 实验头部 -->
        <div class="px-5 py-3.5 flex items-center gap-3">
          <svg class="w-5 h-5 text-slate-400 flex-shrink-0" fill="none" stroke="currentColor" stroke-width="1.5" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" d="M9.75 3.104v5.714a2.25 2.25 0 01-.659 1.591L5 14.5M9.75 3.104c-.251.023-.501.05-.75.082m.75-.082a24.301 24.301 0 014.5 0m0 0v5.714c0 .597.237 1.17.659 1.591L19.8 15.3M14.25 3.104c.251.023.501.05.75.082M19.8 15.3l-1.57.393A9.065 9.065 0 0112 15a9.065 9.065 0 00-6.23.693L5 14.5m14.8.8l1.402 1.402c1.232 1.232.65 3.318-1.067 3.611A48.309 48.309 0 0112 21c-2.773 0-5.491-.235-8.135-.687-1.718-.293-2.3-2.379-1.067-3.61L5 14.5"/>
          </svg>
          <span class="font-semibold text-sm text-slate-700">{{ exp.name }}</span>
          <span :class="['text-[10px] px-2 py-0.5 rounded-full font-medium', statusBadge(exp.status).cls]">
            {{ statusBadge(exp.status).text }}
          </span>
          <div class="flex-1"></div>
          <span class="text-xs text-slate-400 font-mono">{{ exp.createTime }}</span>
          <span class="text-xs text-slate-400">{{ exp.duration }}</span>
          <svg :class="['w-4 h-4 text-slate-300 transition-transform', selectedExp?.id === exp.id ? 'rotate-180' : '']"
               fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" d="M19 9l-7 7-7-7"/>
          </svg>
        </div>

        <!-- 展开详情 -->
        <div v-if="selectedExp?.id === exp.id" class="px-5 pb-4 border-t border-surface-3">
          <div class="grid grid-cols-1 md:grid-cols-2 gap-4 mt-3">
            <!-- 参数 -->
            <div>
              <h4 class="text-xs font-semibold text-slate-500 uppercase tracking-wide mb-2">参数配置</h4>
              <div class="bg-surface-2/50 rounded-lg p-3 space-y-1.5">
                <div v-for="(v, k) in exp.params" :key="k" class="flex items-center text-xs">
                  <span class="text-slate-500 w-24 flex-shrink-0">{{ k }}</span>
                  <span class="text-slate-700 font-mono">{{ v }}</span>
                </div>
              </div>
            </div>
            <!-- 指标 -->
            <div>
              <h4 class="text-xs font-semibold text-slate-500 uppercase tracking-wide mb-2">评估指标</h4>
              <div class="grid grid-cols-2 gap-2">
                <div v-for="(v, k) in exp.metrics" :key="k" class="bg-surface-2/50 rounded-lg p-2.5 text-center">
                  <div class="text-[10px] text-slate-500">{{ k }}</div>
                  <div class="text-sm font-bold font-mono text-brand-600">{{ v }}</div>
                </div>
              </div>
            </div>
          </div>
          <!-- 产物 -->
          <div class="mt-3">
            <h4 class="text-xs font-semibold text-slate-500 uppercase tracking-wide mb-2">产物文件</h4>
            <div class="flex flex-wrap gap-2">
              <span v-for="a in exp.artifacts" :key="a"
                    class="inline-flex items-center gap-1 px-2.5 py-1 bg-surface-2 rounded-md text-xs font-mono text-slate-600">
                <svg class="w-3 h-3 text-slate-400" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24">
                  <path stroke-linecap="round" stroke-linejoin="round" d="M19.5 14.25v-2.625a3.375 3.375 0 00-3.375-3.375h-1.5A1.125 1.125 0 0113.5 7.125v-1.5a3.375 3.375 0 00-3.375-3.375H8.25m2.25 0H5.625c-.621 0-1.125.504-1.125 1.125v17.25c0 .621.504 1.125 1.125 1.125h12.75c.621 0 1.125-.504 1.125-1.125V11.25a9 9 0 00-9-9z"/>
                </svg>
                {{ a }}
              </span>
            </div>
          </div>
          <!-- 操作 -->
          <div v-if="exp.status === 'finished'" class="mt-3 flex items-center gap-2">
            <router-link :to="{ path: '/backtest', query: { model: exp.params?.model?.toLowerCase() } }"
              class="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium text-white bg-brand-600 rounded-lg hover:bg-brand-700 transition cursor-pointer">
              <svg class="w-3.5 h-3.5" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"/>
              </svg>
              回测此模型
            </router-link>
            <router-link to="/model-performance"
              class="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium text-brand-600 bg-brand-50 rounded-lg hover:bg-brand-100 transition cursor-pointer">
              绩效分析
            </router-link>
          </div>
        </div>
      </div>

      <div v-if="!filteredExps.length" class="py-16 text-center">
        <svg class="w-12 h-12 text-slate-200 mx-auto mb-3" fill="none" stroke="currentColor" stroke-width="1" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" d="M19.5 14.25v-2.625a3.375 3.375 0 00-3.375-3.375h-1.5A1.125 1.125 0 0113.5 7.125v-1.5a3.375 3.375 0 00-3.375-3.375H8.25m2.25 0H5.625c-.621 0-1.125.504-1.125 1.125v17.25c0 .621.504 1.125 1.125 1.125h12.75c.621 0 1.125-.504 1.125-1.125V11.25a9 9 0 00-9-9z"/>
        </svg>
        <span class="text-sm text-slate-400">暂无实验记录</span>
      </div>
    </div>
  </div>
</template>
