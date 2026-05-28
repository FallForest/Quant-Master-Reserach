<script setup>
import { ref, computed, watch, onMounted, onUnmounted, nextTick } from 'vue'
import { api } from '../utils/api'
import { useToast } from '../utils/toast'

const toast = useToast()

const running = ref(false)
const progress = ref(0)
const stepText = ref('')
const logs = ref([])
const models = ref([])
const handlers = ref([])
const results = ref(null)

// 配置
const cfg = ref({
  model_id: 'double_ensemble',
  handler_id: 'alpha158',
  universe: '500',
  test_date: new Date().toISOString().slice(0, 10),
  train_start: '2018-01-01',
  train_end: '2023-12-31',
  valid_start: '2024-01-01',
  valid_end: '2025-06-30',
  top_n: 50,
})

// 模型切换时自动推荐 handler
watch(() => cfg.value.model_id, (newId) => {
  const m = models.value.find(m => m.id === newId)
  if (m?.handler) cfg.value.handler_id = m.handler
})

// ---- 可搜索模型选择器 ----
const modelSearch = ref('')
const modelDropdownOpen = ref(false)
const modelSearchInput = ref(null)

const filteredModels = computed(() => {
  const q = modelSearch.value.toLowerCase().trim()
  if (!q) return models.value
  return models.value.filter(m => (
    m.label.toLowerCase().includes(q) ||
    m.category.toLowerCase().includes(q) ||
    (m.desc && m.desc.toLowerCase().includes(q)) ||
    m.id.toLowerCase().includes(q)
  ))
})

const filteredModelCategories = computed(() => {
  const cats = {}
  filteredModels.value.forEach(m => {
    if (!cats[m.category]) cats[m.category] = []
    cats[m.category].push(m)
  })
  return cats
})

const selectedModel = computed(() => models.value.find(m => m.id === cfg.value.model_id))

function toggleModelDropdown() {
  modelDropdownOpen.value = !modelDropdownOpen.value
  if (modelDropdownOpen.value) {
    modelSearch.value = ''
    nextTick(() => modelSearchInput.value?.focus())
  }
}

function selectModel(id) {
  cfg.value.model_id = id
  modelDropdownOpen.value = false
  modelSearch.value = ''
}

function closeModelDropdown() {
  setTimeout(() => { modelDropdownOpen.value = false }, 150)
}

let pollTimer = null
let runStartTime = 0

// 运行历史
const history = ref([])
function loadHistory() {
  try { history.value = JSON.parse(localStorage.getItem('stock_select_history') || '[]') } catch { history.value = [] }
}
function saveHistory() {
  localStorage.setItem('stock_select_history', JSON.stringify(history.value.slice(0, 50)))
}
function recordRun(success, error) {
  const elapsed = runStartTime ? ((Date.now() - runStartTime) / 1000).toFixed(1) + 's' : '--'
  const modelLabel = models.value.find(m => m.id === cfg.value.model_id)?.label || cfg.value.model_id
  history.value.unshift({
    time: new Date().toLocaleString('zh-CN', { hour12: false }),
    model: modelLabel,
    test_date: cfg.value.test_date,
    top_n: cfg.value.top_n,
    success,
    elapsed,
    error: error || '',
  })
  saveHistory()
}

onMounted(async () => {
  loadHistory()
  const data = await api('/api/models')
  if (data) {
    models.value = data.models || []
    handlers.value = data.handlers || []
  }
})

onUnmounted(() => {
  if (pollTimer) clearInterval(pollTimer)
})

function appendLog(level, msg) {
  logs.value.push({ level, msg })
}

function setProgress(pct) { progress.value = pct }

async function startSelection() {
  if (!cfg.value.test_date) {
    toast.error('请选择目标日期')
    return
  }
  running.value = true
  progress.value = 0
  logs.value = []
  results.value = null
  runStartTime = Date.now()
  appendLog('info', '正在启动选股任务...')

  const result = await api('/api/stock-select/run', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(cfg.value),
  })

  if (result?.runId) {
    pollRun(result.runId)
  } else {
    appendLog('error', '启动失败: ' + (result?.error || '无法连接后端'))
    running.value = false
    toast.error(result?.error || '无法连接后端')
  }
}

function cancelSelection() {
  if (pollTimer) clearInterval(pollTimer)
  appendLog('warn', '用户取消')
  running.value = false
}

function pollRun(runId) {
  pollTimer = setInterval(async () => {
    const status = await api(`/api/stock-select/status/${runId}`)
    if (!status) return
    if (status.logs) {
      status.logs.forEach(l => {
        // 避免重复日志
        if (!logs.value.find(existing => existing.msg === l.msg)) {
          appendLog(l.level, l.msg)
        }
      })
    }
    if (status.progress != null) setProgress(status.progress)
    if (status.step) stepText.value = status.step
    if (status.done) {
      clearInterval(pollTimer)
      setProgress(100)
      if (status.success) {
        appendLog('success', '选股完成!')
        toast.success('选股完成')
        // 拉取结果
        const res = await api(`/api/stock-select/results/${runId}`)
        if (res?.results) results.value = res.results
      } else {
        appendLog('error', '选股失败: ' + (status.error || '未知错误'))
        toast.error('选股失败: ' + (status.error || '未知错误'))
      }
      recordRun(status.success, status.error)
      running.value = false
    }
  }, 3000)
}

function exportCSV() {
  if (!results.value?.top?.length) return
  const rows = [['排名', '股票代码', '股票名称', '预测分数']]
  results.value.top.forEach(r => rows.push([r.rank, r.symbol, r.name, r.score]))
  const csv = '﻿' + rows.map(r => r.join(',')).join('\n')
  const blob = new Blob([csv], { type: 'text/csv;charset=utf-8' })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = `stock_selection_${cfg.value.model_id}_${cfg.value.test_date}.csv`
  a.click()
  URL.revokeObjectURL(url)
}

const logColor = {
  info: 'text-brand-200',
  warn: 'text-amber-400',
  error: 'text-red-400',
  success: 'text-emerald-400',
}

function maxScore(items) {
  if (!items?.length) return 1
  return Math.max(...items.map(r => Math.abs(r.score)))
}
</script>

<template>
  <div class="p-4 sm:p-6 space-y-6 animate-slide-in">

    <!-- 配置面板 -->
    <div class="bg-white rounded-xl border border-surface-3 p-5">
      <div class="flex items-center gap-2 mb-4">
        <svg class="w-5 h-5 text-brand-500" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" d="M3 3v18h18M9 17V9m4 8V5m4 12v-4"/>
        </svg>
        <h2 class="text-base font-semibold text-slate-800">模型配置</h2>
      </div>

      <div class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
        <!-- 模型选择（可搜索） -->
        <div class="relative">
          <label class="block text-xs text-slate-500 mb-1">模型</label>
          <button @click="toggleModelDropdown" type="button"
            class="w-full flex items-center justify-between px-3 py-2 text-sm rounded-lg border border-surface-3
                   hover:border-brand-400 focus:border-brand-500 focus:ring-1 focus:ring-brand-500
                   outline-none transition cursor-pointer bg-white text-left">
            <div class="flex-1 min-w-0">
              <div class="font-medium text-slate-700 truncate">{{ selectedModel?.label || '选择模型' }}</div>
              <div v-if="selectedModel?.desc" class="text-xs text-slate-400 truncate">{{ selectedModel.desc }}</div>
            </div>
            <svg class="w-4 h-4 text-slate-400 flex-shrink-0 ml-2 transition-transform"
                 :class="modelDropdownOpen && 'rotate-180'"
                 fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" d="M19 9l-7 7-7-7"/>
            </svg>
          </button>
          <!-- 下拉面板 -->
          <div v-if="modelDropdownOpen"
               @blur="closeModelDropdown"
               class="absolute z-50 mt-1 w-full bg-white rounded-lg border border-surface-3 shadow-lg overflow-hidden"
               style="max-height: 380px;">
            <!-- 搜索框 -->
            <div class="p-2 border-b border-surface-3 sticky top-0 bg-white">
              <div class="relative">
                <svg class="absolute left-2.5 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-400" fill="none"
                     stroke="currentColor" stroke-width="2" viewBox="0 0 24 24">
                  <path stroke-linecap="round" stroke-linejoin="round" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"/>
                </svg>
                <input ref="modelSearchInput" v-model="modelSearch" type="text" placeholder="搜索模型..."
                  class="w-full pl-8 pr-3 py-1.5 text-sm rounded-md border border-surface-3 focus:border-brand-500
                         focus:ring-1 focus:ring-brand-500 outline-none">
              </div>
            </div>
            <!-- 模型列表 -->
            <div class="overflow-y-auto" style="max-height: 310px;">
              <template v-for="(list, cat) in filteredModelCategories" :key="cat">
                <div class="px-3 py-1.5 text-xs font-semibold text-slate-400 bg-surface-2/60 sticky top-0">{{ cat }}</div>
                <button v-for="m in list" :key="m.id" @click="selectModel(m.id)" type="button"
                  :class="[
                    'w-full text-left px-3 py-2 hover:bg-brand-50 transition cursor-pointer',
                    cfg.model_id === m.id ? 'bg-brand-50 border-l-2 border-brand-500' : 'border-l-2 border-transparent'
                  ]">
                  <div class="flex items-center justify-between">
                    <div class="font-medium text-sm text-slate-700">{{ m.label }}</div>
                    <svg v-if="cfg.model_id === m.id" class="w-4 h-4 text-brand-500 flex-shrink-0" fill="currentColor" viewBox="0 0 20 20">
                      <path fill-rule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clip-rule="evenodd"/>
                    </svg>
                  </div>
                  <div v-if="m.desc" class="text-xs text-slate-400 mt-0.5">{{ m.desc }}</div>
                </button>
              </template>
              <div v-if="!Object.keys(filteredModelCategories).length" class="px-3 py-6 text-center text-sm text-slate-400">
                没有匹配的模型
              </div>
            </div>
          </div>
        </div>
        <!-- 数据处理器 -->
        <div>
          <label class="block text-xs text-slate-500 mb-1">数据处理器</label>
          <select v-model="cfg.handler_id"
            class="w-full px-3 py-2 text-sm rounded-lg border border-surface-3 focus:border-brand-500
                   focus:ring-1 focus:ring-brand-500 outline-none transition cursor-pointer bg-white">
            <option v-for="h in handlers" :key="h.id" :value="h.id">{{ h.label }}</option>
          </select>
        </div>
        <!-- 股票池 -->
        <div>
          <label class="block text-xs text-slate-500 mb-1">股票池</label>
          <select v-model="cfg.universe"
            class="w-full px-3 py-2 text-sm rounded-lg border border-surface-3 focus:border-brand-500
                   focus:ring-1 focus:ring-brand-500 outline-none transition cursor-pointer bg-white">
            <option value="300">前 300 只 (快速)</option>
            <option value="500">前 500 只</option>
            <option value="1000">前 1000 只</option>
            <option value="all">全部 (~5200 只, 很慢)</option>
          </select>
        </div>
        <!-- 目标日期 -->
        <div>
          <label class="block text-xs text-slate-500 mb-1">目标日期 <span class="text-danger">*</span></label>
          <input v-model="cfg.test_date" type="date"
            class="w-full px-3 py-2 text-sm rounded-lg border border-surface-3 focus:border-brand-500
                   focus:ring-1 focus:ring-brand-500 outline-none transition">
        </div>
        <!-- 训练区间 -->
        <div>
          <label class="block text-xs text-slate-500 mb-1">训练开始</label>
          <input v-model="cfg.train_start" type="date"
            class="w-full px-3 py-2 text-sm rounded-lg border border-surface-3 focus:border-brand-500
                   focus:ring-1 focus:ring-brand-500 outline-none transition">
        </div>
        <div>
          <label class="block text-xs text-slate-500 mb-1">训练结束</label>
          <input v-model="cfg.train_end" type="date"
            class="w-full px-3 py-2 text-sm rounded-lg border border-surface-3 focus:border-brand-500
                   focus:ring-1 focus:ring-brand-500 outline-none transition">
        </div>
        <div></div>
        <!-- 验证区间 -->
        <div>
          <label class="block text-xs text-slate-500 mb-1">验证开始</label>
          <input v-model="cfg.valid_start" type="date"
            class="w-full px-3 py-2 text-sm rounded-lg border border-surface-3 focus:border-brand-500
                   focus:ring-1 focus:ring-brand-500 outline-none transition">
        </div>
        <div>
          <label class="block text-xs text-slate-500 mb-1">验证结束</label>
          <input v-model="cfg.valid_end" type="date"
            class="w-full px-3 py-2 text-sm rounded-lg border border-surface-3 focus:border-brand-500
                   focus:ring-1 focus:ring-brand-500 outline-none transition">
        </div>
        <div>
          <label class="block text-xs text-slate-500 mb-1">Top N</label>
          <input v-model.number="cfg.top_n" type="number" min="10" max="200" step="10"
            class="w-full px-3 py-2 text-sm rounded-lg border border-surface-3 focus:border-brand-500
                   focus:ring-1 focus:ring-brand-500 outline-none transition">
        </div>
      </div>

      <div class="mt-5 flex items-center gap-3">
        <button @click="startSelection" :disabled="running"
          class="flex items-center gap-2 px-6 py-2.5 bg-cta text-white font-semibold rounded-lg
                 hover:bg-amber-600 transition cursor-pointer shadow-sm
                 disabled:opacity-50 disabled:cursor-not-allowed">
          <svg :class="['w-5 h-5', running && 'animate-spin']" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round"
                  d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"/>
          </svg>
          <span>{{ running ? '运行中...' : '开始选股' }}</span>
        </button>
        <span v-if="running" class="text-xs text-slate-400">模型训练可能需要几分钟，请耐心等待</span>
      </div>
    </div>

    <!-- 进度面板 -->
    <div v-if="running || logs.length" class="bg-white rounded-xl border border-surface-3 overflow-hidden">
      <div class="px-5 py-3 border-b border-surface-3 flex items-center justify-between">
        <div class="flex items-center gap-2">
          <svg v-if="running" class="w-4 h-4 text-brand-500 animate-spin" fill="none" viewBox="0 0 24 24">
            <circle cx="12" cy="12" r="10" stroke="currentColor" stroke-width="3" class="opacity-25"/>
            <path fill="currentColor" d="M4 12a8 8 0 018-8v4a4 4 0 00-4 4H4z" class="opacity-75"/>
          </svg>
          <span class="text-sm font-medium text-slate-700">{{ stepText || '准备就绪' }}</span>
        </div>
        <button v-if="running" @click="cancelSelection"
          class="text-xs text-slate-400 hover:text-danger transition cursor-pointer">取消</button>
      </div>
      <div class="h-1.5 bg-surface-2">
        <div class="h-full bg-brand-500 rounded-r transition-all duration-300" :style="{ width: progress + '%' }"></div>
      </div>
      <div class="log-scroll bg-brand-950 text-brand-200 font-mono text-xs p-4 h-48 overflow-y-auto">
        <div v-for="(l, i) in logs" :key="i" :class="logColor[l.level] || logColor.info">{{ l.msg }}</div>
      </div>
    </div>

    <!-- 选股结果 -->
    <div v-if="results?.top?.length" class="bg-white rounded-xl border border-surface-3 overflow-hidden">
      <div class="px-5 py-3 border-b border-surface-3 flex items-center justify-between">
        <div>
          <h2 class="text-sm font-semibold text-slate-700">选股结果</h2>
          <div class="text-xs text-slate-400 mt-0.5">
            {{ models.find(m => m.id === cfg.model_id)?.label || cfg.model_id }} ·
            目标日期 {{ cfg.test_date }} ·
            共 {{ results.total }} 只股票
          </div>
        </div>
        <button @click="exportCSV"
          class="flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium text-brand-600 bg-brand-50
                 rounded-lg hover:bg-brand-100 transition cursor-pointer">
          <svg class="w-3.5 h-3.5" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" d="M12 10v6m0 0l-3-3m3 3l3-3m2 8H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"/>
          </svg>
          导出 CSV
        </button>
        <router-link to="/backtest"
          class="flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium text-white bg-brand-600
                 rounded-lg hover:bg-brand-700 transition cursor-pointer">
          <svg class="w-3.5 h-3.5" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"/>
          </svg>
          回测选股结果
        </router-link>
      </div>
      <div class="overflow-x-auto">
        <table class="w-full text-sm">
          <thead>
            <tr class="text-left text-xs text-slate-500 border-b border-surface-3">
              <th class="px-5 py-2.5 w-16">排名</th>
              <th class="py-2.5 pr-4">股票代码</th>
              <th class="py-2.5 pr-4">股票名称</th>
              <th class="py-2.5 pr-4 w-48">预测分数</th>
            </tr>
          </thead>
          <tbody class="text-slate-700">
            <tr v-for="r in results.top" :key="r.rank"
              :class="[r.rank <= 10 ? 'bg-brand-50/50' : '', 'border-b border-surface-2/60 hover:bg-surface-2/40 transition']">
              <td class="px-5 py-2.5">
                <span :class="[
                  'inline-flex items-center justify-center w-6 h-6 rounded text-xs font-bold',
                  r.rank <= 3 ? 'bg-cta/10 text-amber-600' :
                  r.rank <= 10 ? 'bg-brand-100 text-brand-600' :
                  'bg-surface-2 text-slate-500'
                ]">{{ r.rank }}</span>
              </td>
              <td class="py-2.5 pr-4 font-mono text-sm">{{ r.symbol }}</td>
              <td class="py-2.5 pr-4">{{ r.name || '--' }}</td>
              <td class="py-2.5 pr-4">
                <div class="flex items-center gap-2">
                  <div class="flex-1 h-1.5 bg-surface-2 rounded-full overflow-hidden">
                    <div class="h-full bg-brand-400 rounded-full transition-all duration-500"
                      :style="{ width: (Math.abs(r.score) / maxScore(results.top) * 100) + '%' }"></div>
                  </div>
                  <span class="font-mono text-xs text-slate-600 w-20 text-right">{{ r.score.toFixed(6) }}</span>
                </div>
              </td>
            </tr>
          </tbody>
        </table>
      </div>

      <!-- Bottom 10 -->
      <div v-if="results?.bottom?.length" class="border-t border-surface-3">
        <div class="px-5 py-2 text-xs font-medium text-slate-400 uppercase tracking-wide">Bottom 10</div>
        <table class="w-full text-sm">
          <tbody class="text-slate-500">
            <tr v-for="r in results.bottom" :key="'b'+r.rank"
              class="border-t border-surface-2/60 hover:bg-surface-2/40 transition">
              <td class="px-5 py-2 w-16 font-mono text-xs text-slate-400">{{ r.rank }}</td>
              <td class="py-2 pr-4 font-mono text-sm">{{ r.symbol }}</td>
              <td class="py-2 pr-4">{{ r.name || '--' }}</td>
              <td class="py-2 pr-4 font-mono text-xs text-red-400 w-24 text-right">{{ r.score.toFixed(6) }}</td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>

    <!-- 运行历史 -->
    <div class="bg-white rounded-xl border border-surface-3 p-5">
      <h2 class="text-sm font-semibold text-slate-500 uppercase tracking-wide mb-4">运行历史</h2>
      <div class="overflow-x-auto">
        <table class="w-full text-sm">
          <thead>
            <tr class="text-left text-xs text-slate-500 border-b border-surface-3">
              <th class="pb-2 pr-4">时间</th>
              <th class="pb-2 pr-4">模型</th>
              <th class="pb-2 pr-4">目标日期</th>
              <th class="pb-2 pr-4">Top N</th>
              <th class="pb-2 pr-4">状态</th>
              <th class="pb-2 pr-4">耗时</th>
              <th class="pb-2">备注</th>
            </tr>
          </thead>
          <tbody class="text-slate-600">
            <tr v-if="!history.length">
              <td colspan="7" class="py-6 text-center text-slate-500">暂无运行记录</td>
            </tr>
            <tr v-for="(h, i) in history" :key="i" class="border-b border-surface-2/60">
              <td class="py-2.5 pr-4 whitespace-nowrap">{{ h.time }}</td>
              <td class="py-2.5 pr-4">{{ h.model }}</td>
              <td class="py-2.5 pr-4 font-mono">{{ h.test_date }}</td>
              <td class="py-2.5 pr-4">{{ h.top_n }}</td>
              <td class="py-2.5 pr-4">
                <span :class="h.success ? 'text-emerald-500' : 'text-red-500'" class="font-medium">
                  {{ h.success ? '成功' : '失败' }}
                </span>
              </td>
              <td class="py-2.5 pr-4 font-mono">{{ h.elapsed }}</td>
              <td class="py-2.5 text-slate-500">{{ h.error || '--' }}</td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  </div>
</template>
