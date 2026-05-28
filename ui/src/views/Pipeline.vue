<script setup>
import { ref, computed, watch, onMounted, onUnmounted } from 'vue'
import { api } from '../utils/api'
import { useToast } from '../utils/toast'

const toast = useToast()

const running = ref(false)
const progress = ref(0)
const stepText = ref('')
const logs = ref([])
const lastUpdate = ref('--')

// 高级设置
const cfg = ref({
  source: 'yahoo',
  region: 'cn',
  data_dir: '~/.quant_master/quant_master_data/cn_data',
  end_date: '',
  max_workers: 8,
  delay: 0.1,
  exists_skip: true,
})

const sources = [
  { id: 'yahoo', label: 'Yahoo Finance', desc: '日线数据，支持 A 股/美股/巴西' },
  { id: 'tdx', label: '通达信 (TDX)', desc: '日线数据，A 股，直连行情服务器' },
]
const regions = [
  { id: 'cn', label: '中国 A 股' },
  { id: 'us', label: '美股' },
  { id: 'br', label: '巴西' },
]

const sourceRegions = {
  yahoo: ['cn', 'us', 'br'],
  tdx: ['cn'],
}
const availableRegions = computed(() => {
  const allowed = sourceRegions[cfg.value.source] || ['cn']
  return regions.filter(r => allowed.includes(r.id))
})
watch(() => cfg.value.source, () => {
  const allowed = sourceRegions[cfg.value.source] || ['cn']
  if (!allowed.includes(cfg.value.region)) {
    cfg.value.region = allowed[0]
  }
})

let pollTimer = null
let runStartTime = 0

// 运行历史
const history = ref([])
function loadHistory() {
  try { history.value = JSON.parse(localStorage.getItem('pipeline_history') || '[]') } catch { history.value = [] }
}
function saveHistory() {
  localStorage.setItem('pipeline_history', JSON.stringify(history.value.slice(0, 50)))
}
function recordRun(success, error) {
  const elapsed = runStartTime ? ((Date.now() - runStartTime) / 1000).toFixed(1) + 's' : '--'
  history.value.unshift({
    time: new Date().toLocaleString('zh-CN', { hour12: false }),
    source: cfg.value.source,
    region: cfg.value.region,
    success,
    elapsed,
    error: error || '',
  })
  saveHistory()
}

onMounted(async () => {
  loadHistory()
  const data = await api('/api/pipeline/status')
  if (data?.lastUpdate) lastUpdate.value = data.lastUpdate
})

onUnmounted(() => {
  if (pollTimer) clearInterval(pollTimer)
})

function appendLog(level, msg) {
  const time = new Date().toLocaleTimeString('zh-CN', { hour12: false })
  logs.value.push({ level, msg: `[${time}] ${msg}` })
}

function setProgress(pct) { progress.value = pct }

async function startUpdate() {
  running.value = true
  progress.value = 0
  logs.value = []
  runStartTime = Date.now()
  appendLog('info', '正在启动数据更新...')
  setProgress(5)

  const result = await api('/api/pipeline/run', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(cfg.value),
  })

  if (result?.runId) {
    pollRun(result.runId)
  } else {
    appendLog('error', '启动失败: ' + (result?.error || '无法连接后端'))
    simulateRun()
  }
}

function cancelUpdate() {
  if (pollTimer) clearInterval(pollTimer)
  appendLog('warn', '用户取消')
  running.value = false
}

function pollRun(runId) {
  pollTimer = setInterval(async () => {
    const status = await api(`/api/pipeline/status/${runId}`)
    if (!status) return
    if (status.logs) status.logs.forEach(l => appendLog(l.level, l.msg))
    if (status.progress != null) setProgress(status.progress)
    if (status.step) stepText.value = status.step
    if (status.done) {
      clearInterval(pollTimer)
      setProgress(100)
      appendLog(status.success ? 'success' : 'error',
        status.success ? '更新完成!' : '更新失败: ' + (status.error || '未知错误'))
      if (status.success) toast.success('数据更新完成')
      else toast.error('更新失败: ' + (status.error || '未知错误'))
      recordRun(status.success, status.error)
      running.value = false
    }
  }, 2000)
}

function simulateRun() {
  const steps = [
    { pct: 10, step: '清理残留文件', logs: [{ level: 'info', msg: '清理 source/ 和 normalize/ 目录' }] },
    { pct: 20, step: '下载数据', logs: [{ level: 'info', msg: '从 Yahoo Finance 下载日线数据...' }] },
    { pct: 50, step: '标准化处理', logs: [{ level: 'info', msg: '对齐交易日历，计算复权因子' }] },
    { pct: 70, step: '写入二进制', logs: [{ level: 'info', msg: '增量追加到 features/' }] },
    { pct: 85, step: '校验数据', logs: [{ level: 'info', msg: '检查日历、instruments 完整性' }] },
    { pct: 95, step: '清理中间文件', logs: [{ level: 'info', msg: '删除临时 CSV 文件' }] },
    { pct: 100, step: '完成', logs: [{ level: 'success', msg: '数据更新完成!' }] },
  ]
  let i = 0
  pollTimer = setInterval(() => {
    if (i >= steps.length) { clearInterval(pollTimer); recordRun(true); running.value = false; toast.success('数据更新完成'); return }
    const s = steps[i]
    setProgress(s.pct)
    stepText.value = s.step
    s.logs.forEach(l => appendLog(l.level, l.msg))
    i++
  }, 1500)
}

const logColor = {
  info: 'text-brand-200',
  warn: 'text-amber-400',
  error: 'text-red-400',
  success: 'text-emerald-400',
}
</script>

<template>
  <div class="p-4 sm:p-6 space-y-6 animate-slide-in">

    <!-- 状态卡片 -->
    <div class="bg-white rounded-xl border border-surface-3 p-5">
      <div class="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
        <div>
          <div class="text-xs text-slate-500 mb-1">数据状态</div>
          <div class="flex items-center gap-2">
            <span class="inline-block w-2.5 h-2.5 rounded-full bg-success pulse-dot"></span>
            <span class="text-lg font-semibold text-slate-800">数据就绪</span>
          </div>
          <div class="text-sm text-slate-500 mt-1">最后更新: {{ lastUpdate }}</div>
        </div>
        <button @click="startUpdate" :disabled="running"
          class="flex items-center gap-2 px-6 py-2.5 bg-cta text-white font-semibold rounded-lg
                 hover:bg-amber-600 transition cursor-pointer shadow-sm
                 disabled:opacity-50 disabled:cursor-not-allowed">
          <svg :class="['w-5 h-5', running && 'animate-spin']" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round"
                  d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"/>
          </svg>
          <span>{{ running ? '更新中...' : '一键更新' }}</span>
        </button>
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
        <button v-if="running" @click="cancelUpdate"
          class="text-xs text-slate-400 hover:text-danger transition cursor-pointer">取消</button>
      </div>
      <div class="h-1.5 bg-surface-2">
        <div class="h-full bg-brand-500 rounded-r transition-all duration-300" :style="{ width: progress + '%' }"></div>
      </div>
      <div ref="logEl" class="log-scroll bg-brand-950 text-brand-200 font-mono text-xs p-4 h-48 overflow-y-auto">
        <div v-for="(l, i) in logs" :key="i" :class="logColor[l.level] || logColor.info">{{ l.msg }}</div>
      </div>
    </div>

    <!-- 高级设置 -->
    <details class="bg-white rounded-xl border border-surface-3 group">
      <summary class="px-5 py-3 cursor-pointer text-sm font-medium text-slate-600
                      hover:text-brand-600 transition select-none flex items-center gap-2">
        <svg class="w-4 h-4 text-slate-400 transition-transform group-open:rotate-90" fill="none"
             stroke="currentColor" stroke-width="2" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" d="M9 5l7 7-7 7"/>
        </svg>
        高级设置
      </summary>
      <div class="px-5 pb-5 pt-2 border-t border-surface-3 space-y-4">
        <div class="grid grid-cols-1 sm:grid-cols-2 gap-4">
          <!-- 数据源 -->
          <div>
            <label class="block text-xs text-slate-500 mb-1">数据源</label>
            <select v-model="cfg.source"
              class="w-full px-3 py-2 text-sm rounded-lg border border-surface-3 focus:border-brand-500
                     focus:ring-1 focus:ring-brand-500 outline-none transition cursor-pointer bg-white">
              <option v-for="s in sources" :key="s.id" :value="s.id">{{ s.label }}</option>
            </select>
            <div class="text-xs text-slate-400 mt-1">{{ sources.find(s => s.id === cfg.source)?.desc }}</div>
          </div>
          <!-- 地区 -->
          <div>
            <label class="block text-xs text-slate-500 mb-1">地区</label>
            <select v-model="cfg.region"
              class="w-full px-3 py-2 text-sm rounded-lg border border-surface-3 focus:border-brand-500
                     focus:ring-1 focus:ring-brand-500 outline-none transition cursor-pointer bg-white">
              <option v-for="r in availableRegions" :key="r.id" :value="r.id">{{ r.label }}</option>
            </select>
          </div>
          <div>
            <label class="block text-xs text-slate-500 mb-1">数据目录 <span class="text-danger">*</span></label>
            <input v-model="cfg.data_dir" type="text"
              class="w-full px-3 py-2 text-sm rounded-lg border border-surface-3 focus:border-brand-500
                     focus:ring-1 focus:ring-brand-500 outline-none transition font-mono">
          </div>
          <div>
            <label class="block text-xs text-slate-500 mb-1">截止日期</label>
            <input v-model="cfg.end_date" type="date"
              class="w-full px-3 py-2 text-sm rounded-lg border border-surface-3 focus:border-brand-500
                     focus:ring-1 focus:ring-brand-500 outline-none transition">
          </div>
          <div>
            <label class="block text-xs text-slate-500 mb-1">并发线程数</label>
            <input v-model.number="cfg.max_workers" type="range" min="1" max="16" class="w-full accent-brand-500">
            <div class="text-xs text-slate-400 text-right">{{ cfg.max_workers }} 线程</div>
          </div>
          <div>
            <label class="block text-xs text-slate-500 mb-1">请求延迟 (秒)</label>
            <input v-model.number="cfg.delay" type="number" min="0" max="5" step="0.05"
              class="w-full px-3 py-2 text-sm rounded-lg border border-surface-3 focus:border-brand-500
                     focus:ring-1 focus:ring-brand-500 outline-none transition">
          </div>
        </div>
        <label class="flex items-center gap-2 text-sm text-slate-600 cursor-pointer">
          <input v-model="cfg.exists_skip" type="checkbox" class="accent-brand-500">
          数据集已存在时跳过初始下载
        </label>
      </div>
    </details>

    <!-- 运行历史 -->
    <div class="bg-white rounded-xl border border-surface-3 p-5">
      <h2 class="text-sm font-semibold text-slate-500 uppercase tracking-wide mb-4">运行历史</h2>
      <div class="overflow-x-auto">
        <table class="w-full text-sm">
          <thead>
            <tr class="text-left text-xs text-slate-500 border-b border-surface-3">
              <th class="pb-2 pr-4">时间</th>
              <th class="pb-2 pr-4">数据源</th>
              <th class="pb-2 pr-4">状态</th>
              <th class="pb-2 pr-4">耗时</th>
              <th class="pb-2">备注</th>
            </tr>
          </thead>
          <tbody class="text-slate-600">
            <tr v-if="!history.length">
              <td colspan="5" class="py-6 text-center text-slate-500">暂无运行记录</td>
            </tr>
            <tr v-for="(h, i) in history" :key="i" class="border-b border-surface-2/60">
              <td class="py-2.5 pr-4 whitespace-nowrap">{{ h.time }}</td>
              <td class="py-2.5 pr-4">{{ sources.find(s => s.id === h.source)?.label || h.source }}</td>
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
