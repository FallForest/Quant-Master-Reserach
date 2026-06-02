<script setup>
import { ref, onMounted, onUnmounted } from 'vue'
import { api } from '../utils/api'
import { useToast } from '../utils/toast'

const toast = useToast()

const running = ref(false)
const progress = ref(0)
const stepText = ref('')
const logs = ref([])
const lastUpdate = ref('--')
const calendarLastDate = ref('--')
const coverageText = ref('--')
const syncError = ref('')
const syncStats = ref(null)

const cfg = ref({
  data_dir: '~/.quant_master/quant_master_data/tdx_cn_data',
})

let pollTimer = null
let runStartTime = 0

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
    success,
    elapsed,
    error: error || '',
  })
  saveHistory()
}

function formatCoverage(covered, total) {
  if (!total) return '--'
  return `${covered}/${total} (${((covered / total) * 100).toFixed(1)}%)`
}

function applyStatus(data) {
  if (!data) return
  if (data.lastUpdate) lastUpdate.value = data.lastUpdate
  if (data.calendarLastDate) calendarLastDate.value = data.calendarLastDate
  coverageText.value = formatCoverage(data.equityCoveredAtLastDate, data.equityCount)
  syncError.value = data.syncError || ''
  syncStats.value = data.syncStats || null
  if (data?.dataDir) cfg.value.data_dir = data.dataDir
}

onMounted(async () => {
  loadHistory()
  const data = await api('/api/pipeline/status')
  applyStatus(data)
})

onUnmounted(() => {
  if (pollTimer) clearInterval(pollTimer)
})

function appendLog(level, msg) {
  const time = new Date().toLocaleTimeString('zh-CN', { hour12: false })
  logs.value.push({ level, msg: `[${time}] ${msg}` })
}

function setProgress(pct) { progress.value = pct }

function failRun(message) {
  appendLog('error', message)
  toast.error(message)
  recordRun(false, message)
  running.value = false
}

async function startUpdate() {
  running.value = true
  progress.value = 0
  logs.value = []
  runStartTime = Date.now()
  appendLog('info', 'Starting pipeline update...')
  setProgress(5)

  const result = await api('/api/pipeline/run', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ data_dir: cfg.value.data_dir }),
  })

  applyStatus(result)
  if (result?.runId) {
    pollRun(result.runId)
  } else {
    appendLog('error', 'Startup failed: ' + (result?.error || 'backend unavailable'))
    failRun('Startup failed: ' + (result?.error || 'backend unavailable'))
  }
}

function cancelUpdate() {
  if (pollTimer) clearInterval(pollTimer)
  appendLog('warn', 'Canceled by user')
  running.value = false
}

function pollRun(runId) {
  pollTimer = setInterval(async () => {
    const status = await api(`/api/pipeline/status/${runId}`)
    if (!status) return
    applyStatus(status)
    if (status.logs) status.logs.forEach(l => appendLog(l.level, l.msg))
    if (status.progress != null) setProgress(status.progress)
    if (status.step) stepText.value = status.step
    if (status.done) {
      clearInterval(pollTimer)
      setProgress(100)
      appendLog(
        status.success ? 'success' : 'error',
        status.success ? 'Update completed!' : 'Update failed: ' + (status.error || 'unknown error'),
      )
      if (status.success) toast.success('Pipeline update completed')
      else toast.error('Update failed: ' + (status.error || 'unknown error'))
      recordRun(status.success, status.error)
      running.value = false
    }
  }, 2000)
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
    <div class="bg-white rounded-xl border border-surface-3 p-5">
      <div class="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
        <div>
          <div class="text-xs text-slate-500 mb-1" data-testid="pipeline-status-label">Data status</div>
          <div class="flex items-center gap-2">
            <span class="inline-block w-2.5 h-2.5 rounded-full bg-success pulse-dot"></span>
            <span class="text-lg font-semibold text-slate-800" data-testid="pipeline-status-value">Ready</span>
          </div>
          <div class="text-sm text-slate-500 mt-1" data-testid="pipeline-last-update">Effective last update: {{ lastUpdate }}</div>
          <div class="text-sm text-slate-500 mt-1">Calendar last date: {{ calendarLastDate }}</div>
          <div class="text-sm text-slate-500 mt-1">Equity coverage at latest date: {{ coverageText }}</div>
          <div v-if="syncError" class="text-sm text-red-500 mt-1">{{ syncError }}</div>
        </div>
        <button
          @click="startUpdate"
          :disabled="running"
          data-testid="pipeline-run-button"
          class="flex items-center gap-2 px-6 py-2.5 bg-cta text-white font-semibold rounded-lg hover:bg-amber-600 transition cursor-pointer shadow-sm disabled:opacity-50 disabled:cursor-not-allowed"
        >
          <svg :class="['w-5 h-5', running && 'animate-spin']" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
          </svg>
          <span>{{ running ? 'Updating...' : 'Run update' }}</span>
        </button>
      </div>
    </div>

    <div v-if="running || logs.length" class="bg-white rounded-xl border border-surface-3 overflow-hidden">
      <div class="px-5 py-3 border-b border-surface-3 flex items-center justify-between">
        <div class="flex items-center gap-2">
          <svg v-if="running" class="w-4 h-4 text-brand-500 animate-spin" fill="none" viewBox="0 0 24 24">
            <circle cx="12" cy="12" r="10" stroke="currentColor" stroke-width="3" class="opacity-25" />
            <path fill="currentColor" d="M4 12a8 8 0 018-8v4a4 4 0 00-4 4H4z" class="opacity-75" />
          </svg>
          <span class="text-sm font-medium text-slate-700">{{ stepText || 'Ready to run' }}</span>
        </div>
        <button v-if="running" @click="cancelUpdate" class="text-xs text-slate-400 hover:text-danger transition cursor-pointer">Cancel</button>
      </div>
      <div class="h-1.5 bg-surface-2">
        <div class="h-full bg-brand-500 rounded-r transition-all duration-300" :style="{ width: progress + '%' }"></div>
      </div>
      <div v-if="syncStats" class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3 px-5 py-4 border-b border-surface-3 bg-surface-1/60 text-sm">
        <div>
          <div class="text-slate-500">Target sync date</div>
          <div class="font-medium text-slate-700">{{ syncStats.targetSyncDate || '--' }}</div>
        </div>
        <div>
          <div class="text-slate-500">Accepted dates</div>
          <div class="font-medium text-slate-700">{{ syncStats.acceptedNewDates?.join(', ') || '--' }}</div>
        </div>
        <div>
          <div class="text-slate-500">Rejected dates</div>
          <div class="font-medium text-slate-700">{{ syncStats.rejectedNewDates?.join(', ') || '--' }}</div>
        </div>
        <div>
          <div class="text-slate-500">Updated symbols</div>
          <div class="font-medium text-slate-700">{{ syncStats.updatedSymbols ?? '--' }}</div>
        </div>
      </div>
      <div class="log-scroll bg-brand-950 text-brand-200 font-mono text-xs p-4 h-48 overflow-y-auto">
        <div v-for="(l, i) in logs" :key="i" :class="logColor[l.level] || logColor.info">{{ l.msg }}</div>
      </div>
    </div>

    <details class="bg-white rounded-xl border border-surface-3 group">
      <summary data-testid="pipeline-advanced-settings" class="px-5 py-3 cursor-pointer text-sm font-medium text-slate-600 hover:text-brand-600 transition select-none flex items-center gap-2">
        <svg class="w-4 h-4 text-slate-400 transition-transform group-open:rotate-90" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" d="M9 5l7 7-7 7" />
        </svg>
        Advanced settings
      </summary>
      <div class="px-5 pb-5 pt-2 border-t border-surface-3 space-y-4">
        <div class="grid grid-cols-1 gap-4">
          <div>
            <label class="block text-xs text-slate-500 mb-1">Data directory <span class="text-danger">*</span></label>
            <input
              v-model="cfg.data_dir"
              type="text"
              data-testid="pipeline-data-dir-input"
              class="w-full px-3 py-2 text-sm rounded-lg border border-surface-3 focus:border-brand-500 focus:ring-1 focus:ring-brand-500 outline-none transition font-mono"
            >
          </div>
        </div>
      </div>
    </details>

    <div class="bg-white rounded-xl border border-surface-3 p-5">
      <h2 class="text-sm font-semibold text-slate-500 uppercase tracking-wide mb-4" data-testid="pipeline-history-title">Run history</h2>
      <div class="overflow-x-auto">
        <table class="w-full text-sm">
          <thead>
            <tr class="text-left text-xs text-slate-500 border-b border-surface-3">
              <th class="pb-2 pr-4">Time</th>
              <th class="pb-2 pr-4">Status</th>
              <th class="pb-2 pr-4">Elapsed</th>
              <th class="pb-2">Notes</th>
            </tr>
          </thead>
          <tbody class="text-slate-600">
            <tr v-if="!history.length">
              <td colspan="4" class="py-6 text-center text-slate-500" data-testid="pipeline-empty-history">No runs yet</td>
            </tr>
            <tr v-for="(h, i) in history" :key="i" class="border-b border-surface-2/60">
              <td class="py-2.5 pr-4 whitespace-nowrap">{{ h.time }}</td>
              <td class="py-2.5 pr-4">
                <span :class="h.success ? 'text-emerald-500' : 'text-red-500'" class="font-medium">
                  {{ h.success ? 'Success' : 'Failed' }}
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
