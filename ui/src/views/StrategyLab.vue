<script setup>
import { ref, computed, onMounted } from 'vue'
import { api } from '../utils/api'

const loading = ref(true)
const strategies = ref([])
const filterCategory = ref('all')
const expandedId = ref(null)
const compareMode = ref(false)
const compareList = ref([])

const categories = computed(() => {
  const cats = [...new Set(strategies.value.map(s => s.category))]
  return ['all', ...cats]
})

const filtered = computed(() => {
  if (filterCategory.value === 'all') return strategies.value
  return strategies.value.filter(s => s.category === filterCategory.value)
})

onMounted(async () => {
  loading.value = true
  const data = await api('/api/strategies')
  strategies.value = data?.strategies || []
  loading.value = false
})

function toggleExpand(id) {
  expandedId.value = expandedId.value === id ? null : id
}

function toggleCompare(id) {
  const idx = compareList.value.indexOf(id)
  if (idx >= 0) compareList.value.splice(idx, 1)
  else if (compareList.value.length < 3) compareList.value.push(id)
}

const compareItems = computed(() =>
  strategies.value.filter(s => compareList.value.includes(s.id))
)

const categoryIcon = {
  '选股': 'M3 3v18h18M9 17V9m4 8V5m4 12v-4',
  '执行': 'M13 10V3L4 14h7v7l9-11h-7z',
  '增强': 'M13 7h8m0 0v8m0-8l-8 8-4-4-6 6',
}
</script>

<template>
  <div class="p-4 sm:p-6 space-y-5 animate-slide-in">

    <!-- 标题 + 筛选 -->
    <div class="flex flex-wrap items-center gap-3">
      <div class="flex items-center gap-2">
        <svg class="w-5 h-5 text-brand-500" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" d="M9.813 15.904L9 18.75l-.813-2.846a4.5 4.5 0 00-3.09-3.09L2.25 12l2.846-.813a4.5 4.5 0 003.09-3.09L9 5.25l.813 2.846a4.5 4.5 0 003.09 3.09L15.75 12l-2.846.813a4.5 4.5 0 00-3.09 3.09zM18.259 8.715L18 9.75l-.259-1.035a3.375 3.375 0 00-2.455-2.456L14.25 6l1.036-.259a3.375 3.375 0 002.455-2.456L18 2.25l.259 1.035a3.375 3.375 0 002.455 2.456L21.75 6l-1.036.259a3.375 3.375 0 00-2.455 2.456z"/>
        </svg>
        <h2 class="text-base font-semibold text-slate-700">策略工坊</h2>
      </div>
      <div class="flex-1"></div>
      <div class="flex items-center gap-2">
        <button @click="compareMode = !compareMode; if (!compareMode) compareList = []"
          :class="['px-3 py-1.5 text-xs font-semibold rounded-lg cursor-pointer transition-all',
                   compareMode ? 'bg-brand-600 text-white' : 'bg-surface-2 text-slate-500 hover:bg-surface-3']">
          {{ compareMode ? '退出对比' : '策略对比' }}
        </button>
        <div class="flex items-center bg-surface-2 rounded-xl p-0.5 gap-0.5">
          <button v-for="c in categories" :key="c" @click="filterCategory = c"
            :class="['px-3 py-1.5 text-xs font-semibold rounded-lg cursor-pointer transition-all duration-200',
                     filterCategory === c ? 'bg-white text-brand-600 shadow-sm' : 'text-slate-500 hover:text-slate-700']">
            {{ c === 'all' ? '全部' : c }}
          </button>
        </div>
      </div>
    </div>

    <!-- 对比面板 -->
    <div v-if="compareMode && compareItems.length >= 2" class="bg-white rounded-xl border border-surface-3 p-4">
      <h3 class="text-sm font-semibold text-slate-600 mb-3">策略对比 ({{ compareItems.length }}/3)</h3>
      <div class="overflow-x-auto">
        <table class="w-full text-sm">
          <thead>
            <tr class="text-left text-[11px] text-slate-500 border-b border-surface-3">
              <th class="px-4 py-2">策略</th>
              <th class="px-4 py-2 text-right">平均收益</th>
              <th class="px-4 py-2 text-right">夏普比率</th>
              <th class="px-4 py-2 text-right">最大回撤</th>
              <th class="px-4 py-2 text-right">换手率</th>
            </tr>
          </thead>
          <tbody class="text-slate-700">
            <tr v-for="s in compareItems" :key="s.id" class="border-b border-surface-2/60">
              <td class="px-4 py-2 font-medium">{{ s.name }}</td>
              <td class="px-4 py-2 text-right font-mono" :class="s.performance?.avgReturn >= 0 ? 'text-bull' : 'text-bear'">
                {{ s.performance?.avgReturn != null ? (s.performance.avgReturn >= 0 ? '+' : '') + s.performance.avgReturn + '%' : '-' }}
              </td>
              <td class="px-4 py-2 text-right font-mono text-brand-600">{{ s.performance?.sharpe ?? '-' }}</td>
              <td class="px-4 py-2 text-right font-mono text-danger">{{ s.performance?.maxDD != null ? s.performance.maxDD + '%' : '-' }}</td>
              <td class="px-4 py-2 text-right font-mono text-slate-500">{{ s.performance?.turnover != null ? s.performance.turnover.toFixed(2) : '-' }}</td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>

    <!-- 骨架屏 -->
    <div v-if="loading" class="space-y-3">
      <div v-for="i in 4" :key="i" class="bg-white rounded-xl border border-surface-3 p-4">
        <div class="flex items-center gap-3">
          <div class="skeleton h-5 w-5 rounded"></div>
          <div class="skeleton h-4 w-32"></div>
          <div class="skeleton h-5 w-12 rounded-full"></div>
          <div class="flex-1"></div>
          <div class="skeleton h-3 w-20"></div>
        </div>
      </div>
    </div>

    <!-- 策略列表 -->
    <div v-else class="space-y-3">
      <div v-for="s in filtered" :key="s.id"
           :class="['bg-white rounded-xl border border-surface-3 overflow-hidden transition-all duration-200',
                    expandedId === s.id ? 'ring-2 ring-brand-400 shadow-sm' : 'hover:shadow-sm hover:border-brand-200']">
        <div class="px-5 py-3.5 flex items-center gap-3 cursor-pointer" @click="toggleExpand(s.id)">
          <!-- 对比复选框 -->
          <label v-if="compareMode" class="flex items-center cursor-pointer" @click.stop>
            <input type="checkbox" :checked="compareList.includes(s.id)" @change="toggleCompare(s.id)"
              class="w-4 h-4 rounded border-surface-3 text-brand-600 focus:ring-brand-500 cursor-pointer">
          </label>
          <svg v-else class="w-5 h-5 text-brand-500 flex-shrink-0" fill="none" stroke="currentColor" stroke-width="1.5" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" :d="categoryIcon[s.category] || 'M9.813 15.904L9 18.75l-.813-2.846a4.5 4.5 0 00-3.09-3.09L2.25 12l2.846-.813a4.5 4.5 0 003.09-3.09L9 5.25l.813 2.846a4.5 4.5 0 003.09 3.09L15.75 12l-2.846.813a4.5 4.5 0 00-3.09 3.09z'"/>
          </svg>
          <span class="font-semibold text-sm text-slate-700">{{ s.name }}</span>
          <span class="text-[10px] px-2 py-0.5 rounded-full font-medium bg-brand-50 text-brand-600">{{ s.category }}</span>
          <div class="flex-1"></div>
          <!-- 绩效摘要 -->
          <div v-if="s.performance?.sharpe" class="hidden sm:flex items-center gap-4 text-xs text-slate-400 font-mono">
            <span>收益 <span :class="s.performance.avgReturn >= 0 ? 'text-bull' : 'text-bear'">{{ s.performance.avgReturn >= 0 ? '+' : '' }}{{ s.performance.avgReturn }}%</span></span>
            <span>夏普 <span class="text-brand-600">{{ s.performance.sharpe }}</span></span>
          </div>
          <svg :class="['w-4 h-4 text-slate-300 transition-transform', expandedId === s.id ? 'rotate-180' : '']"
               fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" d="M19 9l-7 7-7-7"/>
          </svg>
        </div>

        <!-- 展开详情 -->
        <div v-if="expandedId === s.id" class="px-5 pb-4 border-t border-surface-3">
          <p class="text-xs text-slate-500 mt-3 mb-3 leading-relaxed">{{ s.desc }}</p>
          <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
            <!-- 参数配置 -->
            <div>
              <h4 class="text-xs font-semibold text-slate-500 uppercase tracking-wide mb-2">参数配置</h4>
              <div class="bg-surface-2/50 rounded-lg p-3 space-y-2">
                <div v-for="(p, k) in s.params" :key="k" class="flex items-center justify-between text-xs">
                  <span class="text-slate-500">{{ p.label }}</span>
                  <div class="flex items-center gap-2">
                    <span v-if="p.type === 'number'" class="text-slate-700 font-mono">
                      {{ p.default }}{{ p.min != null ? ` [${p.min}~${p.max}]` : '' }}
                    </span>
                    <span v-else-if="p.type === 'select'" class="text-slate-700 font-mono">
                      {{ p.options.join(' / ') }}
                    </span>
                    <span v-else class="text-slate-700 font-mono">{{ p.default }}</span>
                  </div>
                </div>
              </div>
            </div>
            <!-- 绩效指标 -->
            <div v-if="s.performance?.sharpe">
              <h4 class="text-xs font-semibold text-slate-500 uppercase tracking-wide mb-2">回测绩效</h4>
              <div class="grid grid-cols-2 gap-2">
                <div class="bg-surface-2/50 rounded-lg p-2.5 text-center">
                  <div class="text-[10px] text-slate-500">平均收益</div>
                  <div :class="['text-sm font-bold font-mono', s.performance.avgReturn >= 0 ? 'text-bull' : 'text-bear']">
                    {{ s.performance.avgReturn >= 0 ? '+' : '' }}{{ s.performance.avgReturn }}%
                  </div>
                </div>
                <div class="bg-surface-2/50 rounded-lg p-2.5 text-center">
                  <div class="text-[10px] text-slate-500">夏普比率</div>
                  <div class="text-sm font-bold font-mono text-brand-600">{{ s.performance.sharpe }}</div>
                </div>
                <div class="bg-surface-2/50 rounded-lg p-2.5 text-center">
                  <div class="text-[10px] text-slate-500">最大回撤</div>
                  <div class="text-sm font-bold font-mono text-danger">{{ s.performance.maxDD }}%</div>
                </div>
                <div class="bg-surface-2/50 rounded-lg p-2.5 text-center">
                  <div class="text-[10px] text-slate-500">换手率</div>
                  <div class="text-sm font-bold font-mono text-slate-600">{{ s.performance.turnover.toFixed(2) }}</div>
                </div>
              </div>
            </div>
          </div>
          <div class="mt-3">
            <router-link :to="{ path: '/backtest', query: { strategy: s.id } }"
              class="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium text-white bg-brand-600 rounded-lg hover:bg-brand-700 transition cursor-pointer">
              <svg class="w-3.5 h-3.5" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"/>
              </svg>
              用此策略回测
            </router-link>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>
