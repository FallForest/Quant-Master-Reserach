import { createRouter, createWebHistory } from 'vue-router'

const routes = [
  { path: '/',           name: 'overview', component: () => import('../views/Overview.vue'),  meta: { title: '总览' } },
  { path: '/pipeline',   name: 'pipeline', component: () => import('../views/Pipeline.vue'),  meta: { title: '数据管道' } },
  { path: '/browser',    name: 'browser',  component: () => import('../views/Browser.vue'),   meta: { title: '数据浏览' } },
  { path: '/factor',     name: 'factor',   component: () => import('../views/FactorAnalysis.vue'), meta: { title: '因子分析' } },
  { path: '/stock-select', name: 'stock-select', component: () => import('../views/StockSelection.vue'), meta: { title: '模型选股' } },
  { path: '/backtest', name: 'backtest', component: () => import('../views/Backtest.vue'), meta: { title: '策略回测' } },
  { path: '/experiments', name: 'experiments', component: () => import('../views/Experiments.vue'), meta: { title: '实验管理' } },
  { path: '/portfolio', name: 'portfolio', component: () => import('../views/Portfolio.vue'), meta: { title: '持仓分析' } },
  { path: '/model-performance', name: 'model-performance', component: () => import('../views/ModelPerformance.vue'), meta: { title: '模型绩效' } },
  { path: '/strategy-lab', name: 'strategy-lab', component: () => import('../views/StrategyLab.vue'), meta: { title: '策略工坊' } },
  { path: '/model-lab', name: 'model-lab', component: () => import('../views/ModelLab.vue'), meta: { title: '模型工坊' } },
  { path: '/optimizer', name: 'optimizer', component: () => import('../views/Optimizer.vue'), meta: { title: '组合优化' } },
  { path: '/attribution', name: 'attribution', component: () => import('../views/Attribution.vue'), meta: { title: '收益归因' } },
  { path: '/:pathMatch(.*)*', name: 'not-found', redirect: '/' },
]

export default createRouter({
  history: createWebHistory(),
  routes,
})
