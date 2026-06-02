import { createRouter, createWebHistory } from 'vue-router'

const routes = [
  { path: '/', name: 'overview', component: () => import('../views/Overview.vue'), meta: { title: '总览' } },
  { path: '/browser', name: 'browser', component: () => import('../views/Browser.vue'), meta: { title: '数据浏览' } },
  { path: '/model', name: 'model', component: () => import('../views/ModelSelection.vue'), meta: { title: '模型选股' } },
  { path: '/strategy', name: 'strategy', component: () => import('../views/StrategyHub.vue'), meta: { title: '策略调仓' } },
  { path: '/strategy/buffered-rebalance', name: 'strategy-buffered-rebalance', component: () => import('../views/StrategyHub.vue'), meta: { title: '策略调仓' } },
  { path: '/strategy-buffered-rebalance', name: 'strategy-buffered-rebalance-legacy', redirect: '/strategy/buffered-rebalance' },
  { path: '/execution', name: 'execution', component: () => import('../views/Position.vue'), meta: { title: '交易执行' } },
  { path: '/position', name: 'position', redirect: '/execution' },
  { path: '/:pathMatch(.*)*', name: 'not-found', redirect: '/' },
]

export default createRouter({
  history: createWebHistory(),
  routes,
})
