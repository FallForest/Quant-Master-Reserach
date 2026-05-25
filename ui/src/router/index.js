import { createRouter, createWebHistory } from 'vue-router'

const routes = [
  { path: '/',           name: 'overview', component: () => import('../views/Overview.vue'),  meta: { title: '总览' } },
  { path: '/pipeline',   name: 'pipeline', component: () => import('../views/Pipeline.vue'),  meta: { title: '数据管道' } },
  { path: '/browser',    name: 'browser',  component: () => import('../views/Browser.vue'),   meta: { title: '数据浏览' } },
  { path: '/factor',     name: 'factor',   component: () => import('../views/FactorAnalysis.vue'), meta: { title: '因子分析' } },
]

export default createRouter({
  history: createWebHistory(),
  routes,
})
