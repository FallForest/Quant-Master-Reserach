import { ref } from 'vue'

/**
 * Manages execution draft transfer from Strategy page to Execution page
 * via sessionStorage.
 *
 * @param {object} [opts] - Optional dependencies
 * @param {object} [opts.router] - Vue Router instance for navigation.
 *   When omitted (e.g. in unit tests), ``navigateToExecution`` is a no-op.
 */
export function useExecutionDraft(opts = {}) {
  const router = opts.router || null

  const draftMeta = ref(null)
  const draftTrades = ref([])

  function importDraft() {
    const raw = sessionStorage.getItem('executionDraft')
    if (!raw) return false
    let draft
    try {
      draft = JSON.parse(raw)
    } catch {
      sessionStorage.removeItem('executionDraft')
      return false
    }
    draftMeta.value = {
      source: draft.source || 'buffered-rebalance',
      alias: draft.alias || '--',
      tradeDate: draft.tradeDate || '--',
      summary: draft.summary || null,
      config: draft.config || null,
    }
    draftTrades.value = draft.trades || []
    return true
  }

  function saveDraft(source, alias, tradeDate, config, trades, summary) {
    sessionStorage.setItem('executionDraft', JSON.stringify({
      source, alias, tradeDate, config, trades, summary,
    }))
  }

  async function navigateToExecution() {
    if (router) await router.push('/execution')
  }

  function clearDraft() {
    sessionStorage.removeItem('executionDraft')
    draftMeta.value = null
    draftTrades.value = []
  }

  return {
    draftMeta,
    draftTrades,
    importDraft,
    saveDraft,
    navigateToExecution,
    clearDraft,
  }
}
