/**
 * Static layer definitions for model architecture diagrams.
 * Each key matches a model id from /api/model-catalog.
 * @type {Record<string, Array<{name: string, type: 'input'|'hidden'|'output', w: number}>>}
 */
export const archLayers = {}

/** Default layers for models not in archLayers. */
export function defaultLayers(modelName) {
  return [
    { name: '输入', type: 'input', w: 80 },
    { name: modelName + ' 层', type: 'hidden', w: 140 },
    { name: '输出', type: 'output', w: 80 },
  ]
}
