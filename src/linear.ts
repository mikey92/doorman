// A logistic regression over EmbeddingGemma vectors: 768 weights and a bias, trained in
// eval/train.ts and shipped as public/classifier.json. Small enough to read in full.

export interface Classifier {
  model: string;
  prefix: string;
  bias: number;
  weights: number[];
  /** At or above this probability, a message is folded away as a pitch. */
  pitchAt: number;
  /** Below this, it is a customer; in between, unsure. */
  customerBelow: number;
}

export function dot(a: ArrayLike<number>, b: ArrayLike<number>): number {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}

export const sigmoid = (z: number) => 1 / (1 + Math.exp(-z));

export function predict(c: Pick<Classifier, "bias" | "weights">, x: ArrayLike<number>): number {
  return sigmoid(c.bias + dot(c.weights, x));
}

/**
 * Plain batch gradient descent with L2. Pitches are y = 1. `weight` lets a class count more, so
 * that hiding a customer can be made dearer than missing a pitch.
 */
export function train(xs: ArrayLike<number>[], ys: number[], opts: { l2?: number; epochs?: number; lr?: number; customerWeight?: number } = {}) {
  const { l2 = 1e-3, epochs = 400, lr = 0.5, customerWeight = 1 } = opts;
  const d = xs[0].length;
  const w = new Float64Array(d);
  let b = 0;
  const sw = ys.map((y) => (y === 0 ? customerWeight : 1));
  const total = sw.reduce((a, x) => a + x, 0);
  for (let e = 0; e < epochs; e++) {
    const gw = new Float64Array(d);
    let gb = 0;
    for (let i = 0; i < xs.length; i++) {
      const err = (sigmoid(b + dot(w, xs[i])) - ys[i]) * sw[i];
      gb += err;
      const x = xs[i];
      for (let j = 0; j < d; j++) gw[j] += err * x[j];
    }
    for (let j = 0; j < d; j++) w[j] -= lr * (gw[j] / total + l2 * w[j]);
    b -= lr * (gb / total);
  }
  return { bias: b, weights: Array.from(w) };
}
