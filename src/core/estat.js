// Estatística descritiva. O IC95% reproduz `ic95` de barco_continuo.py (t de Student para n ≤ 11; 2,0 até n = 30; 1,96 acima).
// Descriptive statistics; the 95% CI half-width reproduces the Python `ic95`.

const T95 = { 1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228 };

export const media = (v) => (v.length ? v.reduce((a, b) => a + b, 0) / v.length : NaN);

/** Desvio-padrão amostral (ddof = 1). */
export function desvio(v) {
  if (v.length < 2) return NaN;
  const m = media(v);
  return Math.sqrt(v.reduce((a, x) => a + (x - m) ** 2, 0) / (v.length - 1));
}

/** Meia-largura do intervalo de confiança de 95% da média. */
export function ic95(v) {
  const n = v.length;
  if (n < 2) return NaN;
  const t = T95[n - 1] ?? (n > 30 ? 1.96 : 2.0);
  return (t * desvio(v)) / Math.sqrt(n);
}

export const minimo = (v) => v.reduce((a, b) => Math.min(a, b), Infinity);
export const maximo = (v) => v.reduce((a, b) => Math.max(a, b), -Infinity);

/** Resumo `{media, dp, ic, n}` de um vetor. */
export function resumir(v) {
  return { media: media(v), dp: desvio(v), ic: ic95(v), n: v.length };
}
