// Leitura e formatação de números (pt-BR e en). Módulo puro, sem DOM. / Number parsing and formatting, DOM-free.

/** Aceita "0,25", "0.25", "1.234,5", "1,234.5", "−10", " 30 000 ". Devolve NaN se inválido. */
export function lerNumero(s) {
  if (typeof s === 'number') return s;
  let t = String(s ?? '').trim().replace(/[\s  ]/g, '').replace(/[−–]/g, '-');
  if (t === '') return NaN;
  const virg = t.lastIndexOf(','), pto = t.lastIndexOf('.');
  if (virg >= 0 && pto >= 0) {
    t = virg > pto ? t.replace(/\./g, '').replace(',', '.') : t.replace(/,/g, '');
  } else if (virg >= 0) {
    t = t.replace(',', '.');
  }
  return /^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$/.test(t) ? Number(t) : NaN;
}

const loc = (lang) => (lang === 'pt' ? 'pt-BR' : 'en-US');
/** Número com `d` casas fixas. */
export const fmtNum = (n, d = 2, lang = 'pt') =>
  Number.isFinite(Number(n)) ? Number(n).toLocaleString(loc(lang), { minimumFractionDigits: d, maximumFractionDigits: d }) : '–';
/** Número com até `max` casas. */
export const fmtMax = (n, max = 3, lang = 'pt') =>
  Number.isFinite(Number(n)) ? Number(n).toLocaleString(loc(lang), { maximumFractionDigits: max }) : '–';
/** Inteiro com separador de milhar. */
export const fmtInt = (n, lang = 'pt') =>
  Number.isFinite(Number(n)) ? Number(n).toLocaleString(loc(lang), { maximumFractionDigits: 0 }) : '–';
/** Percentual de uma fração (0.43 → "43,0%"). */
export const fmtPct = (f, d = 1, lang = 'pt') => (Number.isFinite(Number(f)) ? fmtNum(Number(f) * 100, d, lang) + '%' : '–');
