// Gráficos em canvas, sem dependências: linhas com faixas (IC, mínimo/máximo), linhas de referência, barras horizontais com barras de erro.
// Dependency-free canvas charts: lines with bands, reference lines, horizontal bars with error bars. Theme-aware, HiDPI, with tooltips.
import { css, prepararCanvas, aoRedimensionar, el } from './util.js';
import { fmtM, getLang } from './i18n.js';

/** Marcações "agradáveis" entre `lo` e `hi` (cerca de `n` intervalos). */
export function ticks(lo, hi, n = 5) {
  if (!(hi > lo)) { hi = lo + 1; }
  const bruto = (hi - lo) / n, mag = 10 ** Math.floor(Math.log10(bruto)), r = bruto / mag;
  const passo = (r < 1.5 ? 1 : r < 3 ? 2 : r < 7 ? 5 : 10) * mag;
  const ini = Math.ceil(lo / passo - 1e-9) * passo, out = [];
  for (let v = ini; v <= hi + passo * 1e-9; v += passo) out.push(Math.abs(v) < passo * 1e-9 ? 0 : +v.toPrecision(12));
  return out;
}
const abrevia = (v) => (Math.abs(v) >= 1e6 ? fmtM(v / 1e6, 2) + ' M' : Math.abs(v) >= 1e3 ? fmtM(v / 1e3, 1) + (getLang() === 'en' ? 'k' : ' mil') : fmtM(v, 3));

function dica(canvas) {
  const pai = canvas.parentElement;
  let d = pai.querySelector(':scope > .dica-g');
  if (!d) { d = el('div', { class: 'dica-g', role: 'presentation' }); d.hidden = true; pai.append(d); if (getComputedStyle(pai).position === 'static') pai.style.position = 'relative'; }
  return d;
}

/**
 * cfg: { series:[{nome, cor, x[], y[], lo?[], hi?[], dash?, largura?, pontos?}], refs:[{y, nome, cor, dash}],
 *        xRot, yRot, xFmt, yFmt, yMin, yMax, xMin, xMax }
 */
export function graficoLinhas(canvas, cfg) {
  canvas.__cfg = cfg;
  if (!canvas.__pronto) {
    canvas.__pronto = true;
    aoRedimensionar(canvas, () => graficoLinhas(canvas, canvas.__cfg));
    const d = dica(canvas);
    canvas.addEventListener('pointermove', (ev) => {
      const m = canvas.__modelo; if (!m) return;
      const r = canvas.getBoundingClientRect(), px = ev.clientX - r.left;
      if (px < m.L || px > m.w - m.R) { d.hidden = true; canvas.__guia = null; return desenhar(); }
      const xv = m.x0 + ((px - m.L) / (m.w - m.L - m.R)) * (m.x1 - m.x0);
      let melhor = null;
      for (const s of m.cfg.series) if (s.dica !== false) for (const x of s.x) if (melhor === null || Math.abs(x - xv) < Math.abs(melhor - xv)) melhor = x;
      if (melhor === null) return;
      const linhas = [];
      for (const s of m.cfg.series) { if (s.dica === false) continue; const i = s.x.indexOf(melhor); if (i >= 0 && Number.isFinite(s.y[i])) linhas.push(`<div><i style="background:${s.cor}"></i>${s.nome}: <b style="display:inline">${(m.cfg.yFmt ?? abrevia)(s.y[i])}</b></div>`); }
      d.innerHTML = `<b>${(m.cfg.xFmt ?? abrevia)(melhor)}</b>${linhas.join('')}`;
      d.hidden = false;
      const gx = m.L + ((melhor - m.x0) / (m.x1 - m.x0)) * (m.w - m.L - m.R);
      canvas.__guia = gx;
      const largura = d.offsetWidth;
      d.style.left = `${Math.min(Math.max(4, gx + 10), r.width - largura - 4)}px`;
      d.style.top = `${canvas.offsetTop + 8}px`;
      desenhar();
    });
    canvas.addEventListener('pointerleave', () => { d.hidden = true; canvas.__guia = null; desenhar(); });
  }
  const desenhar = () => desenharLinhas(canvas);
  desenhar();
}

function desenharLinhas(canvas) {
  const cfg = canvas.__cfg, { w, h, ctx } = prepararCanvas(canvas);
  const L = 54, R = 26, T = 10, B = 34;
  const cinza = css('--g-texto'), eixo = css('--g-eixo'), grade = css('--g-grade');
  ctx.font = '11px Arial, Helvetica, sans-serif';
  const xs = cfg.series.flatMap((s) => s.x), ys = [...cfg.series.flatMap((s) => [...s.y, ...(s.lo ?? []), ...(s.hi ?? [])]), ...(cfg.refs ?? []).map((r) => r.y)].filter(Number.isFinite);
  if (!xs.length || !ys.length) {
    ctx.fillStyle = cinza; ctx.textAlign = 'center'; ctx.fillText(cfg.vazio ?? '–', w / 2, h / 2);
    canvas.__modelo = null; return;
  }
  const x0 = cfg.xMin ?? Math.min(...xs), x1 = cfg.xMax ?? Math.max(...xs);
  let y0 = cfg.yMin ?? Math.min(...ys), y1 = cfg.yMax ?? Math.max(...ys);
  if (cfg.yMin === undefined) y0 -= (y1 - y0) * 0.05 || 1;
  if (cfg.yMax === undefined) y1 += (y1 - y0) * 0.05 || 1;
  const ty = ticks(y0, y1, 5), tx = ticks(x0, x1, 5);
  const X = (v) => L + ((v - x0) / (x1 - x0 || 1)) * (w - L - R), Y = (v) => h - B - ((v - y0) / (y1 - y0 || 1)) * (h - B - T);
  canvas.__modelo = { w, h, L, R, x0, x1, cfg };
  ctx.lineWidth = 1; ctx.textBaseline = 'middle'; ctx.textAlign = 'right';
  for (const v of ty) { ctx.strokeStyle = grade; ctx.beginPath(); ctx.moveTo(L, Y(v) + 0.5); ctx.lineTo(w - R, Y(v) + 0.5); ctx.stroke(); ctx.fillStyle = cinza; ctx.fillText((cfg.yFmt ?? abrevia)(v), L - 6, Y(v)); }
  ctx.textAlign = 'center'; ctx.textBaseline = 'top';
  for (const v of tx) { ctx.strokeStyle = grade; ctx.beginPath(); ctx.moveTo(X(v) + 0.5, T); ctx.lineTo(X(v) + 0.5, h - B); ctx.stroke(); ctx.fillStyle = cinza; ctx.fillText((cfg.xFmt ?? abrevia)(v), X(v), h - B + 5); }
  ctx.strokeStyle = eixo; ctx.beginPath(); ctx.moveTo(L, T); ctx.lineTo(L, h - B); ctx.lineTo(w - R, h - B); ctx.stroke();
  ctx.fillStyle = cinza; ctx.textAlign = 'center'; ctx.textBaseline = 'bottom';
  if (cfg.xRot) ctx.fillText(cfg.xRot, L + (w - L - R) / 2, h - 2);
  if (cfg.yRot) { ctx.save(); ctx.translate(11, T + (h - T - B) / 2); ctx.rotate(-Math.PI / 2); ctx.textBaseline = 'top'; ctx.fillText(cfg.yRot, 0, -3); ctx.restore(); }
  ctx.save(); ctx.beginPath(); ctx.rect(L, T - 2, w - L - R + 2, h - T - B + 4); ctx.clip();
  for (const r of cfg.refs ?? []) {
    ctx.strokeStyle = r.cor; ctx.lineWidth = 1.4; ctx.setLineDash(r.dash ?? [6, 4]);
    ctx.beginPath(); ctx.moveTo(L, Y(r.y)); ctx.lineTo(w - R, Y(r.y)); ctx.stroke(); ctx.setLineDash([]);
  }
  for (const s of cfg.series) {
    if (s.lo && s.hi && s.x.length > 1) {
      ctx.fillStyle = s.cor; ctx.globalAlpha = 0.16; ctx.beginPath();
      s.x.forEach((x, i) => (i ? ctx.lineTo(X(x), Y(s.hi[i])) : ctx.moveTo(X(x), Y(s.hi[i]))));
      for (let i = s.x.length - 1; i >= 0; i--) ctx.lineTo(X(s.x[i]), Y(s.lo[i]));
      ctx.closePath(); ctx.fill(); ctx.globalAlpha = 1;
    }
    ctx.strokeStyle = s.cor; ctx.lineWidth = s.largura ?? 2; ctx.setLineDash(s.dash ?? []); ctx.lineJoin = 'round';
    if (s.alfa !== undefined) ctx.globalAlpha = s.alfa;
    ctx.beginPath();
    let ini = true;
    s.x.forEach((x, i) => { if (!Number.isFinite(s.y[i])) { ini = true; return; } if (ini) { ctx.moveTo(X(x), Y(s.y[i])); ini = false; } else ctx.lineTo(X(x), Y(s.y[i])); });
    ctx.stroke(); ctx.setLineDash([]);
    if (s.pontos) { ctx.fillStyle = s.cor; s.x.forEach((x, i) => { if (Number.isFinite(s.y[i])) { ctx.beginPath(); ctx.arc(X(x), Y(s.y[i]), 2.5, 0, 7); ctx.fill(); } }); }
    ctx.globalAlpha = 1;
  }
  ctx.restore();
  if (canvas.__guia != null) { ctx.strokeStyle = eixo; ctx.lineWidth = 1; ctx.setLineDash([3, 3]); ctx.beginPath(); ctx.moveTo(canvas.__guia, T); ctx.lineTo(canvas.__guia, h - B); ctx.stroke(); ctx.setLineDash([]); }
}

/**
 * Barras horizontais. cfg: { itens:[{rotulo, valor, ic?, cor, pontos?[]}], xMin, xMax, xFmt, ref?:{x, nome, cor} }
 * A altura do canvas acompanha o número de itens.
 */
export function graficoBarras(canvas, cfg) {
  canvas.__cfgB = cfg;
  if (!canvas.__prontoB) { canvas.__prontoB = true; aoRedimensionar(canvas, () => desenharBarras(canvas)); }
  desenharBarras(canvas);
}
function desenharBarras(canvas) {
  const cfg = canvas.__cfgB, linhaH = 26, T = 6, B = 26;
  canvas.style.height = `${cfg.itens.length * linhaH + T + B}px`;
  const { w, h, ctx } = prepararCanvas(canvas);
  const cinza = css('--g-texto'), eixo = css('--g-eixo'), grade = css('--g-grade'), texto = css('--texto');
  ctx.font = '11.5px Arial, Helvetica, sans-serif';
  const rotMax = Math.min(w * 0.38, Math.max(...cfg.itens.map((i) => ctx.measureText(i.rotulo).width)) + 10);
  const L = rotMax, R = 46;
  const vals = cfg.itens.flatMap((i) => [i.valor, i.valor + (i.ic || 0), i.valor - (i.ic || 0), ...(i.pontos ?? [])]).filter(Number.isFinite);
  let x0 = cfg.xMin ?? Math.min(0, ...vals), x1 = cfg.xMax ?? Math.max(0, ...vals);
  if (cfg.xMin === undefined && x0 < 0) x0 -= (x1 - x0) * 0.04;
  if (cfg.xMax === undefined) x1 += (x1 - x0) * 0.04 || 1;
  const X = (v) => L + ((Math.min(Math.max(v, x0), x1) - x0) / (x1 - x0 || 1)) * (w - L - R);
  const tx = ticks(x0, x1, Math.max(2, Math.floor((w - L - R) / 70)));
  ctx.lineWidth = 1; ctx.textAlign = 'center'; ctx.textBaseline = 'top';
  for (const v of tx) { ctx.strokeStyle = grade; ctx.beginPath(); ctx.moveTo(X(v) + 0.5, T); ctx.lineTo(X(v) + 0.5, h - B); ctx.stroke(); ctx.fillStyle = cinza; ctx.fillText((cfg.xFmt ?? abrevia)(v), X(v), h - B + 5); }
  const zero = X(0);
  cfg.itens.forEach((it, i) => {
    const yc = T + i * linhaH + linhaH / 2, bh = linhaH * 0.56;
    ctx.fillStyle = texto; ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
    const rot = ctx.measureText(it.rotulo).width > L - 8 ? it.rotulo.slice(0, 14) + '…' : it.rotulo;
    ctx.fillText(rot, L - 8, yc);
    ctx.fillStyle = it.cor; ctx.globalAlpha = 0.88;
    const xv = X(it.valor);
    ctx.fillRect(Math.min(zero, xv), yc - bh / 2, Math.max(1, Math.abs(xv - zero)), bh);
    ctx.globalAlpha = 1;
    if (it.ic) {
      ctx.strokeStyle = texto; ctx.lineWidth = 1.2; const a = X(it.valor - it.ic), b = X(it.valor + it.ic);
      ctx.beginPath(); ctx.moveTo(a, yc); ctx.lineTo(b, yc); ctx.moveTo(a, yc - 4); ctx.lineTo(a, yc + 4); ctx.moveTo(b, yc - 4); ctx.lineTo(b, yc + 4); ctx.stroke();
    }
    for (const p of it.pontos ?? []) { ctx.fillStyle = '#fff'; ctx.strokeStyle = '#222'; ctx.lineWidth = 0.8; ctx.beginPath(); ctx.arc(X(p), yc, 2.6, 0, 7); ctx.fill(); ctx.stroke(); }
    ctx.fillStyle = texto; ctx.textAlign = 'left'; ctx.fillText((cfg.xFmt ?? abrevia)(it.valor), Math.min(w - R + 4, Math.max(xv, X(it.valor + (it.ic || 0))) + 5), yc);
  });
  ctx.strokeStyle = eixo; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(zero + 0.5, T); ctx.lineTo(zero + 0.5, h - B); ctx.stroke();
  if (cfg.ref) { ctx.strokeStyle = cfg.ref.cor; ctx.setLineDash([5, 4]); ctx.lineWidth = 1.4; ctx.beginPath(); ctx.moveTo(X(cfg.ref.x), T); ctx.lineTo(X(cfg.ref.x), h - B); ctx.stroke(); ctx.setLineDash([]); }
}

/** Legenda HTML (chips com amostra de cor) para um conjunto de séries. */
export function legenda(container, itens) {
  container.replaceChildren(...itens.map((s) => el('span', { class: 'leg-item', html: `<i class="sw" style="background:${s.cor}"></i>${s.nome}` })));
}

/** Mini-gráfico de linha (sem eixos) para o histórico da intensidade da correnteza. */
export function spark(canvas, valores, { yMin = 0, yMax = 1, media = 0.5, rotulo = '' } = {}) {
  const { w, h, ctx } = prepararCanvas(canvas);
  const cor = css('--azul2') || '#4A5AB8', grade = css('--g-grade'), cinza = css('--g-texto');
  const Y = (v) => h - 4 - ((v - yMin) / (yMax - yMin)) * (h - 8);
  ctx.strokeStyle = grade; ctx.lineWidth = 1; ctx.setLineDash([3, 3]); ctx.beginPath(); ctx.moveTo(0, Y(media)); ctx.lineTo(w, Y(media)); ctx.stroke(); ctx.setLineDash([]);
  ctx.fillStyle = cinza; ctx.font = '10.5px Arial, sans-serif'; ctx.textAlign = 'left'; ctx.textBaseline = 'top'; ctx.fillText(rotulo, 2, 1);
  if (valores.length < 2) return;
  const n = Math.max(valores.length - 1, 1), N = Math.max(100, n);
  ctx.strokeStyle = cor; ctx.lineWidth = 2; ctx.lineJoin = 'round'; ctx.beginPath();
  valores.forEach((v, i) => (i ? ctx.lineTo((i / N) * w, Y(v)) : ctx.moveTo(0, Y(v)))); ctx.stroke();
  const ult = valores[valores.length - 1];
  ctx.fillStyle = cor; ctx.beginPath(); ctx.arc(((valores.length - 1) / N) * w, Y(ult), 3, 0, 7); ctx.fill();
}
