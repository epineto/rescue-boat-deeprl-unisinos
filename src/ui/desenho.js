// Desenho da cena em canvas: mapa 6×6, faixa da correnteza, blocos, destroços (discos), pessoas, abrigo, raios, trilha e barco.
// Canvas scene drawing, theme-aware (colors come from CSS variables).
import { css } from './util.js';

export function paleta() {
  const c = {};
  for (const k of ['agua', 'corr', 'corr-t', 'bloco', 'abrigo', 'destr', 'destr-b', 'grade', 'pessoa', 'barco', 'vela', 'trilha']) c[k] = css('--m-' + k);
  c.texto = css('--texto'); c.verde = css('--verde'); c.azul = css('--azul'); c.verm = css('--verm');
  return c;
}

function barco(ctx, cx, cy, u, C, direcao) {
  ctx.save();
  ctx.translate(cx, cy);
  ctx.scale(u, u);
  ctx.lineJoin = 'round';
  ctx.fillStyle = C.barco;
  ctx.beginPath(); ctx.moveTo(-0.28, 0.06); ctx.lineTo(0.28, 0.06); ctx.lineTo(0.17, 0.22); ctx.lineTo(-0.17, 0.22); ctx.closePath(); ctx.fill();
  ctx.strokeStyle = C.texto; ctx.lineWidth = 0.025; ctx.stroke();
  ctx.beginPath(); ctx.moveTo(-0.02, 0.06); ctx.lineTo(-0.02, -0.32); ctx.stroke();
  ctx.fillStyle = C.vela; ctx.strokeStyle = C.azul;
  ctx.beginPath(); ctx.moveTo(0.0, -0.3); ctx.lineTo(0.0, 0.03); ctx.lineTo(0.24, 0.03); ctx.closePath(); ctx.fill(); ctx.stroke();
  ctx.restore();
  if (direcao !== undefined && direcao !== null) { // pequena seta da última ação, ao lado do barco
    const d = [[0, -1], [0, 1], [1, 0], [-1, 0]][direcao];
    const x0 = cx + d[0] * u * 0.42, y0 = cy + d[1] * u * 0.42 - u * 0.04;
    ctx.save(); ctx.fillStyle = C.texto; ctx.globalAlpha = 0.75;
    ctx.beginPath();
    ctx.moveTo(x0 + d[0] * u * 0.12, y0 + d[1] * u * 0.12);
    ctx.lineTo(x0 - d[0] * u * 0.05 - d[1] * u * 0.09, y0 - d[1] * u * 0.05 + d[0] * u * 0.09);
    ctx.lineTo(x0 - d[0] * u * 0.05 + d[1] * u * 0.09, y0 - d[1] * u * 0.05 - d[0] * u * 0.09);
    ctx.closePath(); ctx.fill(); ctx.restore();
  }
}

/**
 * Desenha o estado de `env` (BarcoContinuo) em um contexto já escalado, `w` = lado em px CSS.
 * Opções: `trilha` ([[x, y], …] em células), `raios`, `fase` (0..1, animação do fluxo), `acao` (última ação), `compacto` (rótulos menores).
 */
export function desenharCena(ctx, w, env, o = {}) {
  const C = o.paleta ?? paleta(), u = w / env.nc, p = env.p;
  ctx.clearRect(0, 0, w, w);
  ctx.fillStyle = C.agua; ctx.fillRect(0, 0, w, w);
  ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
  // faixa da correnteza com chevrons que descem; a densidade do traço acompanha a intensidade I
  const [f0, f1] = env.faixa;
  if (f1 > f0) {
    ctx.fillStyle = C.corr; ctx.fillRect(0, f0 * u, w, (f1 - f0) * u);
    const I = env.inten, fase = o.fase ?? 0;
    ctx.strokeStyle = C['corr-t']; ctx.lineWidth = Math.max(1.5, u * 0.035); ctx.lineCap = 'round'; ctx.lineJoin = 'round';
    ctx.globalAlpha = 0.25 + 0.6 * I;
    const nCol = env.nc * 2, nLin = Math.max(1, Math.round((f1 - f0) * 2));
    for (let i = 0; i < nCol; i++) {
      for (let j = 0; j < nLin; j++) {
        const cx = (i + 0.5) * (w / nCol), yy = ((j + fase) % nLin) / nLin, cy = f0 * u + yy * (f1 - f0) * u;
        const h = u * (0.05 + 0.07 * I);
        ctx.beginPath(); ctx.moveTo(cx - h, cy - h * 0.8); ctx.lineTo(cx, cy + h * 0.6); ctx.lineTo(cx + h, cy - h * 0.8); ctx.stroke();
      }
    }
    ctx.globalAlpha = 1;
  }
  // grade e blocos
  ctx.strokeStyle = C.grade; ctx.lineWidth = 1;
  for (let i = 0; i <= env.nc; i++) { ctx.beginPath(); ctx.moveTo(i * u + 0.5, 0); ctx.lineTo(i * u + 0.5, w); ctx.stroke(); }
  for (let i = 0; i <= env.nl; i++) { ctx.beginPath(); ctx.moveTo(0, i * u + 0.5); ctx.lineTo(w, i * u + 0.5); ctx.stroke(); }
  for (let i = 0; i < env.nl; i++) {
    for (let j = 0; j < env.nc; j++) {
      const c = env.mapa[i][j];
      if (c === '#') {
        ctx.fillStyle = C.bloco; ctx.beginPath(); ctx.roundRect(j * u + 1, i * u + 1, u - 2, u - 2, u * 0.08); ctx.fill();
        ctx.strokeStyle = 'rgba(255,255,255,.25)'; ctx.lineWidth = 1;
        for (let k = 1; k < 4; k++) { ctx.beginPath(); ctx.moveTo(j * u + k * u / 4, i * u + 3); ctx.lineTo(j * u + k * u / 4 - u * 0.2, i * u + u - 3); ctx.stroke(); }
      } else if (c === 'H') {
        ctx.fillStyle = C.abrigo; ctx.fillRect(j * u + 1, i * u + 1, u - 2, u - 2);
      }
    }
  }
  // destroços: discos
  for (const [cx, cy] of env.destrocos) {
    ctx.fillStyle = C.destr; ctx.strokeStyle = C['destr-b']; ctx.lineWidth = Math.max(1.5, u * 0.03);
    ctx.beginPath(); ctx.arc(cx * u, cy * u, p.r_destrocos * u, 0, 7); ctx.fill(); ctx.stroke();
    ctx.lineWidth = Math.max(2, u * 0.05); ctx.lineCap = 'round'; const r = u * 0.14;
    ctx.beginPath(); ctx.moveTo(cx * u - r, cy * u - r); ctx.lineTo(cx * u + r, cy * u + r); ctx.moveTo(cx * u + r, cy * u - r); ctx.lineTo(cx * u - r, cy * u + r); ctx.stroke();
  }
  // abrigo: rótulo e raio de desembarque
  const [hx, hy] = env.abrigo;
  ctx.fillStyle = C.verde; ctx.font = `bold ${u * 0.34}px Arial, Helvetica, sans-serif`; ctx.fillText('H', hx * u, hy * u);
  if (o.raios) { ctx.save(); ctx.setLineDash([u * 0.08, u * 0.06]); ctx.strokeStyle = C.verde; ctx.lineWidth = Math.max(1.5, u * 0.03); ctx.beginPath(); ctx.arc(hx * u, hy * u, p.r_abrigo * u, 0, 7); ctx.stroke(); ctx.restore(); }
  // pessoas ilhadas
  env.pessoas.forEach(([px, py], k) => {
    if (env.status[k] !== 0) return;
    if (o.raios) { ctx.save(); ctx.setLineDash([u * 0.08, u * 0.06]); ctx.strokeStyle = C.pessoa; ctx.lineWidth = Math.max(1.5, u * 0.03); ctx.beginPath(); ctx.arc(px * u, py * u, p.r_embarque * u, 0, 7); ctx.stroke(); ctx.restore(); }
    ctx.fillStyle = C.pessoa; ctx.beginPath(); ctx.arc(px * u, py * u, u * 0.19, 0, 7); ctx.fill();
    ctx.fillStyle = '#fff'; ctx.font = `bold ${u * 0.2}px Arial, Helvetica, sans-serif`; ctx.fillText(`P${k + 1}`, px * u, py * u + u * 0.01);
  });
  // salvas: selos verdes no canto do abrigo
  let ns = 0;
  env.status.forEach((s, k) => {
    if (s !== 2) return;
    const r = u * 0.11, X = (Math.floor(hx) + 1) * u - r - u * 0.05 - ns * (2 * r + u * 0.03), Y = Math.floor(hy) * u + r + u * 0.05;
    ctx.fillStyle = C.verde; ctx.beginPath(); ctx.arc(X, Y, r, 0, 7); ctx.fill();
    ctx.fillStyle = '#fff'; ctx.font = `bold ${u * 0.13}px Arial, Helvetica, sans-serif`; ctx.fillText(`P${k + 1}`, X, Y + u * 0.005);
    ns++;
  });
  // trilha
  if (o.trilha && o.trilha.length > 1) {
    ctx.save(); ctx.strokeStyle = C.trilha; ctx.lineWidth = Math.max(2, u * 0.04); ctx.setLineDash([u * 0.1, u * 0.07]); ctx.lineJoin = 'round';
    ctx.beginPath(); o.trilha.forEach(([x, y], i) => (i ? ctx.lineTo(x * u, y * u) : ctx.moveTo(x * u, y * u))); ctx.stroke(); ctx.restore();
  }
  // barco e pessoas a bordo (marcadores sobre o casco)
  barco(ctx, env.x * u, env.y * u - u * 0.02, u, C, o.acao);
  let nb = 0;
  env.status.forEach((s, k) => {
    if (s !== 1) return;
    const r = u * 0.1, X = env.x * u - u * 0.14 + nb * (2 * r + u * 0.02), Y = env.y * u + u * 0.03;
    ctx.fillStyle = '#fff'; ctx.strokeStyle = C.verm; ctx.lineWidth = Math.max(1.5, u * 0.025); ctx.beginPath(); ctx.arc(X, Y, r, 0, 7); ctx.fill(); ctx.stroke();
    ctx.fillStyle = C.verm; ctx.font = `bold ${u * 0.11}px Arial, Helvetica, sans-serif`; ctx.fillText(`P${k + 1}`, X, Y + u * 0.005);
    nb++;
  });
}
