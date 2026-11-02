// Utilidades de DOM e cores (lidas das variáveis CSS para acompanhar o tema claro/escuro). / DOM and theme-aware color helpers.
export const $ = (s, r = document) => r.querySelector(s);
export const $$ = (s, r = document) => [...r.querySelectorAll(s)];
export const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));

export function el(tag, attrs = {}, filhos = []) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === 'class') e.className = v;
    else if (k === 'text') e.textContent = v;
    else if (k === 'html') e.innerHTML = v;
    else e.setAttribute(k, v);
  }
  for (const f of [].concat(filhos)) e.append(f);
  return e;
}

let _estilo = null, _chaveTema = '';
/** Valor de uma variável CSS no tema atual (cache invalidado quando o tema muda). */
export function css(nome) {
  const chave = document.documentElement.dataset.theme + '|' + (window.matchMedia?.('(prefers-color-scheme: dark)').matches ? 'd' : 'l');
  if (!_estilo || chave !== _chaveTema) { _estilo = getComputedStyle(document.documentElement); _chaveTema = chave; }
  return _estilo.getPropertyValue(nome).trim();
}

/** Cor do agente/algoritmo/linha de base no tema atual. */
export function corAgente(id) {
  const k = { DQN: '--c-dqn', PPO: '--c-ppo', A2C: '--c-a2c', SARSA: '--c-sarsa', sarsa: '--c-sarsa', sarsa_pre: '--c-sarsa', sarsa_user: '--c-sarsa',
    tabular: '--c-tabular', heuristica: '--c-heuristica', aleatoria: '--c-aleatoria' }[id];
  return k ? css(k) : css('--cinza');
}

/** Dimensiona um canvas ao tamanho CSS com a densidade de pixels da tela. Devolve `{w, h, ctx}` já com a escala aplicada. */
export function prepararCanvas(canvas) {
  const dpr = window.devicePixelRatio || 1;
  const r = canvas.getBoundingClientRect();
  const w = Math.max(10, Math.round(r.width)), h = Math.max(10, Math.round(r.height));
  if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) { canvas.width = Math.round(w * dpr); canvas.height = Math.round(h * dpr); }
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);
  return { w, h, ctx };
}

/** Chama `fn` quando o elemento muda de tamanho (inclusive quando sua seção passa a ser exibida). */
export function aoRedimensionar(elemento, fn) {
  if (typeof ResizeObserver === 'undefined') { window.addEventListener('resize', fn); return; }
  let q = 0;
  new ResizeObserver(() => { cancelAnimationFrame(q); q = requestAnimationFrame(fn); }).observe(elemento);
}

export const reduzMovimento = () => window.matchMedia?.('(prefers-reduced-motion: reduce)').matches ?? false;
export const ceder = () => new Promise((r) => setTimeout(r, 0));
