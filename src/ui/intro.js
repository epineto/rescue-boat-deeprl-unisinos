// Seção #intro: números-chave calculados dos dados, lista de agentes e demonstração (episódio da heurística em laço).
// Intro section: key numbers from the data, agent list and a looping demo episode.
import { BarcoContinuo } from '../core/env.js';
import { politicaHeuristica } from '../core/politicas.js';
import { ALGOS } from '../core/dados.js';
import { $, el, reduzMovimento } from './util.js';
import { Painel } from './painel.js';
import * as I18N from './i18n.js';
import { onTema } from './tema.js';

const { t, fmt, fmtI } = I18N;

export function iniciarIntro(S) {
  const D = S.dados, cat = S.catalogo;
  function numeros() {
    const c = D.indice.contagem ?? {};
    const algs = D.finalResumo.filter((r) => ALGOS.includes(r.agente)).sort((a, b) => b.ret_desc - a.ret_desc);
    const cfgs = D.buscaResumo ? D.buscaResumo.length : null;
    const nomes = (r) => (r.agente === 'SARSA' ? 'Sarsa' : r.agente);
    const tiles = [
      [fmtI((c.busca ?? 0) + (c.final ?? 0)), t('intro.n.runs')],
      [cfgs === null ? '–' : fmtI(cfgs), t('intro.n.cfg')],
      [algs.length ? `${nomes(algs[0])} · ${fmt(algs[0].ret_desc, 1)}` : '–', t('intro.n.melhor')],
      [D.base?.heuristica ? fmt(D.base.heuristica.ret_desc, 1) : '–', t('intro.n.heur')],
    ];
    $('#hero-nums').replaceChildren(...tiles.map(([v, k]) => el('div', {}, [el('strong', { text: v }), el('span', { text: k })])));
  }
  function parametros() {
    const a = D.indice.ambiente ?? {};
    const itens = [['passo', fmt(a.passo, 2)], ['sigma', fmt(a.sigma_mov, 2)], ['re', fmt(a.r_embarque, 2)], ['ra', fmt(a.r_abrigo, 2)], ['rd', fmt(a.r_destrocos, 2)],
      ['vmax', fmt(a.corr_vmax, 2)], ['mu', fmt(a.corr_media, 2)], ['theta', fmt(a.corr_theta, 2)], ['sig_i', fmt(a.corr_sigma_i, 2)], ['hor', fmtI(a.max_passos)]];
    $('#intro-params').replaceChildren(...itens.map(([k, v]) => el('div', {}, [el('dt', { text: t('intro.par.' + k) }), el('dd', { text: v })])));
  }
  function agentes() {
    const ids = ['DQN', 'PPO', 'A2C', 'sarsa_pre', 'tabular', 'heuristica', 'aleatoria'];
    $('#intro-agentes').replaceChildren(...ids.map((id) => el('li', {}, [el('i', { class: 'sw', style: `background:${cat.cor(id)}` }),
      el('span', { html: `<strong>${id === 'sarsa_pre' ? 'Sarsa' : cat.rotulo(id)}:</strong> ${t('intro.ag.' + id)}` })])));
  }

  // demonstração: heurística em laço (ou quadro final estático se o usuário prefere menos movimento)
  const canvas = $('#intro-canvas');
  const env0 = new BarcoContinuo();
  const painel = new Painel(canvas, politicaHeuristica(env0), { raios: true });
  let ativa = false, raf = 0, ult = 0, acc = 0, sem = 10001, espera = 0;
  const legendaDemo = () => {
    const m = painel.metricas;
    $('#intro-demo-leg').textContent = t('intro.demo.leg', { s: sem, p: m.passos, e: m.entregues, r: fmt(m.ret, 0) });
    canvas.setAttribute('aria-label', t('intro.demo.aria') + ' ' + $('#intro-demo-leg').textContent);
  };
  function estatico() { painel.iniciar(sem); while (painel.passo()) { /* roda tudo */ } painel.desenhar(); legendaDemo(); }
  function quadro(ts) {
    if (!ativa) return;
    const dt = Math.min(0.25, (ts - ult) / 1000); ult = ts;
    if (espera > 0) { espera -= dt; if (espera <= 0) { sem++; painel.iniciar(sem); legendaDemo(); } }
    else {
      acc += dt * 9;
      let n = Math.min(4, Math.floor(acc)); acc -= Math.floor(acc);
      while (n-- > 0 && painel.passo()) { /* avança */ }
      painel.desenhar(((ts / 1400) % 1));
      legendaDemo();
      if (painel.terminado) espera = 2.2;
    }
    raf = requestAnimationFrame(quadro);
  }
  numeros(); parametros(); agentes(); legendaDemo();
  I18N.onLang(() => { numeros(); parametros(); agentes(); legendaDemo(); });
  onTema(() => painel.desenhar());
  return {
    ativar() {
      ativa = true; cancelAnimationFrame(raf);
      if (reduzMovimento()) estatico();
      else { painel.iniciar(sem); espera = 0; acc = 0; ult = performance.now(); raf = requestAnimationFrame(quadro); }
    },
    desativar() { ativa = false; cancelAnimationFrame(raf); },
  };
}
