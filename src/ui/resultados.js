// Seção #resultados: explorador dos resultados do experimento (resultado final, busca completa, curvas, hipóteses e dados).
// Results explorer: final table, full search, learning curves with bands, hypotheses and data files.
import { ALGOS, agregarBusca, curvaMedia, curvasPorSemente, contarAprendizes } from '../core/dados.js';
import { desvio, minimo, maximo } from '../core/estat.js';
import { $, $$, el, corAgente, css } from './util.js';
import { graficoBarras, graficoLinhas, legenda } from './graficos.js';
import * as I18N from './i18n.js';
import { onTema } from './tema.js';

const { t, fmt, fmtP, fmtI } = I18N;
const NOME_ALGO = (a) => (a === 'SARSA' ? 'Sarsa' : a);
const PALETA_CFG = ['#5B6CD6', '#E4574A', '#2E9E6B', '#C99A1E', '#A25BD0', '#18A3B8'];

/** Texto compacto dos hiperparâmetros: `chave=valor, …` (vazio = padrão do SB3). */
export function textoHp(hp) {
  if (!hp || typeof hp === 'string') return hp || '';
  const e = Object.entries(hp);
  return e.length ? e.map(([k, v]) => `${k}=${Array.isArray(v) ? '[' + v.join(',') + ']' : v}`).join(', ') : t('res.hp_padrao');
}

/** Traduz os textos em português de hipoteses.csv (gerados por experimentos_t3.py) para exibição em EN; em PT devolve o original. */
export function traduzHipotese(txt, lang = I18N.getLang()) {
  if (lang === 'pt') return txt;
  const ref = { tabular: 'tabular baseline', heuristica: 'heuristic' };
  let m;
  if ((m = txt.match(/^H1: DQN vs (\w+) \(AUC do retorno descontado\)$/))) return `H1: DQN vs ${m[1]} (AUC of the discounted return)`;
  if (/^H2: desvio entre sementes, PPO vs A2C/.test(txt)) return 'H2: spread across seeds, PPO vs A2C (final discounted return)';
  if ((m = txt.match(/^H3: (\w+) vs (\w+) \(retorno descontado final\)$/))) return `H3: ${m[1]} vs ${ref[m[2]] ?? m[2]} (final discounted return)`;
  return txt;
}
export function traduzVeredito(txt, lang = I18N.getLang()) {
  if (lang === 'pt') return txt;
  const mapa = [[/^confirmada \(abaixo da refer[eê]ncia\)$/, 'confirmed (below the reference)'], [/^refutada \(acima da refer[eê]ncia\)$/, 'refuted (above the reference)'],
    [/^inconclusiva \(sem diferen[cç]a significativa\)$/, 'inconclusive (no significant difference)'], [/^inconclusiva \(dire[cç][aã]o favor[aá]vel, p ≥ 0,05\)$/, 'inconclusive (favorable direction, p ≥ 0.05)'],
    [/^refutada \(dire[cç][aã]o contr[aá]ria\)$/, 'refuted (opposite direction)'], [/^inconclusiva \(sem diferen[cç]a\)$/, 'inconclusive (no difference)'], [/^confirmada$/, 'confirmed']];
  for (const [re, en] of mapa) if (re.test(txt)) return en;
  return txt;
}
export const classeVeredito = (txt) => (/^confirmada/.test(txt) ? 'v-conf' : /^refutada/.test(txt) ? 'v-refut' : 'v-inc');


/**
 * Primeira avaliação da curva com sucesso ≥ `limiar`: `{passo, anterior}` (passo da avaliação anterior; `null` se for a primeira) ou `null` se nunca ocorre.
 * `curva`: linhas `{passos, sucesso}` de uma única run.
 */
export function primeiroSucesso(curva, limiar = 0.5) {
  const ord = [...curva].sort((a, b) => a.passos - b.passos);
  const i = ord.findIndex((l) => l.sucesso >= limiar);
  return i < 0 ? null : { passo: ord[i].passos, anterior: i > 0 ? ord[i - 1].passos : null };
}

export function iniciarResultados(S) {
  const D = S.dados, cat = S.catalogo, indice = D.indice;
  const melhores = indice.melhores ?? {};
  const algosPresentes = ALGOS.filter((a) => D.finalResumo.some((r) => r.agente === a) || D.buscaRuns?.some((r) => r.algo === a));

  // ---- abas ------------------------------------------------------------------------------------------------------------
  const abas = $$('.abas [role="tab"]');
  function abrirAba(id, foco) {
    for (const b of abas) {
      const on = b.id === id;
      b.setAttribute('aria-selected', String(on)); b.tabIndex = on ? 0 : -1;
      $('#' + b.getAttribute('aria-controls')).hidden = !on;
      if (on && foco) b.focus();
    }
    window.dispatchEvent(new Event('resize'));
    desenharTudo();
  }
  abas.forEach((b, i) => {
    b.addEventListener('click', () => abrirAba(b.id));
    b.addEventListener('keydown', (e) => {
      const k = { ArrowRight: 1, ArrowLeft: -1 }[e.key];
      if (k) { e.preventDefault(); abrirAba(abas[(i + k + abas.length) % abas.length].id, true); }
      else if (e.key === 'Home') { e.preventDefault(); abrirAba(abas[0].id, true); }
      else if (e.key === 'End') { e.preventDefault(); abrirAba(abas[abas.length - 1].id, true); }
    });
  });
  const abaAtual = () => abas.find((b) => b.getAttribute('aria-selected') === 'true').id;

  // ---- resultado final -----------------------------------------------------------------------------------------------------
  const nomeAg = (a) => ({ tabular: t('ag.tabular'), heuristica: t('ag.heuristica'), aleatoria: t('ag.aleatoria') }[a] ?? NOME_ALGO(a));
  const rotuloAg = (r) => (ALGOS.includes(r.agente) ? NOME_ALGO(r.agente) : nomeAg(r.agente));
  function final() {
    const p = indice.protocolo ?? {};
    $('#res-final-d').textContent = t('res.final.d', { orc: p.orcamento_final ? fmtI(p.orcamento_final) : '–', n: p.sementes_final?.length ?? '–', crit: p.criterio ?? t('res.final.crit') });
    const linhas = [...ALGOS.map((a) => D.finalResumo.find((r) => r.agente === a)).filter(Boolean), ...['tabular', 'heuristica', 'aleatoria'].map((a) => D.finalResumo.find((r) => r.agente === a)).filter(Boolean)];
    const algs = linhas.filter((r) => ALGOS.includes(r.agente));
    const melhorRd = Math.max(...algs.map((r) => r.ret_desc)), melhorSuc = Math.max(...algs.map((r) => r.sucesso));
    const cols = [['agente', 'res.c.agente'], ['config', 'res.c.config'], ['n_sementes', 'res.c.sem'], ['ret_desc', 'm.ret_desc'], ['ret', 'm.ret'], ['sucesso', 'm.sucesso'], ['passos_ep', 'm.passos_ep'], ['toca_destrocos', 'm.toca']];
    const celula = (r, k) => {
      const ic = r[k + '_ic95'], pm = (v, f) => `${f(v)}${Number.isFinite(ic) ? ' ± ' + (k === 'sucesso' || k === 'toca_destrocos' ? fmt(ic * 100, 1) : fmt(ic, 1)) : ''}`;
      if (k === 'sucesso' || k === 'toca_destrocos') return pm(r[k], (v) => fmtP(v, 1));
      return pm(r[k], (v) => fmt(v, 1));
    };
    $('#res-final-tab').replaceChildren(
      el('thead', {}, [el('tr', {}, cols.map(([, k]) => el('th', { scope: 'col', text: t(k) })))]),
      el('tbody', {}, linhas.map((r) => {
        const alg = ALGOS.includes(r.agente);
        return el('tr', {}, [el('th', { scope: 'row', html: `<i class="sw" style="background:${corAgente(r.agente)}"></i>${rotuloAg(r)}${alg ? '' : `<span class="tag cinza">${t('res.ref')}</span>`}` }),
          el('td', { class: 'hp', text: r.config === '-' ? '–' : r.config }), el('td', { text: alg ? String(r.n_sementes) : '–' }),
          el('td', { class: alg && Math.abs(r.ret_desc - melhorRd) < 1e-9 ? 'melhor' : '', text: celula(r, 'ret_desc') }), el('td', { text: celula(r, 'ret') }),
          el('td', { class: alg && Math.abs(r.sucesso - melhorSuc) < 1e-9 ? 'melhor' : '', text: celula(r, 'sucesso') }), el('td', { text: celula(r, 'passos_ep') }), el('td', { text: celula(r, 'toca_destrocos') })]);
      })));
    const porSemente = (a, k) => (D.finalRuns ?? []).filter((x) => x.algo === a).map((x) => x[k]);
    const itens = (k, escala = 1) => linhas.map((r) => ({ rotulo: rotuloAg(r), valor: r[k] * escala, ic: Number.isFinite(r[k + '_ic95']) ? r[k + '_ic95'] * escala : 0, cor: corAgente(r.agente), pontos: ALGOS.includes(r.agente) ? porSemente(r.agente, k).map((v) => v * escala) : [] }));
    graficoBarras($('#res-g-ret'), { itens: itens('ret_desc'), xFmt: (v) => fmt(v, 0) });
    graficoBarras($('#res-g-suc'), { itens: itens('sucesso'), xMin: 0, xMax: 1, xFmt: (v) => fmtP(v, 0) });
    graficoBarras($('#res-g-toca'), { itens: itens('toca_destrocos'), xMin: 0, xMax: 1, xFmt: (v) => fmtP(v, 0) });
    $('#res-g-ret').setAttribute('aria-label', `${t('res.g.ret')}: ` + linhas.map((r) => `${rotuloAg(r)} ${fmt(r.ret_desc, 1)}`).join('; '));
    $('#res-g-suc').setAttribute('aria-label', `${t('res.g.suc')}: ` + linhas.map((r) => `${rotuloAg(r)} ${fmtP(r.sucesso, 0)}`).join('; '));
    $('#res-g-toca').setAttribute('aria-label', `${t('res.g.toca')}: ` + linhas.map((r) => `${rotuloAg(r)} ${fmtP(r.toca_destrocos, 0)}`).join('; '));
  }

  // ---- busca completa ------------------------------------------------------------------------------------------------------
  const estBusca = { algos: new Set(algosPresentes), q: '', ordem: 'ret_desc', dir: -1, sementes: false };
  function chipsAlgos(box, conj, aoMudar, multi = true) {
    box.replaceChildren(...algosPresentes.map((a) => {
      const lab = el('label', { class: 'chip-check' }, [el('input', { type: multi ? 'checkbox' : 'radio', name: box.id, ...(conj.has(a) ? { checked: '' } : {}) }), el('i', { class: 'sw', style: `background:${corAgente(a)}` }), el('span', { text: NOME_ALGO(a) })]);
      lab.querySelector('input').addEventListener('change', (e) => { if (e.target.checked) conj.add(a); else conj.delete(a); aoMudar(); });
      return lab;
    }));
  }
  function busca() {
    if (!D.buscaRuns) { $('#busca-tab').replaceChildren(); $('#busca-contagem').textContent = t('res.sem_dados'); return; }
    const p = indice.protocolo ?? {};
    $('#res-busca-d').textContent = t('res.busca.d', { orc: p.orcamento_busca ? fmtI(p.orcamento_busca) : '–', n: p.sementes_busca?.length ?? '–' });
    let agg = agregarBusca(D.buscaRuns);
    const pos = {};
    for (const a of ALGOS) agg.filter((r) => r.algo === a).sort((x, y) => y.ret_desc - x.ret_desc || y.sucesso - x.sucesso).forEach((r, i) => { pos[a + '|' + r.config] = i + 1; });
    const total = agg.length;
    const q = estBusca.q.trim().toLowerCase();
    agg = agg.filter((r) => estBusca.algos.has(r.algo) && (!q || (r.config + ' ' + textoHp(r.hp)).toLowerCase().includes(q)));
    const k = estBusca.ordem;
    agg.sort((a, b) => {
      const va = a[k], vb = b[k];
      const c = typeof va === 'string' ? va.localeCompare(vb) : (va ?? -Infinity) - (vb ?? -Infinity);
      return c * estBusca.dir || ALGOS.indexOf(a.algo) - ALGOS.indexOf(b.algo) || a.config.localeCompare(b.config);
    });
    $('#busca-contagem').textContent = t('res.busca.contagem', { n: agg.length, total });
    const cols = [['algo', 'res.c.algo', false], ['config', 'res.c.config', false], ['n', 'res.c.n', true], ['ret_desc', 'm.ret_desc', true], ['ret_desc_dp', 'res.c.dp', true], ['sucesso', 'm.sucesso', true],
      ['passos_ep', 'm.passos_ep', true], ['toca_destrocos', 'm.toca', true], ['tempo_s', 'res.c.tempo', true], ['hp', 'res.c.hp', false]];
    const cab = el('tr', {}, cols.map(([c, rot]) => {
      const th = el('th', { scope: 'col', 'aria-sort': estBusca.ordem === c ? (estBusca.dir > 0 ? 'ascending' : 'descending') : 'none' });
      const b = el('button', { type: 'button', class: 'ord', text: t(rot) });
      b.addEventListener('click', () => { if (estBusca.ordem === c) estBusca.dir *= -1; else { estBusca.ordem = c; estBusca.dir = ['algo', 'config', 'hp'].includes(c) ? 1 : -1; } busca(); });
      th.append(b); return th;
    }));
    const maxRd = Math.max(...agg.map((r) => r.ret_desc)), linhas = [];
    for (const r of agg) {
      const esc = melhores[r.algo]?.config === r.config;
      linhas.push(el('tr', { class: esc ? 'escolhida' : '' }, [
        el('th', { scope: 'row', html: `<i class="sw" style="background:${corAgente(r.algo)}"></i>${NOME_ALGO(r.algo)}` }),
        el('td', { class: 'hp', html: `${r.config}${esc ? `<span class="tag">${t('res.escolhida')}</span>` : ''}<span class="tag cinza">#${pos[r.algo + '|' + r.config]}</span>` }),
        el('td', { text: String(r.n) }),
        el('td', { class: Math.abs(r.ret_desc - maxRd) < 1e-9 ? 'melhor' : '', text: `${fmt(r.ret_desc, 1)} ± ${fmt(r.ret_desc_ic95 || 0, 1)}` }),
        el('td', { text: Number.isFinite(r.ret_desc_dp) ? fmt(r.ret_desc_dp, 1) : '–' }), el('td', { text: fmtP(r.sucesso, 0) }), el('td', { text: fmt(r.passos_ep, 1) }), el('td', { text: fmtP(r.toca_destrocos, 0) }),
        el('td', { text: fmt(r.tempo_s, 0) }), el('td', { class: 'hp', text: textoHp(r.hp) })]));
      if (estBusca.sementes) {
        for (const s of D.buscaRuns.filter((x) => x.algo === r.algo && x.config === r.config)) {
          linhas.push(el('tr', { class: 'sub' }, [el('th', { scope: 'row', text: '' }), el('td', { class: 'hp', text: t('res.semente', { s: s.semente }) }), el('td', { text: '' }), el('td', { text: fmt(s.ret_desc, 1) }), el('td', { text: '' }),
            el('td', { text: fmtP(s.sucesso, 0) }), el('td', { text: fmt(s.passos_ep, 1) }), el('td', { text: fmtP(s.toca_destrocos, 0) }), el('td', { text: fmt(s.tempo_s, 0) }), el('td', { text: '' })]));
        }
      }
    }
    $('#busca-tab').replaceChildren(el('thead', {}, [cab]), el('tbody', {}, linhas));
  }
  chipsAlgos($('#busca-algos'), estBusca.algos, busca);
  $('#busca-q').addEventListener('input', (e) => { estBusca.q = e.target.value; busca(); });
  $('#busca-sementes').addEventListener('change', (e) => { estBusca.sementes = e.target.checked; busca(); });

  // ---- curvas ------------------------------------------------------------------------------------------------------------------
  const estCv = { fonte: 'final', metrica: 'ret_desc', banda: 'ic', algos: new Set(algosPresentes), algo: algosPresentes[0], refs: new Set(['tabular', 'heuristica']) };
  const REF_COL = { ret_desc: 'ret_desc', ret: 'ret', sucesso: 'sucesso', passos_ep: 'passos' };
  const REFS = ['tabular', 'heuristica', 'aleatoria'];
  function curvas() {
    const fonte = estCv.fonte === 'final' ? D.curvasFinais : D.curvasBusca;
    if (!fonte) { graficoLinhas($('#cv-grafico'), { series: [], vazio: t('res.sem_dados') }); $('#cv-nota').textContent = ''; return; }
    $('#cv-algo-campo').hidden = estCv.fonte !== 'busca';
    $('#cv-algos').hidden = estCv.fonte === 'busca';
    const m = estCv.metrica, series = [];
    const pct = m === 'sucesso';
    const fAlgo = (a) => fonte.filter((l) => l.algo === a);
    const empilha = (nome, cor, linhas, extra = {}) => {
      const c = curvaMedia(linhas, m);
      if (!c.passos.length) return;
      const s = { nome, cor, x: c.passos, y: c.media, ...extra };
      if (estCv.banda === 'ic') { s.lo = c.media.map((v, i) => v - c.ic[i]); s.hi = c.media.map((v, i) => v + c.ic[i]); }
      else if (estCv.banda === 'minmax') { s.lo = c.min; s.hi = c.max; }
      else if (estCv.banda === 'sementes') {
        for (const [sem, cs] of curvasPorSemente(linhas, m)) series.push({ nome: `${nome} s${sem}`, cor, x: cs.passos, y: cs.valores, largura: 1, alfa: 0.35, dica: false });
      }
      series.push(s);
    };
    if (estCv.fonte === 'final') {
      for (const a of ALGOS) if (estCv.algos.has(a)) empilha(NOME_ALGO(a), corAgente(a), fAlgo(a), a === 'SARSA' ? { dash: [6, 3] } : {});
    } else {
      const cfgs = [...new Set(fAlgo(estCv.algo).map((l) => l.config))].sort();
      cfgs.forEach((c, i) => empilha(c.replace(/^[a-z0-9]+_/, '') || c, PALETA_CFG[i % PALETA_CFG.length], fAlgo(estCv.algo).filter((l) => l.config === c)));
    }
    // séries de sementes individuais não entram na legenda nem na dica
    const refs = REFS.filter((r) => estCv.refs.has(r) && D.base?.[r]).map((r) => ({ y: D.base[r][REF_COL[m]], cor: css('--c-' + r), nome: nomeAg(r), dash: r === 'heuristica' ? [2, 3] : r === 'aleatoria' ? [8, 3, 2, 3] : [6, 4] }));
    const principais = series.filter((s) => s.dica !== false);
    const todas = series;
    const unidade = (v) => (pct ? fmtP(v, 0) : m === 'passos_ep' ? fmt(v, 0) : fmt(v, 0));
    graficoLinhas($('#cv-grafico'), { series: todas, refs, xRot: t('tr.eixo_passos'), yRot: t('m.' + m), yFmt: unidade, yMin: pct ? 0 : undefined, yMax: pct ? 1 : undefined, vazio: t('res.sem_dados') });
    const leg = [...principais.map((s) => ({ nome: s.nome, cor: s.cor })), ...refs.map((r) => ({ nome: r.nome + ' (' + t('res.ref') + ')', cor: r.cor }))];
    legenda($('#cv-legenda'), leg);
    const nS = new Set(fonte.filter((l) => estCv.fonte === 'final' || l.algo === estCv.algo).map((l) => `${l.algo}|${l.config}|${l.semente}`)).size;
    $('#cv-nota').textContent = t('res.curvas.nota', { n: nS });
    $('#cv-grafico').setAttribute('aria-label', t('res.curvas.aria') + ' ' + principais.map((s) => `${s.nome}: ${unidade(s.y[s.y.length - 1])}`).join('; '));
  }
  function iniciaControlesCurvas() {
    $('#cv-fonte').addEventListener('change', (e) => { estCv.fonte = e.target.value; estCv.banda = e.target.value === 'busca' ? 'nenhuma' : 'ic'; $('#cv-banda').value = estCv.banda; curvas(); });
    $('#cv-metrica').addEventListener('change', (e) => { estCv.metrica = e.target.value; curvas(); });
    $('#cv-banda').addEventListener('change', (e) => { estCv.banda = e.target.value; curvas(); });
    $('#cv-algo').replaceChildren(...algosPresentes.map((a) => el('option', { value: a, text: NOME_ALGO(a) })));
    $('#cv-algo').addEventListener('change', (e) => { estCv.algo = e.target.value; curvas(); });
    chipsAlgos($('#cv-algos'), estCv.algos, curvas);
    desenhaRefsChips();
  }
  function desenhaRefsChips() {
    $('#cv-refs').replaceChildren(...REFS.filter((r) => D.base?.[r]).map((r) => {
      const lab = el('label', { class: 'chip-check' }, [el('input', { type: 'checkbox', ...(estCv.refs.has(r) ? { checked: '' } : {}) }), el('i', { class: 'sw', style: `background:${css('--c-' + r)}` }), el('span', { text: nomeAg(r) })]);
      lab.querySelector('input').addEventListener('change', (e) => { if (e.target.checked) estCv.refs.add(r); else estCv.refs.delete(r); curvas(); });
      return lab;
    }));
  }

  // ---- hipóteses ---------------------------------------------------------------------------------------------------------------
  function hipoteses() {
    $('#hip-lista').replaceChildren(...['h1', 'h2', 'h3'].map((h) => el('li', { html: t('res.hip.' + h) })));
    const corpo = $('#hip-corpo');
    const desc = el('div', {}, []);
    const dpTab = (() => {
      const alg = ['DQN', 'PPO', 'A2C', 'SARSA'].filter((a) => (D.finalRuns ?? []).some((r) => r.algo === a));
      if (!alg.length) return null;
      return el('div', { class: 'rolagem' }, [el('table', { class: 'tab' }, [
        el('thead', {}, [el('tr', {}, ['res.c.agente', 'res.hip.dp', 'res.hip.min', 'res.hip.max'].map((k) => el('th', { scope: 'col', text: t(k) })))]),
        el('tbody', {}, alg.map((a) => { const v = D.finalRuns.filter((r) => r.algo === a).map((r) => r.ret_desc); return el('tr', {}, [el('th', { scope: 'row', html: `<i class="sw" style="background:${corAgente(a)}"></i>${NOME_ALGO(a)}` }), el('td', { text: Number.isFinite(desvio(v)) ? fmt(desvio(v), 1) : '–' }), el('td', { text: fmt(minimo(v), 1) }), el('td', { text: fmt(maximo(v), 1) })]); }))])]);
    })();
    if (!D.hipoteses) {
      corpo.replaceChildren(el('p', { class: 'aviso', text: t('res.hip.indisp') }), el('h4', { text: t('res.hip.desc') }), ...(dpTab ? [dpTab] : []));
      return;
    }
    const linhas = D.hipoteses.map((h) => el('tr', {}, [el('th', { scope: 'row', text: traduzHipotese(h.hipotese) }), el('td', { text: fmt(h.diferenca, 2) }), el('td', { text: fmt(h.estatistica, 2) }),
      el('td', { text: h.p < 0.001 ? '< 0,001'.replace(',', I18N.getLang() === 'pt' ? ',' : '.') : fmt(h.p, 3) }), el('td', { html: `<span class="veredito ${classeVeredito(h.veredito)}">${traduzVeredito(h.veredito)}</span>` })]));
    corpo.replaceChildren(
      el('div', { class: 'rolagem' }, [el('table', { class: 'tab tab-txt' }, [el('thead', {}, [el('tr', {}, ['res.hip.c_h', 'res.hip.c_dif', 'res.hip.c_est', 'res.hip.c_p', 'res.hip.c_ver'].map((k) => el('th', { scope: 'col', text: t(k) })))]), el('tbody', {}, linhas)])]),
      el('p', { class: 'mini just', text: t('res.hip.cautela') }), desc, el('h4', { text: t('res.hip.desc') }), ...(dpTab ? [dpTab] : []));
  }


  // ---- orçamento maior (experimento suplementar) -----------------------------------------------------------------------------------
  const E = D.extra;
  const corCfg = (c) => (/^a2c/.test(c) ? corAgente('A2C') : c === 'ppo_8envs' ? corAgente('PPO') : PALETA_CFG[4]);
  const TRACOS = [[], [6, 3], [2, 3]];
  const estEx = { metrica: 'sucesso' };
  function extra() {
    if (!E) {
      $('#ex-d').textContent = t('res.extra.indisp');
      for (const id of ['#ex-achados', '#ex-tab-resumo', '#ex-tab-sem', '#ex-tab-cp']) $(id).replaceChildren();
      for (const id of ['#ex-lim', '#ex-curvas-nota', '#ex-sem-nota', '#ex-cp-d', '#ex-curvas-t', '#ex-sem-t', '#ex-legenda']) $(id).textContent = '';
      graficoLinhas($('#ex-grafico'), { series: [], vazio: t('res.sem_dados') });
      return;
    }
    const info = E.info, lim = info.limiar_aprendeu ?? 0.5, base = info.passos_comparacao;
    const x = Math.round(info.passos / base);
    const sementes = Object.values(info.sementes).flat();
    const curvaDe = (c, s) => E.curvas.filter((l) => l.config === c && l.semente === s);
    const runsDe = (c) => E.runs.filter((r) => r.config === c).sort((a, b) => a.semente - b.semente);
    const primeiro = (r) => primeiroSucesso(curvaDe(r.config, r.semente), lim);
    $('#ex-d').textContent = t('res.extra.d', { base: fmtI(base), orc: fmtI(info.passos), x, sem: `${Math.min(...sementes)}–${Math.max(...sementes)}` });

    // leitura dos resultados: contagens, platô e passos calculados a partir dos dados
    const achados = [];
    for (const rs of E.resumo) {
      const rr = runsDe(rs.config), c = contarAprendizes(rr, lim);
      const aprendiz = rr.find((r) => r.sucesso >= lim), pr = aprendiz && primeiro(aprendiz);
      const naoApr = rr.filter((r) => r.sucesso < lim);
      const vars = { a: c.aprenderam, n: c.n, x, base: fmtI(base), p0: pr?.anterior != null ? fmtI(pr.anterior) : '–', p1: pr ? fmtI(pr.passo) : '–',
        plato: naoApr.length ? fmt(naoApr.reduce((acc, r) => acc + r.ret_desc, 0) / naoApr.length, 1) : '–',
        s400: aprendiz ? fmtP(aprendiz.sucesso_400mil, 0) : '–', sfim: aprendiz ? fmtP(aprendiz.sucesso, 0) : '–' };
      achados.push(el('li', { class: 'just', html: t(rs.algo === 'A2C' ? 'res.extra.a2c' : rs.n === 1 ? 'res.extra.clip' : 'res.extra.ppo', vars) }));
    }
    const ent = E.runs.filter((r) => r.sucesso < lim).map((r) => r.entropia).filter(Number.isFinite);
    achados.push(el('li', { class: 'just', html: t('res.extra.sintese', { x, emin: ent.length ? fmt(Math.min(...ent), 2) : '–', emax: ent.length ? fmt(Math.max(...ent), 2) : '–' }) }));
    $('#ex-achados').replaceChildren(...achados);

    const sw = (a) => `<i class="sw" style="background:${corAgente(a)}"></i>`;
    const th = (k, vars) => el('th', { scope: 'col', text: t(k, vars) });
    const colsR = ['res.c.config', 'res.c.sem', 'res.extra.c.passos', 'res.extra.c.aprend', 'm.ret_desc', 'm.sucesso', 'm.passos_ep', 'res.c.tempo'];
    $('#ex-tab-resumo').replaceChildren(
      el('thead', {}, [el('tr', {}, colsR.map((k) => th(k)))]),
      el('tbody', {}, E.resumo.map((r) => el('tr', {}, [
        el('th', { scope: 'row', html: `${sw(r.algo)}${r.config}` }), el('td', { text: String(r.n) }), el('td', { text: fmtI(r.passos) }),
        el('td', { text: `${r.n_aprenderam} / ${r.n}` }),
        el('td', { text: `${fmt(r.ret_desc, 1)}${Number.isFinite(r.ret_desc_ic95) ? ' ± ' + fmt(r.ret_desc_ic95, 1) : ''}` }),
        el('td', { text: fmtP(r.sucesso, 1) }), el('td', { text: fmt(r.passos_ep, 1) }), el('td', { text: fmt(r.tempo_s, 0) })]))));
    const nMax = Math.max(...E.resumo.map((r) => r.n)), ppo8 = E.resumo.find((r) => r.config === 'ppo_8envs') ?? E.resumo.find((r) => r.n > 1 && r.algo === 'PPO');
    $('#ex-lim').innerHTML = t('res.extra.lim', { n: nMax, ic: ppo8 && Number.isFinite(ppo8.ret_desc_ic95) ? fmt(ppo8.ret_desc_ic95, 1) : '–' });

    // tabela por semente
    const pb = fmtI(base);
    const colsS = [['res.c.config'], ['res.extra.c.suc_a', { p: pb }], ['res.extra.c.ret_a', { p: pb }], ['res.extra.c.suc_f'], ['res.extra.c.ret_f'], ['res.extra.c.primeiro'], ['res.extra.c.entropia'], ['res.extra.c.patamar']];
    const linhasS = [];
    for (const rs of E.resumo) for (const r of runsDe(rs.config)) {
      const pr = primeiro(r);
      linhasS.push(el('tr', {}, [el('th', { scope: 'row', html: `${sw(r.algo)}${r.config} <span class="tag cinza">${t('res.semente', { s: r.semente })}</span>` }),
        el('td', { class: r.sucesso_400mil >= lim ? 'melhor' : '', text: fmtP(r.sucesso_400mil, 0) }), el('td', { text: fmt(r.ret_desc_400mil, 1) }),
        el('td', { class: r.sucesso >= lim ? 'melhor' : '', text: fmtP(r.sucesso, 1) }), el('td', { text: fmt(r.ret_desc, 1) }),
        el('td', { text: pr ? fmtI(pr.passo) : t('res.extra.nunca') }), el('td', { text: Number.isFinite(r.entropia) ? fmt(r.entropia, 2) : '–' }),
        el('td', { text: Number.isFinite(r.frac_patamar) ? fmtP(r.frac_patamar, 0) : '–' })]));
    }
    $('#ex-sem-t').textContent = t('res.extra.sem.t', { p: pb });
    $('#ex-tab-sem').replaceChildren(el('thead', {}, [el('tr', {}, colsS.map(([k, v]) => th(k, v)))]), el('tbody', {}, linhasS));
    $('#ex-sem-nota').textContent = t('res.extra.sem.nota', { p: pb });

    // curvas por semente
    const m = estEx.metrica, pct = m === 'sucesso';
    const series = [], cfgs = [];
    for (const rs of E.resumo) {
      cfgs.push({ nome: rs.config, cor: corCfg(rs.config) });
      runsDe(rs.config).forEach((r, i) => {
        const cv = curvaDe(r.config, r.semente).sort((a, b) => a.passos - b.passos);
        series.push({ nome: `${rs.config} s${r.semente}`, cor: corCfg(rs.config), dash: TRACOS[i % TRACOS.length], x: cv.map((l) => l.passos), y: cv.map((l) => l[m]), largura: 2 });
      });
    }
    const refs = ['tabular', 'heuristica'].filter((r) => D.base?.[r]).map((r) => ({ y: D.base[r][REF_COL[m]], cor: css('--c-' + r), nome: nomeAg(r), dash: r === 'heuristica' ? [2, 3] : [6, 4] }));
    $('#ex-curvas-t').textContent = t('res.extra.curvas.t', { orc: fmtI(info.passos) });
    graficoLinhas($('#ex-grafico'), { series, refs, xRot: t('tr.eixo_passos'), yRot: t('m.' + m), yFmt: (v) => (pct ? fmtP(v, 0) : fmt(v, 0)), yMin: pct ? 0 : undefined, yMax: pct ? 1 : undefined, vazio: t('res.sem_dados') });
    legenda($('#ex-legenda'), [...cfgs, ...refs.map((r) => ({ nome: `${r.nome} (${t('res.ref')})`, cor: r.cor }))]);
    $('#ex-curvas-nota').textContent = t('res.extra.curvas.nota', { base: fmtI(base) });

    // sanidade no CartPole
    const cp = E.cartpoleResumo;
    $('#ex-cp-d').textContent = cp ? t('res.extra.cp.d', { passos: fmtI(info.cartpole_passos), n: info.cartpole_sementes?.length ?? '–' }) : '';
    if (cp) {
      $('#ex-tab-cp').replaceChildren(
        el('thead', {}, [el('tr', {}, ['res.c.algo', 'res.c.config', 'res.c.sem', 'res.extra.cp.c', 'res.extra.cp.c_mm', 'res.extra.cp.c_475', 'res.c.tempo'].map((k, i) => { const h = th(k); if (i === 1) h.style.textAlign = 'left'; return h; }))]),
        el('tbody', {}, cp.map((r) => el('tr', {}, [el('th', { scope: 'row', html: `${sw(r.algo)}${NOME_ALGO(r.algo)}` }), el('td', { class: 'hp', text: r.config }), el('td', { text: String(r.n) }),
          el('td', { text: fmt(r.ret, 1) }), el('td', { text: `${fmt(r.ret_min_sem, 1)} – ${fmt(r.ret_max_sem, 1)}` }), el('td', { text: fmtP(r.frac_ep_475, 0) }), el('td', { text: fmt(r.tempo_s, 0) })]))));
    } else $('#ex-tab-cp').replaceChildren();
  }
  $('#ex-metrica').addEventListener('change', (e) => { estEx.metrica = e.target.value; extra(); });

  // ---- dados -----------------------------------------------------------------------------------------------------------------------
  function dados() {
    const p = indice.protocolo ?? {}, v = indice.versoes ?? {}, c = indice.contagem ?? {};
    const itens = [['res.d.busca', `${c.busca ?? 0} / ${c.busca_esperado ?? '–'}`], ['res.d.final', `${c.final ?? 0} / ${c.final_esperado ?? '–'}`],
      ['res.d.orc_b', p.orcamento_busca ? fmtI(p.orcamento_busca) : '–'], ['res.d.orc_f', p.orcamento_final ? fmtI(p.orcamento_final) : '–'],
      ['res.d.sem_b', p.sementes_busca?.join(', ') ?? '–'], ['res.d.sem_f', p.sementes_final?.join(', ') ?? '–'], ['res.d.crit', p.criterio ?? '–'],
      ['res.d.v2', indice.v_estrela_t2 ? fmt(indice.v_estrela_t2, 2) : '–'],
      ['res.d.versoes', Object.keys(v).length ? Object.entries(v).map(([k, x]) => `${k} ${x}`).join(' · ') : '–'], ['res.d.estado', indice.completo ? t('res.d.final_ok') : t('res.d.incompleto')]];
    $('#dados-protocolo').replaceChildren(el('dl', { class: 'protocolo' }, itens.map(([k, x]) => el('div', {}, [el('dt', { text: t(k) }), el('dd', { text: x })]))));
    const arqs = [...(indice.arquivos ?? []), 'indice.json', ...indice.modelos.map((m) => m.arquivo), ...(indice.sarsa ? [indice.sarsa.arquivo] : [])];
    $('#dados-arquivos').replaceChildren(...arqs.map((a) => el('li', { html: `<a href="data/${a}" download>${a}</a>` })));
  }

  function desenharTudo() {
    const a = abaAtual();
    if (a === 'aba-final') final();
    else if (a === 'aba-busca') busca();
    else if (a === 'aba-curvas') curvas();
    else if (a === 'aba-hip') hipoteses();
    else if (a === 'aba-extra') extra();
    else if (a === 'aba-dados') dados();
  }
  iniciaControlesCurvas();
  final(); dados();
  I18N.onLang(() => { chipsAlgos($('#busca-algos'), estBusca.algos, busca); chipsAlgos($('#cv-algos'), estCv.algos, curvas); desenhaRefsChips(); desenharTudo(); });
  onTema(() => { desenhaRefsChips(); desenharTudo(); });
  return { ativar() { desenharTudo(); }, desativar() {} };
}
