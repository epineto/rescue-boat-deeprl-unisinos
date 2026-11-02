// Seção #politicas: assistir a vários agentes no mesmo episódio (mesma semente de ruído), métricas ao vivo e comparação em N episódios.
// Watch several agents on the same episode (same noise seed), live metrics and an N-episode comparison in a Web Worker.
import { BarcoContinuo } from '../core/env.js';
import { lerNumero } from '../core/numeros.js';
import { media, ic95 } from '../core/estat.js';
import { $, $$, el } from './util.js';
import { Painel } from './painel.js';
import { graficoBarras } from './graficos.js';
import * as I18N from './i18n.js';
import { onTema } from './tema.js';

const { t, fmt, fmtI, fmtP } = I18N;
const PADRAO = ['heuristica', 'DQN', 'PPO', 'A2C'];
const MAX_PAINEIS = 6;

export function iniciarPoliticas(S) {
  const cat = S.catalogo;
  let selecionados = PADRAO.filter((id) => cat.ids().includes(id));
  if (selecionados.length < 2) selecionados = cat.ids().slice(0, 4);
  let paineis = new Map();      // id → { painel, fig, estado, dds }
  let semente = 10001, fechado = false, tocando = false, raf = 0, ult = 0, acc = 0, ativa = false, proximoTimer = 0;
  const vivo = new Map();       // id → acumuladores
  const vistos = new Map();     // id → Set de sementes já contabilizadas
  const grade = $('#pol-paineis');

  // ---- seleção de agentes ----------------------------------------------------------------------------------------
  function chips() {
    const box = $('#pol-agentes');
    const cheio = selecionados.length >= MAX_PAINEIS;
    box.replaceChildren(...cat.ids().map((id) => {
      const marcado = selecionados.includes(id);
      const lab = el('label', { class: 'chip-check', ...(cheio && !marcado ? { 'aria-disabled': 'true' } : {}) }, [
        el('input', { type: 'checkbox', 'data-id': id, ...(marcado ? { checked: '' } : {}), ...(cheio && !marcado ? { disabled: '' } : {}) }),
        el('i', { class: 'sw', style: `background:${cat.cor(id)}` }), el('span', { text: cat.rotulo(id) })]);
      lab.querySelector('input').addEventListener('change', (e) => {
        selecionados = e.target.checked ? [...selecionados, id] : selecionados.filter((x) => x !== id);
        chips(); construirPaineis();
      });
      return lab;
    }));
    const sel = $('#pol-sementes-modelo');
    sel.replaceChildren(...['DQN', 'PPO', 'A2C'].filter((a) => cat.sementes(a).length > 1).map((a) => {
      const id = `pol-sem-${a}`;
      const campo = el('div', { class: 'campo' }, [el('label', { for: id, text: t('pol.sem_modelo', { a }) }),
        el('select', { id }, cat.sementes(a).map((m) => el('option', { value: m.semente, ...(m.semente === cat.sel[a] ? { selected: '' } : {}), text: `s${m.semente}${m.semente === cat.melhor?.[a] ? ' ' + t('pol.melhor_sem') : ''}` })))]);
      campo.querySelector('select').addEventListener('change', (e) => { cat.sel[a] = Number(e.target.value); const p = paineis.get(a); if (p) { p.painel.trocarPolitica(cat.politica(a, p.painel.env)); p.estadoVivo = null; rotulosPaineis(); pintar(); } });
      return campo;
    }));
  }

  // ---- painéis ---------------------------------------------------------------------------------------------------
  function construirPaineis() {
    cancelAnimationFrame(raf); tocando = false; atualizaBotao();
    paineis.clear();
    const figs = selecionados.filter((id) => cat.ids().includes(id)).map((id) => {
      const canvas = el('canvas', { role: 'img' });
      const dds = {};
      const dl = el('dl', { class: 'p-met' }, ['passos', 'ret', 'entregues', 'toques'].map((k) => el('div', {}, [el('dt', { 'data-k': k }), (dds[k] = el('dd'))])));
      const estado = el('span', { class: 'p-estado' });
      const titulo = el('span', { class: 'p-titulo' });
      const fig = el('figure', { class: 'painel', style: `border-top-color:${cat.cor(id)}` }, [el('figcaption', {}, [titulo, estado]), canvas, dl]);
      const painel = new Painel(canvas, cat.politica(id, new BarcoContinuo()), { raios: false });
      paineis.set(id, { painel, fig, estado, titulo, dds, canvas });
      return fig;
    });
    grade.replaceChildren(...figs);
    grade.style.setProperty('--cols', String(Math.max(1, Math.min(figs.length, 4))));
    reiniciarEpisodio(semente);
    rotulosPaineis();
  }
  function rotulosPaineis() {
    for (const [id, p] of paineis) {
      p.titulo.textContent = cat.modelos[id] ? cat.rotulo(id, cat.sel[id]) : cat.rotulo(id);
      p.fig.title = cat.descricao(id);
      $$('dt', p.fig).forEach((d) => { d.textContent = t('pol.m.' + d.dataset.k); });
    }
    pintar();
  }
  function pintar() {
    for (const [id, p] of paineis) {
      const m = p.painel.metricas;
      p.dds.passos.textContent = String(m.passos);
      p.dds.ret.textContent = fmt(m.ret, 0);
      p.dds.entregues.textContent = `${m.entregues}/3`;
      p.dds.toques.textContent = String(m.toques);
      p.estado.className = 'p-estado' + (m.terminado ? (m.sucesso ? ' ok' : ' ruim') : '');
      p.estado.textContent = m.terminado ? (m.sucesso ? t('pol.fim.ok', { n: m.passos }) : t('pol.fim.tempo')) : (m.passos ? t('pol.andamento') : t('pol.pronto'));
      p.canvas.setAttribute('aria-label', t('pol.canvas_aria', { a: p.titulo.textContent, p: m.passos, e: m.entregues, r: fmt(m.ret, 0) }));
      p.painel.desenhar();
    }
  }

  function reiniciarEpisodio(s) {
    semente = s;
    $('#pol-semente').value = String(s);
    for (const p of paineis.values()) p.painel.iniciar(s);
    acc = 0; fechado = false;
    pintar();
  }

  // ---- laço de reprodução --------------------------------------------------------------------------------------------
  function registrar(id, p) {
    const m = p.painel.metricas;
    const set = vistos.get(id) ?? new Set(); vistos.set(id, set);
    if (set.has(p.painel.semente)) return;
    set.add(p.painel.semente);
    const a = vivo.get(id) ?? { n: 0, ret: 0, retd: 0, suc: 0, passos: 0, toca: 0 }; vivo.set(id, a);
    a.n++; a.ret += m.ret; a.retd += m.retDesc; a.suc += m.sucesso ? 1 : 0; a.passos += m.passos; a.toca += m.toques > 0 ? 1 : 0;
  }
  function tabelaVivo() {
    const tab = $('#pol-vivo');
    const cab = ['ep', 'ret', 'retd', 'suc', 'passos', 'toca'];
    const linhas = [...paineis.keys()].map((id) => {
      const a = vivo.get(id);
      return el('tr', {}, [el('th', { scope: 'row', html: `<i class="sw" style="background:${cat.cor(id)}"></i>${cat.modelos[id] ? cat.rotulo(id, cat.sel[id]) : cat.rotulo(id)}` }),
        ...(a ? [fmtI(a.n), fmt(a.ret / a.n, 1), fmt(a.retd / a.n, 1), fmtP(a.suc / a.n, 0), fmt(a.passos / a.n, 1), fmtP(a.toca / a.n, 0)] : ['0', '–', '–', '–', '–', '–']).map((x) => el('td', { text: x }))]);
    });
    tab.replaceChildren(el('thead', {}, [el('tr', {}, [el('th', { scope: 'col', text: t('pol.vivo.agente') }), ...cab.map((c) => el('th', { scope: 'col', text: t('pol.vivo.' + c) }))])]), el('tbody', {}, linhas));
  }

  function aoTerminarTodos() {
    const falas = [];
    for (const [id, p] of paineis) { registrar(id, p); const m = p.painel.metricas; falas.push(`${p.titulo.textContent}: ${m.sucesso ? t('pol.fim.ok', { n: m.passos }) : t('pol.fim.tempo')}`); }
    $('#pol-live').textContent = t('pol.concluido', { s: semente }) + ' ' + falas.join('. ');
    tabelaVivo();
    if ($('#pol-continuo').checked && tocando) {
      clearTimeout(proximoTimer);
      proximoTimer = setTimeout(() => { if (tocando && ativa) reiniciarEpisodio(semente + 1); }, 1100);
    } else { tocando = false; atualizaBotao(); }
  }
  function passoTodos() {
    let algum = false;
    for (const p of paineis.values()) algum = p.painel.passo() || algum;
    return algum;
  }
  const todosFim = () => paineis.size > 0 && [...paineis.values()].every((p) => p.painel.terminado);
  function verificaFim() { if (!fechado && todosFim()) { fechado = true; aoTerminarTodos(); } }
  function quadro(ts) {
    if (!tocando || !ativa) return;
    const dt = Math.min(0.25, (ts - ult) / 1000); ult = ts;
    acc += dt * Number($('#pol-vel').value);
    let n = Math.min(5, Math.floor(acc)); acc -= Math.floor(acc);
    while (n-- > 0 && !todosFim()) passoTodos();
    pintar();
    verificaFim();
    if (tocando) raf = requestAnimationFrame(quadro);
  }
  function atualizaBotao() {
    const b = $('#pol-play');
    b.setAttribute('aria-pressed', String(tocando));
    b.textContent = t(tocando ? 'pol.pausar' : 'pol.play');
  }
  function play() {
    if (tocando) { tocando = false; cancelAnimationFrame(raf); atualizaBotao(); return; }
    if (todosFim()) reiniciarEpisodio(semente);
    tocando = true; atualizaBotao(); ult = performance.now(); acc = 0;
    raf = requestAnimationFrame(quadro);
  }

  $('#pol-play').addEventListener('click', play);
  $('#pol-passo').addEventListener('click', () => {
    tocando = false; cancelAnimationFrame(raf); atualizaBotao();
    if (todosFim()) { reiniciarEpisodio(semente); return; }
    passoTodos(); pintar(); verificaFim();
  });
  $('#pol-reiniciar').addEventListener('click', () => { clearTimeout(proximoTimer); reiniciarEpisodio(semente); });
  $('#pol-proxima').addEventListener('click', () => { clearTimeout(proximoTimer); reiniciarEpisodio(semente + 1); });
  $('#pol-semente').addEventListener('change', (e) => { const v = lerNumero(e.target.value); reiniciarEpisodio(Number.isFinite(v) ? Math.max(0, Math.floor(v)) : semente); });
  $('#pol-vel').addEventListener('input', (e) => { $('#pol-vel-v').textContent = e.target.value; });
  $('#pol-zerar')?.addEventListener('click', () => { vivo.clear(); vistos.clear(); tabelaVivo(); });

  // ---- comparação em N episódios (Web Worker) ------------------------------------------------------------------------------
  let worker = null, idJob = 0;
  const NOMES = ['ret_desc', 'ret', 'sucesso', 'passos', 'toca'];
  function encerrarWorker() { worker?.terminate(); worker = null; }
  function comparar() {
    const n = Math.max(1, Math.min(2000, Math.floor(lerNumero($('#cmp-n').value) || 100)));
    const s0 = Math.max(0, Math.floor(lerNumero($('#cmp-s0').value) || 10000));
    $('#cmp-n').value = String(n); $('#cmp-s0').value = String(s0);
    const todas = $('#cmp-todas').checked;
    const agentes = [];
    for (const id of cat.ids()) {
      if (cat.modelos[id] && todas) for (const m of cat.sementes(id)) agentes.push({ id: `${id}#${m.semente}`, base: id, semente: m.semente, espec: cat.espec(id, m.semente) });
      else agentes.push({ id, base: id, semente: cat.modelos[id] ? cat.sel[id] : null, espec: cat.espec(id) });
    }
    encerrarWorker();
    worker = new Worker(new URL('../worker.js', import.meta.url), { type: 'module' });
    const id = ++idJob;
    $('#cmp-go').disabled = true; $('#cmp-parar').disabled = false; $('#cmp-prog').hidden = false; $('#cmp-res').hidden = true; $('#cmp-barra').style.width = '0%';
    $('#cmp-prog-t').textContent = t('pol.cmp.exec', { f: 0, n });
    worker.onmessage = (ev) => {
      const m = ev.data;
      if (m.id !== id) return;
      if (m.tipo === 'progresso') { $('#cmp-barra').style.width = `${(m.feitos / m.n) * 100}%`; $('#cmp-prog-t').textContent = t('pol.cmp.exec', { f: m.feitos, n: m.n }); }
      else if (m.tipo === 'fim') { fimComparar(m.resultado, agentes, s0, todas); encerrarWorker(); }
      else if (m.tipo === 'erro') { $('#cmp-prog-t').textContent = t('erro.worker', { m: m.mensagem }); fimBotoes(); encerrarWorker(); }
    };
    worker.onerror = (e) => { $('#cmp-prog-t').textContent = t('erro.worker', { m: e.message }); fimBotoes(); encerrarWorker(); };
    worker.postMessage({ tipo: 'comparar', id, agentes: agentes.map((a) => ({ id: a.id, espec: a.espec })), n, semente0: s0, gamma: 0.99 });
    $('#cmp-parar').onclick = () => worker?.postMessage({ tipo: 'parar', id });
  }
  function fimBotoes() { $('#cmp-go').disabled = false; $('#cmp-parar').disabled = true; }
  let ultimoResultado = null;
  function fimComparar(res, agentes, s0, todas) {
    fimBotoes();
    $('#cmp-prog').hidden = true;
    ultimoResultado = { res, agentes, s0, todas };
    mostrarComparacao();
  }
  function mostrarComparacao() {
    if (!ultimoResultado) return;
    const { res, agentes, s0, todas } = ultimoResultado;
    const linhas = agentes.map((a) => {
      const r = res.resultados[a.id].resumo;
      return { id: a.id, base: a.base, rotulo: a.semente !== null && cat.modelos[a.base] ? `${a.base} (s${a.semente})` : cat.rotulo(a.base), cor: cat.cor(a.base), ...r, n: res.n };
    });
    if (todas) { // média das sementes de treino de cada algoritmo (IC95% entre sementes)
      for (const algo of ['DQN', 'PPO', 'A2C']) {
        const sub = linhas.filter((l) => l.base === algo);
        if (sub.length < 2) continue;
        const agg = { id: algo + '*', base: algo, rotulo: t('pol.cmp.media_sem', { a: algo, n: sub.length }), cor: cat.cor(algo), n: res.n, sementes: sub.length };
        for (const k of NOMES) { const v = sub.map((l) => l[k]); agg[k] = media(v); agg[k + '_ic95'] = ic95(v); }
        linhas.splice(linhas.indexOf(sub[sub.length - 1]) + 1, 0, agg);
      }
    }
    $('#cmp-res').hidden = false;
    const melhorRet = Math.max(...linhas.filter((l) => !l.sementes).map((l) => l.ret_desc)), melhorSuc = Math.max(...linhas.filter((l) => !l.sementes).map((l) => l.sucesso));
    const col = [['ret_desc', 1], ['ret', 1], ['sucesso', 'p'], ['passos', 1], ['toca', 'p']];
    const cab = el('thead', {}, [el('tr', {}, [el('th', { scope: 'col', text: t('pol.vivo.agente') }), ...col.map(([k]) => el('th', { scope: 'col', text: t('pol.cmp.c_' + k) }))])]);
    const corpo = el('tbody', {}, linhas.map((l) => el('tr', { class: l.sementes ? 'sub' : '' }, [
      el('th', { scope: 'row', html: `<i class="sw" style="background:${l.cor}"></i>${l.rotulo}` }),
      ...col.map(([k, f]) => {
        const v = l[k], ic = l[k + '_ic95'];
        const txt = f === 'p' ? `${fmtP(v, 1)} ± ${fmt((ic || 0) * 100, 1)}` : `${fmt(v, 1)} ± ${fmt(ic || 0, 1)}`;
        const melhor = !l.sementes && ((k === 'ret_desc' && Math.abs(v - melhorRet) < 1e-9) || (k === 'sucesso' && Math.abs(v - melhorSuc) < 1e-9));
        return el('td', { class: melhor ? 'melhor' : '', text: txt });
      })])));
    $('#cmp-tab').replaceChildren(cab, corpo);
    const itens = (k, escala = 1) => linhas.map((l) => ({ rotulo: l.rotulo, valor: l[k] * escala, ic: Number.isFinite(l[k + '_ic95']) ? l[k + '_ic95'] * escala : 0, cor: l.cor }));
    graficoBarras($('#cmp-g-ret'), { itens: itens('ret_desc'), xFmt: (v) => fmt(v, 0) });
    graficoBarras($('#cmp-g-suc'), { itens: itens('sucesso'), xMin: 0, xMax: 1, xFmt: (v) => fmtP(v, 0) });
    $('#cmp-nota').textContent = t('pol.cmp.nota', { n: res.n, a: s0, b: s0 + res.n - 1 }) + (res.parcial ? ' ' + t('pol.cmp.parcial') : '') + (todas ? ' ' + t('pol.cmp.nota_sem') : '');
  }
  $('#cmp-go').addEventListener('click', comparar);
  $('#cmp-todas-l').hidden = !Object.values(cat.modelos).some((m) => m.length > 1);

  // ---- ciclo de vida ---------------------------------------------------------------------------------------------------
  chips();
  construirPaineis();
  tabelaVivo();
  I18N.onLang(() => { chips(); atualizaBotao(); rotulosPaineis(); tabelaVivo(); mostrarComparacao(); });
  onTema(() => { pintar(); mostrarComparacao(); });
  return {
    ativar() { ativa = true; pintar(); },
    desativar() { ativa = false; tocando = false; cancelAnimationFrame(raf); atualizaBotao(); },
    /** Chamado quando um Sarsa é treinado na seção Treinar: inclui o agente no catálogo e selecioná-lo. */
    novoAgente() { chips(); if (!selecionados.includes('sarsa_user')) selecionados = [...selecionados.slice(0, MAX_PAINEIS - 1), 'sarsa_user']; chips(); construirPaineis(); },
  };
}
