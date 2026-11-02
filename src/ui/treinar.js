// Seção #treinar: Sarsa semi-gradiente com tile coding treinado no navegador (Web Worker), curva ao vivo e política aprendida em ação.
// Browser training of linear semi-gradient Sarsa with tile coding; live learning curve and replay of the learned policy.
import { politicaSarsa } from '../core/politicas.js';
import { sarsaDeJson } from '../core/sarsa.js';
import { lerNumero, fmtInt } from '../core/numeros.js';
import { $, el, reduzMovimento } from './util.js';
import { Painel } from './painel.js';
import { graficoLinhas } from './graficos.js';
import { corAgente, css } from './util.js';
import * as I18N from './i18n.js';
import { onTema } from './tema.js';

const { t, fmt, fmtP } = I18N;
const PRESETS = { '8x8x3_0.3': [8, 8, 3, 0.3], '6x6x3_0.3': [6, 6, 3, 0.3], '12x12x4_0.3': [12, 12, 4, 0.3], '12x12x4_0.1': [12, 12, 4, 0.1] };

/** Lê e valida os campos do formulário. Devolve `{cfg, passos}` ou `{erro}`. */
export function lerParametros(v) {
  const num = (k, nome, lo, hi, inteiro) => {
    const x = lerNumero(v[k]);
    if (!Number.isFinite(x) || x < lo || x > hi || (inteiro && !Number.isInteger(x))) throw new Error(t('tr.erro', { nome, lo: I18N.fmtM(lo), hi: I18N.fmtM(hi) }));
    return x;
  };
  try {
    const nTiles = [num('nx', 'x', 1, 24, true), num('ny', 'y', 1, 24, true), num('ni', 'I', 1, 8, true)];
    const nTil = num('ntil', t('tr.ntil'), 1, 16, true);
    const cfg = { nTiles, nTil, alpha: num('alfa', 'α', 0.001, 2), eps0: num('eps0', t('tr.eps0'), 0, 1), eps1: num('eps1', t('tr.eps1'), 0, 1),
      gamma: num('gama', 'γ', 0.5, 1), semente: num('sem', t('tr.semente'), 0, 2 ** 31, true) };
    const passos = num('passos', t('tr.passos'), 10000, 3000000, true);
    return { cfg, passos };
  } catch (e) { return { erro: e.message }; }
}

export function iniciarTreinar(S) {
  const base = S.dados.base, ids = ['nx', 'ny', 'ni', 'ntil', 'alfa', 'eps0', 'eps1', 'gama', 'sem', 'passos'];
  const campo = (k) => $(`#tr-${k}`);
  const valores = () => Object.fromEntries(ids.map((k) => [k, campo(k).value]));
  let worker = null, jobId = 0, curva = [], passosTotal = 300000, painel = null, modelo = null, tocando = false, raf = 0, ult = 0, acc = 0, ativa = false, semEp = 10001;

  // ---- formulário ----------------------------------------------------------------------------------------------------
  const fmtCampo = (x) => String(x).replace('.', I18N.getLang() === 'pt' ? ',' : '.');
  function aplicarPreset(p) {
    const [nx, ny, ni, a] = PRESETS[p];
    campo('nx').value = nx; campo('ny').value = ny; campo('ni').value = ni; campo('alfa').value = fmtCampo(a);
    grade();
  }
  function grade() {
    const r = lerParametros(valores());
    if (r.erro) { $('#tr-grade-d').textContent = ''; return; }
    const tam = r.cfg.nTiles.map((n) => n + 1).reduce((a, b) => a * b, 1) * r.cfg.nTil;
    $('#tr-grade-d').textContent = t('tr.grade_d', { a: r.cfg.nTil, t: fmtInt(tam, I18N.getLang()), w: fmtInt(tam * 27 * 4, I18N.getLang()) });
  }
  $('#tr-preset').addEventListener('change', (e) => { if (e.target.value !== 'custom') aplicarPreset(e.target.value); });
  for (const k of ['nx', 'ny', 'ni', 'alfa']) campo(k).addEventListener('input', () => {
    const cur = [campo('nx').value, campo('ny').value, campo('ni').value].map(Number), a = lerNumero(campo('alfa').value);
    const hit = Object.entries(PRESETS).find(([, p]) => p[0] === cur[0] && p[1] === cur[1] && p[2] === cur[2] && p[3] === a && campo('ntil').value === '8');
    $('#tr-preset').value = hit ? hit[0] : 'custom'; grade();
  });
  campo('ntil').addEventListener('input', grade);

  // ---- gráficos ----------------------------------------------------------------------------------------------------------
  function desenharGraficos() {
    const refs = [];
    if (base?.tabular) refs.push({ y: base.tabular.ret_desc, cor: css('--c-tabular'), nome: t('ag.tabular') });
    if (base?.heuristica) refs.push({ y: base.heuristica.ret_desc, cor: css('--c-heuristica'), nome: t('ag.heuristica'), dash: [2, 3] });
    const x = curva.map((p) => p.passos), cor = corAgente('SARSA');
    graficoLinhas($('#tr-g-ret'), { series: [{ nome: t('tr.leg_aval'), cor, x, y: curva.map((p) => p.ret_desc), pontos: curva.length < 45 }], refs, xMin: 0, xMax: passosTotal,
      xRot: t('tr.eixo_passos'), yRot: t('m.ret_desc'), yFmt: (v) => fmt(v, 0), vazio: t('tr.sem_dados') });
    graficoLinhas($('#tr-g-suc'), { series: [
      { nome: t('tr.leg_aval'), cor, x, y: curva.map((p) => p.sucesso), pontos: curva.length < 45 },
      { nome: t('tr.leg_treino'), cor: css('--c-ppo'), x, y: curva.map((p) => p.sucesso_treino), dash: [5, 3], largura: 1.6 }],
    xMin: 0, xMax: passosTotal, yMin: 0, yMax: 1, xRot: t('tr.eixo_passos'), yRot: t('m.sucesso'), yFmt: (v) => fmtP(v, 0), vazio: t('tr.sem_dados') });
  }

  // ---- execução ----------------------------------------------------------------------------------------------------------
  function emExecucao(sim) { $('#tr-go').disabled = sim; $('#tr-parar').disabled = !sim; for (const k of ids) campo(k).disabled = sim; $('#tr-preset').disabled = sim; }
  function treinar() {
    const r = lerParametros(valores());
    const er = $('#tr-erro');
    if (r.erro) { er.hidden = false; er.textContent = r.erro; return; }
    er.hidden = true;
    curva = []; passosTotal = r.passos; modelo = null; $('#tr-res-cartao').hidden = true; pararReplay();
    $('#tr-barra').style.width = '0%';
    desenharGraficos();
    emExecucao(true);
    $('#tr-status').textContent = t('tr.treinando', { p: 0, n: fmtInt(r.passos, I18N.getLang()), e: 0, eps: fmt(r.cfg.eps0, 2) });
    worker?.terminate();
    worker = new Worker(new URL('../worker-treino.js', import.meta.url), { type: 'module' });
    const id = ++jobId;
    worker.onmessage = (ev) => {
      const m = ev.data;
      if (m.id !== id) return;
      if (m.tipo === 'ponto') {
        curva.push(m.ponto);
        $('#tr-barra').style.width = `${(m.ponto.passos / m.passosTotal) * 100}%`;
        $('#tr-status').textContent = t('tr.treinando', { p: fmtInt(m.ponto.passos, I18N.getLang()), n: fmtInt(m.passosTotal, I18N.getLang()), e: fmtInt(m.ponto.episodios, I18N.getLang()), eps: fmt(m.ponto.eps, 2) });
        desenharGraficos();
      } else if (m.tipo === 'fim') { fim(m, r); worker?.terminate(); worker = null; }
      else if (m.tipo === 'erro') { er.hidden = false; er.textContent = t('erro.worker', { m: m.mensagem }); emExecucao(false); worker?.terminate(); worker = null; }
    };
    worker.onerror = (e) => { er.hidden = false; er.textContent = t('erro.worker', { m: e.message }); emExecucao(false); };
    worker.postMessage({ tipo: 'treinar', id, cfg: r.cfg, passos: r.passos, animar: $('#tr-anima').checked && !reduzMovimento() });
    $('#tr-parar').onclick = () => worker?.postMessage({ tipo: 'parar', id });
  }
  $('#tr-go').addEventListener('click', treinar);

  function fim(m, r) {
    emExecucao(false);
    modelo = m.modelo; ultimoFim = { m, r };
    const ult = m.curva[m.curva.length - 1];
    $('#tr-barra').style.width = m.interrompido ? $('#tr-barra').style.width : '100%';
    $('#tr-status').textContent = t(m.interrompido ? 'tr.interrompido' : 'tr.concluido', { p: fmtInt(ult?.passos ?? 0, I18N.getLang()), s: fmt(m.ms / 1000, 1) });
    S.catalogo.definirUsuario(m.modelo, { cfg: r.cfg });
    S.politicas?.novoAgente();
    tabelaResultado();
    $('#tr-res-cartao').hidden = false;
    montarReplay();
  }
  let ultimoFim = null;
  function tabelaResultado() {
    if (!ultimoFim) return;
    const { m, r } = ultimoFim, a = m.metricas;
    const linhas = [[t('tr.res.este', { s: r.cfg.semente }), a, true]];
    const sarsaExp = S.dados.finalResumo.find((x) => x.agente === 'SARSA');
    if (sarsaExp) linhas.push([t('tr.res.exp', { n: sarsaExp.n_sementes }), { ret_desc: sarsaExp.ret_desc, ret_desc_ic95: sarsaExp.ret_desc_ic95, sucesso: sarsaExp.sucesso, sucesso_ic95: sarsaExp.sucesso_ic95, passos: sarsaExp.passos_ep, toca: sarsaExp.toca_destrocos }, false]);
    for (const k of ['tabular', 'heuristica']) if (base?.[k]) linhas.push([t('ag.' + k), { ret_desc: base[k].ret_desc, ret_desc_ic95: base[k].ret_desc_ic95, sucesso: base[k].sucesso, sucesso_ic95: base[k].sucesso_ic95, passos: base[k].passos, toca: base[k].toca }, false]);
    const cab = ['agente', 'ret_desc', 'sucesso', 'passos', 'toca'];
    $('#tr-res').replaceChildren(
      el('thead', {}, [el('tr', {}, cab.map((c) => el('th', { scope: 'col', text: c === 'agente' ? t('pol.vivo.agente') : t('pol.cmp.c_' + c) })))]),
      el('tbody', {}, linhas.map(([nome, v, destaque]) => el('tr', { class: destaque ? 'escolhida' : '' }, [el('th', { scope: 'row', text: nome }),
        el('td', { text: `${fmt(v.ret_desc, 1)} ± ${fmt(v.ret_desc_ic95 || 0, 1)}` }), el('td', { text: `${fmtP(v.sucesso, 0)} ± ${fmt((v.sucesso_ic95 || 0) * 100, 0)}` }),
        el('td', { text: fmt(v.passos, 1) }), el('td', { text: fmtP(v.toca, 0) })]))));
    $('#tr-res-nota').textContent = t('tr.res.nota');
  }

  // ---- replay da política aprendida ------------------------------------------------------------------------------------------
  function montarReplay() {
    const pol = politicaSarsa(sarsaDeJson(modelo));
    if (!painel) painel = new Painel($('#tr-canvas'), pol, { raios: false });
    else painel.trocarPolitica(pol);
    painel.gamma = ultimoFim?.r.cfg.gamma ?? 0.99;
    reiniciarReplay(semEp);
  }
  function infoEp() {
    if (!painel) return;
    const m = painel.metricas;
    $('#tr-ep-info').textContent = m.terminado ? (m.sucesso ? t('pol.fim.ok', { n: m.passos }) : t('pol.fim.tempo')) + ' · ' + t('tr.ep_ret', { r: fmt(m.ret, 0) }) : t('tr.ep_info', { p: m.passos, e: m.entregues, r: fmt(m.ret, 0) });
    $('#tr-canvas').setAttribute('aria-label', t('tr.canvas_aria') + ' ' + $('#tr-ep-info').textContent);
  }
  function reiniciarReplay(s) { semEp = s; $('#tr-ep-sem').value = String(s); painel?.iniciar(s); infoEp(); }
  function pararReplay() { tocando = false; cancelAnimationFrame(raf); }
  function quadro(ts) {
    if (!tocando || !ativa) return;
    acc += Math.min(0.25, (ts - ult) / 1000) * 12; ult = ts;
    let n = Math.min(4, Math.floor(acc)); acc -= Math.floor(acc);
    while (n-- > 0 && painel.passo()) { /* avança */ }
    painel.desenhar(); infoEp();
    if (painel.terminado) { tocando = false; return; }
    raf = requestAnimationFrame(quadro);
  }
  $('#tr-play').addEventListener('click', () => {
    if (!painel) return;
    reiniciarReplay(Math.max(0, Math.floor(lerNumero($('#tr-ep-sem').value) || semEp)));
    tocando = true; ult = performance.now(); acc = 0; raf = requestAnimationFrame(quadro);
  });
  $('#tr-prox').addEventListener('click', () => { pararReplay(); reiniciarReplay(semEp + 1); tocando = true; ult = performance.now(); acc = 0; raf = requestAnimationFrame(quadro); });

  // ---- ciclo de vida ---------------------------------------------------------------------------------------------------------
  // campos decimais acompanham a convenção do idioma (vírgula em PT, ponto em EN)
  const ajustaDecimais = () => { for (const k of ['alfa', 'eps0', 'eps1', 'gama']) { const x = lerNumero(campo(k).value); if (Number.isFinite(x)) campo(k).value = fmtCampo(x); } };
  aplicarPreset('8x8x3_0.3');
  ajustaDecimais();
  desenharGraficos();
  I18N.onLang(() => {
    ajustaDecimais();
    grade(); desenharGraficos(); tabelaResultado();
    if (!worker && !curva.length) $('#tr-status').textContent = t('tr.status0');
  });
  onTema(() => { desenharGraficos(); painel?.desenhar(); });
  return {
    ativar() { ativa = true; desenharGraficos(); painel?.desenhar(); },
    desativar() { ativa = false; pararReplay(); },
  };
}
