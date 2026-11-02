// Seção #ambiente: controle manual do barco (teclado e botões), com posição contínua, ocupação, medidor da correnteza,
// recompensa acumulada e a observação de 13 números. / Manual control of the boat with live state readout.
import { BarcoContinuo, DIM_OBS } from '../core/env.js';
import { lerNumero } from '../core/numeros.js';
import { $, $$, el, prepararCanvas, aoRedimensionar, reduzMovimento } from './util.js';
import { desenharCena } from './desenho.js';
import { spark } from './graficos.js';
import * as I18N from './i18n.js';
import { onTema } from './tema.js';

const { t, fmt, fmtM } = I18N;
const TECLAS = { ArrowUp: 0, ArrowDown: 1, ArrowRight: 2, ArrowLeft: 3, n: 0, s: 1, l: 2, e: 2, o: 3, w: 3 };

export function iniciarAmbiente() {
  const env = new BarcoContinuo();
  const canvas = $('#amb-canvas');
  const obs = new Float32Array(DIM_OBS);
  const gamma = 0.99;
  let semente = 1, ativa = false, raf = 0, fase = 0, ult = 0;
  let ret = 0, retDesc = 0, fator = 1, ultimaRec = null, acao = null, trilha = [], histI = [], fim = null;

  const hud = $('#amb-hud');
  const campos = ['passo', 'pos', 'cel', 'bordo', 'ret', 'retd', 'rec', 'toques'];
  hud.replaceChildren(...campos.map((k) => el('div', {}, [el('dt', { 'data-k': k }), el('dd', { id: `hud-${k}` })])));
  const tbody = $('#amb-obs tbody');
  tbody.replaceChildren(...Array.from({ length: DIM_OBS }, (_, i) => el('tr', {}, [el('th', { scope: 'row', text: String(i) }), el('td', { id: `obs-n${i}` }), el('td', { id: `obs-v${i}` }), el('td', { class: 'o-barra', html: `<div class="o-b"><i id="obs-b${i}"></i></div>` })])));
  $('#amb-obs tbody').querySelectorAll('td:nth-child(3)').forEach((c) => { c.style.textAlign = 'right'; });

  function nomesObs() {
    const nomes = ['x / 6', 'y / 6'];
    for (let k = 0; k < 3; k++) for (let s = 0; s < 3; s++) nomes.push(t('amb.obs.pessoa', { k: k + 1, s: t('estado.' + s) }));
    nomes.push(t('amb.obs.ocupacao'), t('amb.obs.inten'));
    return nomes;
  }
  function rotulos() {
    campos.forEach((k) => { $(`[data-k="${k}"]`, hud).textContent = t('amb.hud.' + k); });
    nomesObs().forEach((n, i) => { $(`#obs-n${i}`).textContent = n; });
    $('#amb-corr-d').innerHTML = t('amb.corr.d', { v: fmt(env.p.corr_vmax, 1), m: fmt(env.p.corr_media, 1), s: fmt(env.p.corr_sigma, 2) });
    $('#amb-med-media').title = t('amb.corr.media_t', { m: fmt(env.p.corr_media, 2) });
  }

  function mensagemFim() {
    const m = $('#amb-fim');
    m.className = 'msg-fim' + (fim ? (fim.sucesso ? ' ok' : ' ruim') : '');
    if (!fim) { m.textContent = ''; return; }
    m.textContent = fim.sucesso ? t('amb.fim.ok', { n: env.passos, r: fmt(ret, 0) }) : t('amb.fim.tempo', { e: env.entregues, r: fmt(ret, 0) });
  }

  function atualizar() {
    const o = env.observacao(obs);
    $('#hud-passo').textContent = `${env.passos} / ${env.p.max_passos}`;
    $('#hud-pos').textContent = `(${fmt(env.x, 2)}; ${fmt(env.y, 2)})`;
    $('#hud-cel').textContent = `(${env.celula[0]}, ${env.celula[1]})`;
    $('#hud-bordo').textContent = `${env.aboard} / ${env.cap}`;
    $('#hud-ret').textContent = fmt(ret, 0);
    $('#hud-retd').textContent = fmt(retDesc, 1);
    $('#hud-rec').textContent = ultimaRec === null ? '–' : fmt(ultimaRec, 0);
    $('#hud-toques').textContent = String(env.toques);
    $('#amb-pessoas').replaceChildren(...env.status.map((s, k) => el('li', { class: `st-${s}`, text: `P${k + 1}: ${t('estado.' + s)}` })));
    for (let i = 0; i < DIM_OBS; i++) { $(`#obs-v${i}`).textContent = fmt(o[i], 3); $(`#obs-b${i}`).style.width = `${o[i] * 100}%`; }
    const I = env.inten;
    $('#amb-med-barra').style.left = `${I * 100}%`;
    $('#amb-med-media').style.left = `calc(${env.p.corr_media * 100}% - 1px)`;
    $('#amb-med-val').textContent = `I = ${fmt(I, 2)}`;
    $('#amb-medidor').setAttribute('aria-valuenow', I.toFixed(3));
    $('#amb-medidor').setAttribute('aria-valuetext', t('amb.corr.valor', { v: fmt(I, 2), a: fmt(env.p.corr_vmax * I, 2) }));
    spark($('#amb-spark'), histI, { rotulo: t('amb.corr.spark') });
    mensagemFim();
    desenhar();
  }

  function desenhar() {
    const { w, ctx } = prepararCanvas(canvas);
    desenharCena(ctx, w, env, { trilha: $('#amb-trilha').checked ? trilha : null, raios: $('#amb-raios').checked, fase, acao });
  }

  function reiniciar(s = semente) {
    semente = s;
    $('#amb-semente').value = String(s);
    env.reset(s);
    ret = 0; retDesc = 0; fator = 1; ultimaRec = null; acao = null; fim = null;
    trilha = [[env.x, env.y]]; histI = [env.inten];
    atualizar();
  }

  function mover(a) {
    if (fim) return;
    const o = env.step(a, obs);
    acao = a; ultimaRec = o.recompensa; ret += o.recompensa; retDesc += fator * o.recompensa; fator *= gamma;
    trilha.push([env.x, env.y]); histI.push(env.inten);
    if (o.terminated || o.truncated) fim = { sucesso: o.terminated };
    atualizar();
  }

  $$('.dpad [data-acao]').forEach((b) => b.addEventListener('click', () => { mover(Number(b.dataset.acao)); }));
  document.addEventListener('keydown', (ev) => {
    if (!ativa || ev.ctrlKey || ev.metaKey || ev.altKey) return;
    const alvo = ev.target, edit = alvo.closest?.('input, textarea, select');
    if (edit) return;
    const k = ev.key.length === 1 ? ev.key.toLowerCase() : ev.key;
    if (k === 'r') { reiniciar(semente); return; }
    if (!(k in TECLAS)) return;
    const seta = k.startsWith('Arrow');
    if (seta && !(alvo === canvas || alvo.closest?.('.dpad') || alvo === document.body)) return; // não rouba as setas de outros controles
    ev.preventDefault();
    mover(TECLAS[k]);
  });
  $('#amb-reiniciar').addEventListener('click', () => reiniciar(semente));
  $('#amb-proxima').addEventListener('click', () => reiniciar(semente + 1));
  $('#amb-semente').addEventListener('change', (e) => { const v = lerNumero(e.target.value); reiniciar(Number.isFinite(v) ? Math.max(0, Math.floor(v)) : semente); });
  $('#amb-raios').addEventListener('change', desenhar);
  $('#amb-trilha').addEventListener('change', desenhar);
  aoRedimensionar(canvas, desenhar);
  aoRedimensionar($('#amb-spark'), () => spark($('#amb-spark'), histI, { rotulo: t('amb.corr.spark') }));
  onTema(() => { desenhar(); spark($('#amb-spark'), histI, { rotulo: t('amb.corr.spark') }); });

  function quadro(ts) {
    if (!ativa) return;
    fase = (fase + ((ts - ult) / 1000) * (0.25 + 0.9 * env.inten)) % 1;
    ult = ts;
    desenhar();
    raf = requestAnimationFrame(quadro);
  }
  const t0 = () => { cancelAnimationFrame(raf); if (ativa && !reduzMovimento()) { ult = performance.now(); raf = requestAnimationFrame(quadro); } };

  rotulos();
  I18N.onLang(() => { rotulos(); atualizar(); });
  reiniciar(1);
  return {
    ativar() { ativa = true; t0(); atualizar(); },
    desativar() { ativa = false; cancelAnimationFrame(raf); },
    reiniciar,
  };
}
