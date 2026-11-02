// Web Worker: treina o Sarsa semi-gradiente (tile coding) sem travar a página. Mensagens:
//   entrada  {tipo:'treinar', id, cfg, passos, animar}  |  {tipo:'parar', id}
//   saída    {tipo:'ponto', id, ponto, passosTotal}  |  {tipo:'fim', id, curva, modelo, metricas, ms, interrompido}  |  {tipo:'erro', id, mensagem}
import { SarsaSemiGradiente, sarsaParaJson } from './core/sarsa.js';
import { avaliar, resumo } from './core/avaliar.js';
import { politicaSarsa } from './core/politicas.js';

const parar = new Set();
const ceder = (ms = 0) => new Promise((r) => setTimeout(r, ms));

self.onmessage = async (ev) => {
  const m = ev.data;
  if (m.tipo === 'parar') { parar.add(m.id); return; }
  if (m.tipo !== 'treinar') return;
  try {
    const t0 = performance.now();
    const ag = new SarsaSemiGradiente(m.cfg);
    const nPontos = 40, freq = Math.max(1, Math.floor(m.passos / nPontos));
    const { curva } = await ag.treinar({
      passos: m.passos, freqAval: freq, nEpAval: 20, gammaAval: m.cfg.gamma,
      aoPonto: (ponto) => self.postMessage({ tipo: 'ponto', id: m.id, ponto, passosTotal: m.passos }),
      deveParar: () => parar.has(m.id),
      ceder: () => ceder(m.animar ? 70 : 0),
    });
    const pol = politicaSarsa(ag);
    const metricas = resumo(avaliar(pol, 100, 10000, m.cfg.gamma));
    self.postMessage({ tipo: 'fim', id: m.id, curva, modelo: sarsaParaJson(ag), metricas, ms: performance.now() - t0, interrompido: parar.has(m.id) });
  } catch (e) {
    self.postMessage({ tipo: 'erro', id: m.id, mensagem: String((e && e.message) || e) });
  } finally { parar.delete(m.id); }
};
