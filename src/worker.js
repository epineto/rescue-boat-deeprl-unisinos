// Web Worker: comparação de agentes em N episódios sem travar a página. Mensagens:
//   entrada  {tipo:'comparar', id, agentes, n, semente0, gamma}  |  {tipo:'parar', id}
//   saída    {tipo:'progresso', id, feitos, n}  |  {tipo:'fim', id, resultado}  |  {tipo:'erro', id, mensagem}
import { compararAgentes } from './core/comparar.js';

const parar = new Set();
const ceder = () => new Promise((r) => setTimeout(r, 0)); // deixa o worker receber a mensagem "parar" entre blocos

self.onmessage = async (ev) => {
  const m = ev.data;
  if (m.tipo === 'parar') { parar.add(m.id); return; }
  if (m.tipo !== 'comparar') return;
  try {
    const resultado = await compararAgentes({
      agentes: m.agentes, n: m.n, semente0: m.semente0, gamma: m.gamma,
      aoProgresso: (feitos, n) => self.postMessage({ tipo: 'progresso', id: m.id, feitos, n }),
      deveParar: () => parar.has(m.id), ceder,
    });
    self.postMessage({ tipo: 'fim', id: m.id, resultado });
  } catch (e) {
    self.postMessage({ tipo: 'erro', id: m.id, mensagem: String((e && e.message) || e) });
  } finally { parar.delete(m.id); }
};
