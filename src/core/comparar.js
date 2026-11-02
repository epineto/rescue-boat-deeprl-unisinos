// Comparação de agentes em N episódios (mesmas sementes para todos). Roda no Web Worker (src/worker.js) e nos testes.
// Multi-agent comparison over N episodes with shared seeds; runs in the Web Worker and in tests.
import { BarcoContinuo } from './env.js';
import { criarPolitica } from './politicas.js';
import { rodarEpisodio, resumo } from './avaliar.js';

/**
 * @param {{agentes: {id: string, espec: object}[], n: number, semente0: number, gamma: number, params?: object,
 *   aoProgresso?: (feitos: number, n: number) => void, deveParar?: () => boolean, ceder?: () => Promise<void>}} o
 * @returns {Promise<{n: number, parcial: boolean, resultados: Object<string, {vetores: object, resumo: object}>}>}
 */
export async function compararAgentes({ agentes, n, semente0, gamma = 0.99, params = {}, aoProgresso, deveParar, ceder }) {
  const env = new BarcoContinuo(params);
  const pols = agentes.map((a) => ({ id: a.id, pol: criarPolitica(a.espec, env) }));
  const vet = Object.fromEntries(pols.map((p) => [p.id, { ret: [], ret_desc: [], sucesso: [], passos: [], toca: [], entregues: [] }]));
  const bloco = Math.max(1, Math.floor(n / 50));
  let feitos = 0;
  for (let e = 0; e < n; e++) {
    for (const p of pols) {
      const r = rodarEpisodio(env, p.pol, semente0 + e, gamma);
      for (const k of Object.keys(vet[p.id])) vet[p.id][k].push(r[k]);
    }
    feitos = e + 1;
    if (feitos % bloco === 0 || feitos === n) {
      aoProgresso?.(feitos, n);
      if (ceder) await ceder();
      if (deveParar?.()) break;
    }
  }
  return { n: feitos, parcial: feitos < n, resultados: Object.fromEntries(pols.map((p) => [p.id, { vetores: vet[p.id], resumo: resumo(vet[p.id]) }])) };
}
