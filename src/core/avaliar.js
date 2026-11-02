// Avaliação de políticas: episódios com sementes fixas (as mesmas para todos os agentes ⇒ comparação pareada).
// Policy evaluation with fixed seeds, mirroring `avaliar` / `resumo` of the Python reference.
import { BarcoContinuo, DIM_OBS } from './env.js';
import { media, ic95 } from './estat.js';

export const SEMENTE_AVAL = 10000;

/** Executa um episódio completo e devolve as métricas. `aoPasso(env, info)` é opcional. */
export function rodarEpisodio(env, pol, semente, gamma = 0.99, aoPasso) {
  const obs = env.reset(semente);
  pol.reiniciar?.(semente);
  let G = 0, Gd = 0, f = 1, fim = false;
  while (!fim) {
    const a = pol.agir(env, obs);
    const o = env.step(a, obs);
    G += o.recompensa; Gd += f * o.recompensa; f *= gamma;
    fim = o.terminated || o.truncated;
    aoPasso?.(env, { acao: a, recompensa: o.recompensa });
  }
  return { ret: G, ret_desc: Gd, sucesso: env.sucesso ? 1 : 0, passos: env.passos, toca: env.toques > 0 ? 1 : 0, entregues: env.entregues };
}

/** `n` episódios com sementes `semente0 + e`. Devolve vetores por episódio. */
export function avaliar(pol, n = 100, semente0 = SEMENTE_AVAL, gamma = 0.99, params = {}, env = new BarcoContinuo(params)) {
  const out = { ret: [], ret_desc: [], sucesso: [], passos: [], toca: [], entregues: [] };
  for (let e = 0; e < n; e++) {
    const r = rodarEpisodio(env, pol, semente0 + e, gamma);
    for (const k of Object.keys(out)) out[k].push(r[k]);
  }
  return out;
}

/** Médias e IC95% entre episódios de cada métrica de `avaliar`. */
export function resumo(av) {
  const o = {};
  for (const [k, v] of Object.entries(av)) { o[k] = media(v); o[k + '_ic95'] = ic95(v); }
  return o;
}
export { DIM_OBS };
