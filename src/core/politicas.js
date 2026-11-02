// Políticas (agentes) do laboratório. Cada uma expõe `{id, agir(env, obs) → ação, reiniciar?(semente)}`.
//  - aleatoria      ações uniformes (gerador mulberry32 próprio)
//  - heuristica     caminho mais curto (BFS) que evita blocos e destroços; ignora a correnteza
//  - tabular        política ótima do T2 (Iteração de Valor) consultada pela célula da posição contínua
//  - mlp            rede treinada pelo stable-baselines3 (DQN, PPO ou A2C), argmax determinístico
//  - sarsa          Sarsa semi-gradiente com tile coding (pré-treinado em Python ou treinado no navegador)
import { criarRng } from './rng.js';
import { criarMLP } from './mlp.js';
import { criarMDP, iteracaoValor } from './mdp_t2.js';
import { sarsaDeJson } from './sarsa.js';

const DELTA = [[-1, 0], [1, 0], [0, 1], [0, -1]];

export function politicaAleatoria() {
  let rng = criarRng(0);
  return { id: 'aleatoria', reiniciar: (semente) => { rng = criarRng((semente >>> 0) ^ 0x5bd1e995); }, agir: () => Math.floor(rng.u() * 4) };
}

/** Heurística de caminho mais curto (porte de PoliticaHeuristica do Python). */
export function politicaHeuristica(env) {
  const { nl, nc } = env;
  const livre = new Set();
  for (let i = 0; i < nl; i++) for (let j = 0; j < nc; j++) if (env.mapa[i][j] !== '#' && env.mapa[i][j] !== 'x') livre.add(i * nc + j);
  const chave = (c) => c[0] * nc + c[1];
  const alvos = { H: [Math.floor(env.abrigo[1]), Math.floor(env.abrigo[0])] };
  env.pessoas.forEach(([px, py], k) => { alvos['P' + k] = [Math.floor(py), Math.floor(px)]; });
  const bfs = (alvo) => {
    const d = new Map([[chave(alvo), 0]]), fila = [alvo];
    for (let q = 0; q < fila.length; q++) {
      const c = fila[q];
      for (const [dl, dc] of DELTA) {
        const n = [c[0] + dl, c[1] + dc];
        if (n[0] < 0 || n[0] >= nl || n[1] < 0 || n[1] >= nc) continue;
        if (livre.has(chave(n)) && !d.has(chave(n))) { d.set(chave(n), d.get(chave(c)) + 1); fila.push(n); }
      }
    }
    return d;
  };
  const dist = {};
  for (const k of Object.keys(alvos)) dist[k] = bfs(alvos[k]);
  return {
    id: 'heuristica',
    agir(e) {
      const cel = e.celula, kc = chave(cel);
      const espera = [];
      e.status.forEach((s, k) => { if (s === 0) espera.push('P' + k); });
      let ch;
      if (e.aboard >= e.cap || !espera.length) ch = 'H';
      else { // pessoa ilhada mais próxima (empate: a primeira na ordem de leitura, como o min do Python)
        ch = espera[0];
        for (const c of espera) if ((dist[c].get(kc) ?? 1e9) < (dist[ch].get(kc) ?? 1e9)) ch = c;
      }
      const d = dist[ch], alvo = alvos[ch];
      let tc;
      if ((cel[0] === alvo[0] && cel[1] === alvo[1]) || !d.has(kc)) tc = alvo;
      else {
        let melhor = null;
        for (const [dl, dc] of DELTA) {
          const n = [cel[0] + dl, cel[1] + dc];
          if (n[0] < 0 || n[0] >= nl || n[1] < 0 || n[1] >= nc || !d.has(chave(n))) continue;
          if (melhor === null || d.get(chave(n)) < d.get(chave(melhor))) melhor = n;
        }
        tc = melhor;
      }
      const ex = tc[1] + 0.5 - e.x, ey = tc[0] + 0.5 - e.y;
      if (Math.abs(ex) > Math.abs(ey)) return ex > 0 ? 2 : 3;
      return ey > 0 ? 1 : 0;
    },
  };
}

let _tabularCache = null;
/** Linha de base tabular discretizada. A Iteração de Valor (832 estados) é calculada uma vez e guardada. */
export function politicaTabular(gamma = 0.99) {
  if (!_tabularCache || _tabularCache.gamma !== gamma) {
    const mdp = criarMDP();
    _tabularCache = { gamma, mdp, ...iteracaoValor(mdp, gamma) };
  }
  const { mdp, pi } = _tabularCache;
  return { id: 'tabular', v0: _tabularCache.V[mdp.s0], agir: (e) => pi[mdp.codifica(e.celula, e.status)] };
}
export const valorOtimoT2 = (gamma = 0.99) => { politicaTabular(gamma); return _tabularCache.V[_tabularCache.mdp.s0]; };
export const politicaOtimaT2 = (gamma = 0.99) => { politicaTabular(gamma); return _tabularCache.pi; };

export function politicaMLP(modelo) {
  const m = criarMLP(modelo);
  return { id: modelo.id ?? modelo.algo, agir: (e, obs) => m.agir(obs), saidas: m.saidas };
}

export function politicaSarsa(agenteOuJson) {
  const ag = agenteOuJson.w ? agenteOuJson : sarsaDeJson(agenteOuJson);
  return { id: 'sarsa', ...ag.politica(), agente: ag };
}

/** Fábrica a partir de uma especificação serializável (usada também dentro de Web Workers). */
export function criarPolitica(espec, env) {
  switch (espec.tipo) {
    case 'aleatoria': return politicaAleatoria();
    case 'heuristica': return politicaHeuristica(env);
    case 'tabular': return politicaTabular();
    case 'mlp': return politicaMLP(espec.dados);
    case 'sarsa': return politicaSarsa(espec.dados);
    default: throw new Error(`tipo de política desconhecido: ${espec.tipo}`);
  }
}
