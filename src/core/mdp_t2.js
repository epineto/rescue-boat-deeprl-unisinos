// MDP discreto do Trabalho 2 (barco de resgate 6×6), recriado de forma compacta, e Iteração de Valor.
// Serve de referência: a política ótima do T2 consultada pela célula da posição contínua é a "linha de base tabular discretizada".
// Discrete T2 MDP (compact re-implementation) and Value Iteration; mirrors RescueBoatEnv / iteracao_valor of the Python reference.
import { MAPA_PADRAO } from './env.js';

const DELTA = [[-1, 0], [1, 0], [0, 1], [0, -1]]; // (linha, coluna) para N, S, L, O

export function criarMDP({ mapa = MAPA_PADRAO, pCorr = 0.2, capacidade = 2, rPasso = -1, rDestrocos = -10, rPessoa = 50, rFinal = 100 } = {}) {
  const nl = mapa.length, nc = mapa[0].length;
  const todas = [];
  for (let i = 0; i < nl; i++) for (let j = 0; j < nc; j++) todas.push([i, j]);
  const ch = (c, k) => mapa[c[0]][c[1]] === k;
  const bloqueios = new Set(todas.filter((c) => ch(c, '#')).map((c) => c.join(',')));
  const corr = new Set(todas.filter((c) => ch(c, '~')).map((c) => c.join(',')));
  const destrocos = new Set(todas.filter((c) => ch(c, 'x')).map((c) => c.join(',')));
  const abrigo = todas.find((c) => ch(c, 'H'));
  const pessoas = todas.filter((c) => ch(c, 'P'));
  const nPes = pessoas.length;
  const celulas = todas.filter((c) => !bloqueios.has(c.join(',')));
  const idxCel = new Map(celulas.map((c, k) => [c.join(','), k]));
  const situacoes = [];
  const gera = (pref) => {
    if (pref.length === nPes) { if (pref.filter((x) => x === 1).length <= capacidade) situacoes.push(pref.slice()); return; }
    for (let v = 0; v < 3; v++) { pref.push(v); gera(pref); pref.pop(); }
  };
  gera([]);
  const idxSit = new Map(situacoes.map((e, k) => [e.join(','), k]));
  const nSit = situacoes.length, nS = celulas.length * nSit, nA = 4;
  const codifica = (cel, sit) => idxCel.get(cel.join(',')) * nSit + idxSit.get(sit.join(','));
  const decodifica = (s) => { const k = Math.floor(s / nSit); return { cel: celulas[k], sit: situacoes[s - k * nSit] }; };
  const s0 = codifica(abrigo, new Array(nPes).fill(0));

  const mover = (cel, d) => {
    const n = [cel[0] + d[0], cel[1] + d[1]];
    return n[0] < 0 || n[0] >= nl || n[1] < 0 || n[1] >= nc || bloqueios.has(n.join(',')) ? cel : n;
  };
  const resultado = (sit, nova) => {
    let r = destrocos.has(nova.join(',')) ? rDestrocos : rPasso;
    const s = sit.slice();
    pessoas.forEach((pos, i) => {
      if (pos[0] === nova[0] && pos[1] === nova[1] && s[i] === 0 && s.filter((x) => x === 1).length < capacidade) s[i] = 1;
    });
    if (nova[0] === abrigo[0] && nova[1] === abrigo[1]) for (let i = 0; i < nPes; i++) if (s[i] === 1) { s[i] = 2; r += rPessoa; }
    const fim = s.every((x) => x === 2);
    if (fim) r += rFinal;
    return { nova, sit: s, r, fim };
  };
  // P[s][a] = lista de {p, s2, r, f}, como em `env.P` do Python
  const P = [];
  for (let s = 0; s < nS; s++) {
    const { cel, sit } = decodifica(s);
    P.push([]);
    for (let a = 0; a < nA; a++) {
      if (sit.every((x) => x === 2)) { P[s].push([{ p: 1, s2: s, r: 0, f: true }]); continue; }
      const saidas = corr.has(cel.join(',')) ? [[1 - pCorr, DELTA[a]], [pCorr, DELTA[1]]] : [[1, DELTA[a]]];
      const acum = new Map();
      for (const [p, d] of saidas) {
        const res = resultado(sit, mover(cel, d)), s2 = codifica(res.nova, res.sit), k = `${s2}|${res.r}|${res.fim}`;
        if (acum.has(k)) acum.get(k).p += p; else acum.set(k, { p, s2, r: res.r, f: res.fim });
      }
      P[s].push([...acum.values()]);
    }
  }
  return { nl, nc, nS, nA, s0, celulas, situacoes, codifica, decodifica, P, abrigo, pessoas };
}

/** Iteração de Valor síncrona (a mesma ordem de somas do Python vetorizado). Devolve `{V, pi, Q, iteracoes}`. */
export function iteracaoValor(mdp, gamma = 0.99, theta = 1e-10, maxIt = 100000) {
  const { nS, nA, P } = mdp;
  const q = (V, s, a) => { let acc = 0; for (const t of P[s][a]) acc += t.p * (t.r + gamma * V[t.s2] * (t.f ? 0 : 1)); return acc; };
  let V = new Float64Array(nS), it = 0;
  for (; it < maxIt; it++) {
    const Vn = new Float64Array(nS);
    let d = 0;
    for (let s = 0; s < nS; s++) {
      let m = -Infinity;
      for (let a = 0; a < nA; a++) m = Math.max(m, q(V, s, a));
      Vn[s] = m;
      d = Math.max(d, Math.abs(m - V[s]));
    }
    V = Vn;
    if (d < theta) { it++; break; }
  }
  const pi = new Int32Array(nS), Q = new Float64Array(nS * nA);
  for (let s = 0; s < nS; s++) {
    let b = 0, bv = -Infinity;
    for (let a = 0; a < nA; a++) { const v = q(V, s, a); Q[s * nA + a] = v; if (v > bv) { bv = v; b = a; } }
    pi[s] = b;
  }
  return { V, pi, Q, iteracoes: it };
}
