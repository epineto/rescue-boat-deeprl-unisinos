// Sarsa semi-gradiente linear com tile coding (Sutton e Barto, caps. 9 e 10; Aulas 12 e 13).
// Mesma construção de características e mesma regra de atualização de reference/python/barco_continuo.py
// (classes TileCoder e SarsaSemiGradiente). O gerador de números aleatórios da exploração é o mulberry32 (o Python usa o
// do NumPy): o algoritmo é o mesmo, mas as curvas de aprendizado são estatisticamente equivalentes, não idênticas.
// Linear semi-gradient Sarsa with tile coding; same features and update rule as the Python reference.
import { BarcoContinuo, DIM_OBS } from './env.js';
import { criarRng } from './rng.js';

export const N_SITUACOES = 27;
export const N_ACOES = 4;

/** Tile coding assimétrico: `nTil` grades deslocadas de k/nTil de tile, `d` dimensões em [0, 1]. */
export class TileCoder {
  constructor(nTiles, nTil = 8) {
    this.nTiles = Int32Array.from(nTiles);
    this.nTil = nTil;
    const d = this.nTiles.length;
    this.d = d;
    this.dims = this.nTiles.map((n) => n + 1);
    this.porGrade = this.dims.reduce((a, b) => a * b, 1);
    this.tam = this.porGrade * nTil;
    this.desl = new Float64Array(nTil * d);
    for (let k = 0; k < nTil; k++) for (let i = 0; i < d; i++) this.desl[k * d + i] = ((k * (2 * i + 1)) % nTil) / nTil;
    this.mult = new Int32Array(d).fill(1);
    for (let i = d - 2; i >= 0; i--) this.mult[i] = this.mult[i + 1] * this.dims[i + 1];
  }

  /** `z`: vetor em [0, 1]^d. Escreve em `saida` os `nTil` índices ativos (já com o deslocamento de cada grade). */
  indices(z, saida = new Int32Array(this.nTil)) {
    const { d, nTil, nTiles, dims, mult, desl, porGrade } = this;
    for (let k = 0; k < nTil; k++) {
      let ix = k * porGrade;
      for (let i = 0; i < d; i++) {
        let t = Math.trunc(z[i] * nTiles[i] + desl[k * d + i]);
        if (t > dims[i] - 1) t = dims[i] - 1;
        ix += t * mult[i];
      }
      saida[k] = ix;
    }
    return saida;
  }
}

/** Índice da situação das 3 pessoas (0..26) a partir da observação: argmax de cada one-hot. */
export function situacao(obs) {
  let s = 0;
  for (let k = 0; k < 3; k++) {
    const o = 2 + 3 * k;
    const a = obs[o + 1] > obs[o] ? (obs[o + 2] > obs[o + 1] ? 2 : 1) : (obs[o + 2] > obs[o] ? 2 : 0);
    s = s * 3 + a;
  }
  return s;
}

const DEFAULTS = { nTiles: [12, 12, 4], nTil: 8, alpha: 0.1, gamma: 0.99, eps0: 0.3, eps1: 0.02, semente: 0 };

export class SarsaSemiGradiente {
  constructor(cfg = {}) {
    const c = { ...DEFAULTS, ...cfg };
    this.cfg = { nTiles: [...c.nTiles], nTil: c.nTil, alpha: c.alpha, gamma: c.gamma, eps0: c.eps0, eps1: c.eps1, semente: c.semente };
    this.tc = new TileCoder(c.nTiles, c.nTil);
    this.alpha = c.alpha / c.nTil;           // passo efetivo por característica ativa
    this.gamma = c.gamma; this.eps0 = c.eps0; this.eps1 = c.eps1;
    this.w = new Float64Array(N_SITUACOES * N_ACOES * this.tc.tam);
    this.rng = criarRng(c.semente);
    this._z = new Float64Array(3);
  }

  get tam() { return this.tc.tam; }
  /** Características de uma observação: preenche `idx` com os índices ativos e devolve o índice da situação. */
  feat(obs, idx) {
    this._z[0] = obs[0]; this._z[1] = obs[1]; this._z[2] = obs[12];
    this.tc.indices(this._z, idx);
    return situacao(obs);
  }
  /** q̂(s, a): soma dos pesos das características ativas. */
  qa(sit, a, idx) {
    const base = (sit * N_ACOES + a) * this.tc.tam;
    let s = 0;
    for (let k = 0; k < idx.length; k++) s += this.w[base + idx[k]];
    return s;
  }
  /** Escolha gulosa com desempate aleatório entre os máximos (como `rng.choice(flatnonzero(q == q.max()))`). */
  gulosa(sit, idx, rng = this.rng) {
    const q = [0, 1, 2, 3].map((a) => this.qa(sit, a, idx));
    const m = Math.max(...q);
    const emp = [];
    for (let a = 0; a < 4; a++) if (q[a] === m) emp.push(a);
    return emp.length === 1 ? emp[0] : emp[Math.floor(rng.u() * emp.length)];
  }
  /** ε-gulosa. */
  escolhe(sit, idx, eps) {
    if (this.rng.u() < eps) return Math.floor(this.rng.u() * 4);
    return this.gulosa(sit, idx);
  }
  /** Uma atualização Sarsa semi-gradiente: w ← w + α/nTil · (alvo − q̂(S, A)), alvo = R (terminal) ou R + γ·q̂(S', A'). */
  atualiza(sit, idx, a, r, terminal, sit2, idx2, a2) {
    const alvo = terminal ? r : r + this.gamma * this.qa(sit2, a2, idx2);
    const delta = alvo - this.qa(sit, a, idx);
    const base = (sit * N_ACOES + a) * this.tc.tam, inc = this.alpha * delta;
    for (let k = 0; k < idx.length; k++) this.w[base + idx[k]] += inc;
    return delta;
  }

  /**
   * Treina por `passos` passos de interação. Opções: `params` (parâmetros do ambiente), `freqAval`, `nEpAval`, `gammaAval`,
   * `aoPonto(ponto)` (chamada a cada avaliação, com a curva até o momento), `deveParar()` e `ceder()` (async; devolve o
   * controle ao laço de eventos entre as avaliações). Devolve `{curva, episodios}` (episódios de treino: [passo, retorno,
   * comprimento, sucesso, toca]).
   */
  async treinar({ passos, params = {}, freqAval, nEpAval = 20, gammaAval = 0.99, aoPonto, deveParar, ceder } = {}) {
    const env = new BarcoContinuo(params), envAval = new BarcoContinuo(params);
    const freq = freqAval || Math.max(1, Math.floor(passos / 10));
    const n = this.tc.nTil;
    let idx = new Int32Array(n), idx2 = new Int32Array(n);
    const obsBuf = new Float32Array(DIM_OBS);
    const curva = [], episodios = [];
    env.reset(this.cfg.semente);
    let sit = this.feat(env.observacao(obsBuf), idx);
    let a = this.escolhe(sit, idx, this.eps0);
    let G = 0, proxAval = freq, ultimosRet = [], ultimosSuc = [];
    for (let t = 1; t <= passos; t++) {
      const eps = Math.max(this.eps1, this.eps0 - ((this.eps0 - this.eps1) * t) / (0.5 * passos));
      const { recompensa: r, terminated: term, truncated: trunc } = env.step(a, obsBuf);
      G += r;
      const fim = term || trunc;
      if (fim) {
        episodios.push([t, G, env.passos, env.sucesso ? 1 : 0, env.toques > 0 ? 1 : 0]);
        ultimosRet.push(G); ultimosSuc.push(env.sucesso ? 1 : 0);
        if (ultimosRet.length > 50) { ultimosRet.shift(); ultimosSuc.shift(); }
        G = 0;
      }
      const sit2 = this.feat(obsBuf, idx2);
      let a2 = this.escolhe(sit2, idx2, eps);
      this.atualiza(sit, idx, a, r, term, sit2, idx2, a2);   // truncamento: continua a fazer bootstrap
      let s2 = sit2;
      if (fim) {
        env.reset();
        s2 = this.feat(env.observacao(obsBuf), idx2);
        a2 = this.escolhe(s2, idx2, eps);
      }
      [idx, idx2] = [idx2, idx];
      sit = s2; a = a2;
      if (t >= proxAval || t === passos) {
        const av = this.avaliar(nEpAval, 20000, gammaAval, envAval, params);
        const mm = (v) => (v.length ? v.reduce((x, y) => x + y, 0) / v.length : NaN);
        const ponto = { passos: t, ret: av.ret, ret_desc: av.ret_desc, sucesso: av.sucesso, passos_ep: av.passos,
          ret_treino: mm(ultimosRet), sucesso_treino: mm(ultimosSuc), episodios: episodios.length, eps };
        curva.push(ponto);
        proxAval += freq;
        if (aoPonto) aoPonto(ponto, curva);
        if (ceder) await ceder();
        if (deveParar && deveParar()) break;
      }
    }
    return { curva, episodios };
  }

  /** Avalia a política gulosa em `n` episódios (sementes `semente0 + e`); devolve as médias. O desempate usa um gerador próprio. */
  avaliar(n, semente0 = 10000, gamma = 0.99, env = new BarcoContinuo(), params = {}) {
    const pol = this.politica();
    const soma = { ret: 0, ret_desc: 0, sucesso: 0, passos: 0 };
    for (let e = 0; e < n; e++) {
      const obs = new Float32Array(DIM_OBS);
      env.reset(semente0 + e);
      pol.reiniciar(semente0 + e);
      env.observacao(obs);
      let G = 0, Gd = 0, f = 1, fim = false;
      while (!fim) {
        const o = env.step(pol.agir(env, obs), obs);
        G += o.recompensa; Gd += f * o.recompensa; f *= gamma;
        fim = o.terminated || o.truncated;
      }
      soma.ret += G; soma.ret_desc += Gd; soma.sucesso += env.sucesso ? 1 : 0; soma.passos += env.passos;
    }
    for (const k of Object.keys(soma)) soma[k] /= n;
    return soma;
  }

  /** Política gulosa `{agir(env, obs), reiniciar(semente)}` (desempate com gerador próprio, reiniciado a cada episódio). */
  politica() {
    const idx = new Int32Array(this.tc.nTil);
    let rng = criarRng(0xC0FFEE);
    return {
      reiniciar: (semente) => { rng = criarRng(((semente >>> 0) ^ 0x9e3779b9) >>> 0); },
      agir: (env, obs) => { const sit = this.feat(obs, idx); return this.gulosa(sit, idx, rng); },
    };
  }

  /** Q(s, a) das 4 ações para uma observação (para testes e inspeção). */
  qValores(obs) {
    const idx = new Int32Array(this.tc.nTil), sit = this.feat(obs, idx);
    return [0, 1, 2, 3].map((a) => this.qa(sit, a, idx));
  }
}

// ---------------------------------------------------------------------------------------------------------------------
// Serialização esparsa (pesos não nulos em float32, índices em int32; ambos em base64 little-endian).
const b64Bytes = (b64) => { const s = atob(b64), u = new Uint8Array(s.length); for (let i = 0; i < s.length; i++) u[i] = s.charCodeAt(i); return u; };
const bytesB64 = (u8) => { let s = ''; for (let i = 0; i < u8.length; i += 0x8000) s += String.fromCharCode(...u8.subarray(i, i + 0x8000)); return btoa(s); };

/** Agente → JSON esparso (mesmo formato escrito por tools/exportar_para_lab.py). */
export function sarsaParaJson(ag, extra = {}) {
  const idx = [], val = [];
  for (let i = 0; i < ag.w.length; i++) if (ag.w[i] !== 0) { idx.push(i); val.push(ag.w[i]); }
  return { tipo: 'sarsa', config: ag.cfg, n_pesos: ag.w.length, n_nao_nulos: idx.length,
    indices_b64: bytesB64(new Uint8Array(Int32Array.from(idx).buffer)), valores_b64: bytesB64(new Uint8Array(Float32Array.from(val).buffer)), ...extra };
}

/** JSON esparso → agente pronto para inferência (e para continuar o treino). */
export function sarsaDeJson(j) {
  const ag = new SarsaSemiGradiente(j.config);
  if (ag.w.length !== j.n_pesos) throw new Error(`tamanho de pesos inconsistente: ${ag.w.length} vs ${j.n_pesos}`);
  const ib = b64Bytes(j.indices_b64), vb = b64Bytes(j.valores_b64);
  const idx = new Int32Array(ib.buffer.slice(ib.byteOffset, ib.byteOffset + ib.length));
  const val = new Float32Array(vb.buffer.slice(vb.byteOffset, vb.byteOffset + vb.length));
  if (idx.length !== j.n_nao_nulos || val.length !== j.n_nao_nulos) throw new Error('número de pesos não nulos inconsistente');
  for (let k = 0; k < idx.length; k++) ag.w[idx[k]] = val[k];
  return ag;
}
