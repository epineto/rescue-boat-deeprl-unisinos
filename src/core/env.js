// Ambiente contínuo do barco de resgate (RescueBoatContEnv) em JavaScript puro.
// Mesma dinâmica, mesma ordem de operações e mesmo gerador de números aleatórios de reference/python/barco_continuo.py;
// os testes comparam passo a passo com uma trajetória de referência gerada pelo Python.
// Continuous rescue-boat environment in plain JS: same dynamics, operation order and RNG as the Python reference.
import { criarRng } from './rng.js';

export const MAPA_PADRAO = ['P..P..', '.#..#P', '~~~~~~', '..xx..', '.#..#.', '..H...'];
export const NOMES_ACOES = ['N', 'S', 'L', 'O'];
const DX = [0, 0, 1, -1];
const DY = [-1, 1, 0, 0];
export const DIM_OBS = 13;

/** Parâmetros calibrados no piloto (resultados/piloto_calibracao.md); idênticos a PARAMS_PADRAO do Python. */
export const PARAMS_PADRAO = Object.freeze({
  passo: 0.7, sigma_mov: 0.05, r_embarque: 0.8, r_abrigo: 0.9, r_destrocos: 0.5,
  corr_vmax: 0.5, corr_sigma: 0.05, corr_media: 0.5, corr_theta: 0.1, corr_sigma_i: 0.12, corr_sigma_0: 0.25,
  max_passos: 100, capacidade: 2, r_passo: -1.0, r_destrocos_rec: -10.0, r_pessoa: 50.0, r_final: 100.0,
});

const EPS = 1e-9;
const recorta = (v, hi) => (v < 0 ? 0 : v > hi ? hi : v);

export class BarcoContinuo {
  /** @param {object} params sobrescrita parcial de PARAMS_PADRAO  @param {string[]} mapa linhas do mapa ('.', '#', '~', 'x', 'P', 'H') */
  constructor(params = {}, mapa = MAPA_PADRAO) {
    const desconhecidos = Object.keys(params).filter((k) => !(k in PARAMS_PADRAO));
    if (desconhecidos.length) throw new TypeError(`parâmetros desconhecidos: ${desconhecidos.join(', ')}`);
    this.p = { ...PARAMS_PADRAO, ...params };
    this.mapa = mapa.map(String);
    this.nl = this.mapa.length;
    this.nc = this.mapa[0].length;
    this.bloq = this.mapa.map((l) => [...l].map((c) => c === '#'));
    this.destrocos = []; this.pessoas = []; this.abrigo = null;
    const linhasCorr = [];
    for (let i = 0; i < this.nl; i++) {
      for (let j = 0; j < this.nc; j++) {
        const c = this.mapa[i][j];
        if (c === '~') linhasCorr.push(i);
        else if (c === 'x') this.destrocos.push([j + 0.5, i + 0.5]);
        else if (c === 'P') this.pessoas.push([j + 0.5, i + 0.5]);
        else if (c === 'H') this.abrigo = [j + 0.5, i + 0.5];
      }
    }
    this.faixa = linhasCorr.length ? [Math.min(...linhasCorr), Math.max(...linhasCorr) + 1] : [0, 0];
    this.nPes = this.pessoas.length;
    this.cap = this.p.capacidade;
    this.limX = this.nc - EPS;
    this.limY = this.nl - EPS;
    this.rng = criarRng(0);
    this._semeado = false;
    this.x = this.abrigo[0]; this.y = this.abrigo[1];
    this.status = new Array(this.nPes).fill(0);
    this.aboard = 0; this.inten = 0.5; this.passos = 0; this.toques = 0;
  }

  /** Reinicia o episódio. Com `semente`, reinicia o gerador (ruído idêntico para a mesma semente); sem ela, continua o fluxo. */
  reset(semente) {
    if (semente !== undefined && semente !== null) { this.rng = criarRng(semente); this._semeado = true; }
    else if (!this._semeado) { this.rng = criarRng(0); this._semeado = true; }
    const p = this.p;
    [this.x, this.y] = this.abrigo;
    this.status = new Array(this.nPes).fill(0);
    this.aboard = 0;
    const [z0] = this.rng.par();
    this.inten = Math.min(1, Math.max(0, p.corr_media + p.corr_sigma_0 * z0));
    this.passos = 0; this.toques = 0;
    return this.observacao();
  }

  /** Célula discreta [linha, coluna] que contém a posição atual. */
  get celula() { return [Math.floor(this.y), Math.floor(this.x)]; }
  get sucesso() { return this.status.every((s) => s === 2); }
  get entregues() { return this.status.filter((s) => s === 2).length; }

  /** Observação (13 componentes em [0, 1], float32): x/nc, y/nl, situação das pessoas em one-hot, a bordo/capacidade, intensidade. */
  observacao(saida = new Float32Array(DIM_OBS)) {
    saida[0] = this.x / this.nc;
    saida[1] = this.y / this.nl;
    for (let k = 0; k < this.nPes; k++) {
      const s = this.status[k];
      saida[2 + 3 * k] = s === 0 ? 1 : 0;
      saida[3 + 3 * k] = s === 1 ? 1 : 0;
      saida[4 + 3 * k] = s === 2 ? 1 : 0;
    }
    saida[2 + 3 * this.nPes] = this.aboard / this.cap;
    saida[3 + 3 * this.nPes] = this.inten;
    return saida;
  }

  /** Um passo. Devolve `{obs, recompensa, terminated, truncated}`; `obs` é reutilizada se `saidaObs` for passada. */
  step(acao, saidaObs) {
    const p = this.p, a = acao | 0;
    const [zx, zy] = this.rng.par();
    const [zc, zi] = this.rng.par();
    const x0 = this.x, y0 = this.y;
    // fase 1: movimento comandado mais ruído; colisão com bloco ⇒ não se move
    let x1 = recorta(x0 + p.passo * DX[a] + p.sigma_mov * zx, this.limX);
    let y1 = recorta(y0 + p.passo * DY[a] + p.sigma_mov * zy, this.limY);
    if (this.bloq[Math.floor(y1)][Math.floor(x1)]) { x1 = x0; y1 = y0; }
    // fase 2: correnteza (arrasto para o sul na faixa)
    let y2 = y1;
    if (this.faixa[0] <= y1 && y1 < this.faixa[1]) {
      y2 = recorta(y1 + p.corr_vmax * this.inten + p.corr_sigma * zc, this.limY);
      if (this.bloq[Math.floor(y2)][Math.floor(x1)]) y2 = y1;
    }
    // intensidade: AR(1) recortado em [0, 1]
    const i = this.inten + p.corr_theta * (p.corr_media - this.inten) + p.corr_sigma_i * zi;
    this.inten = i < 0 ? 0 : i > 1 ? 1 : i;
    this.x = x1; this.y = y2;
    // eventos
    let r = p.r_passo;
    const rd2 = p.r_destrocos ** 2;
    for (const [cx, cy] of this.destrocos) {
      if ((x1 - cx) ** 2 + (y2 - cy) ** 2 <= rd2 && (x0 - cx) ** 2 + (y0 - cy) ** 2 > rd2) { r = p.r_destrocos_rec; this.toques++; break; }
    }
    const re2 = p.r_embarque ** 2;
    for (let k = 0; k < this.nPes; k++) {
      if (this.status[k] === 0 && this.aboard < this.cap) {
        const [px, py] = this.pessoas[k];
        if ((x1 - px) ** 2 + (y2 - py) ** 2 <= re2) { this.status[k] = 1; this.aboard++; }
      }
    }
    if (this.aboard > 0 && (x1 - this.abrigo[0]) ** 2 + (y2 - this.abrigo[1]) ** 2 <= p.r_abrigo ** 2) {
      for (let k = 0; k < this.nPes; k++) if (this.status[k] === 1) { this.status[k] = 2; r += p.r_pessoa; }
      this.aboard = 0;
    }
    const terminated = this.status.every((s) => s === 2);
    if (terminated) r += p.r_final;
    this.passos++;
    const truncated = !terminated && this.passos >= p.max_passos;
    return { obs: this.observacao(saidaObs), recompensa: r, terminated, truncated };
  }
}
