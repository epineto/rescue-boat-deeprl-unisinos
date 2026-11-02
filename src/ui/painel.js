// Painel de episódio: um ambiente, uma política e um canvas. Usado em Políticas, Treinar e na demonstração da introdução.
// Episode panel: one environment, one policy, one canvas.
import { BarcoContinuo, DIM_OBS } from '../core/env.js';
import { prepararCanvas, aoRedimensionar } from './util.js';
import { desenharCena } from './desenho.js';

export class Painel {
  /** @param {HTMLCanvasElement} canvas  @param {{agir, reiniciar?}} politica  @param {{gamma?, raios?, trilha?}} opcoes */
  constructor(canvas, politica, { gamma = 0.99, raios = false, trilha = true, params = {} } = {}) {
    this.canvas = canvas; this.pol = politica; this.gamma = gamma; this.raios = raios; this.mostraTrilha = trilha;
    this.env = new BarcoContinuo(params);
    this.obs = new Float32Array(DIM_OBS);
    this.fase = 0;
    this.iniciar(1);
    aoRedimensionar(canvas, () => this.desenhar());
  }

  trocarPolitica(pol) { this.pol = pol; this.iniciar(this.semente); }

  iniciar(semente) {
    this.semente = semente;
    this.env.reset(semente);
    this.env.observacao(this.obs);
    this.pol.reiniciar?.(semente);
    this.trilha = [[this.env.x, this.env.y]];
    this.ret = 0; this.retDesc = 0; this.fator = 1; this.acao = null; this.ultimaRec = 0;
    this.terminado = false; this.sucesso = false; this.registrado = false;
    this.desenhar();
  }

  /** Avança um passo. Devolve `false` se o episódio já havia terminado. */
  passo() {
    if (this.terminado) return false;
    const a = this.pol.agir(this.env, this.obs);
    const o = this.env.step(a, this.obs);
    this.acao = a; this.ultimaRec = o.recompensa;
    this.ret += o.recompensa; this.retDesc += this.fator * o.recompensa; this.fator *= this.gamma;
    this.trilha.push([this.env.x, this.env.y]);
    if (o.terminated || o.truncated) { this.terminado = true; this.sucesso = o.terminated; }
    return true;
  }

  get metricas() {
    const e = this.env;
    return { passos: e.passos, ret: this.ret, retDesc: this.retDesc, entregues: e.entregues, toques: e.toques, terminado: this.terminado, sucesso: this.sucesso, inten: e.inten };
  }

  desenhar(fase = this.fase) {
    this.fase = fase;
    const { w, ctx } = prepararCanvas(this.canvas);
    desenharCena(ctx, w, this.env, { trilha: this.mostraTrilha ? this.trilha : null, raios: this.raios, fase, acao: this.acao });
  }
}
