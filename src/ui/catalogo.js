// Catálogo de agentes do laboratório: linhas de base, redes do SB3 (por semente) e Sarsa (pré-treinado e treinado no navegador).
// Agent catalog: baselines, SB3 networks (per training seed) and Sarsa (pre-trained and browser-trained).
import { criarPolitica } from '../core/politicas.js';
import { corAgente } from './util.js';
import { t } from './i18n.js';

const ORDEM = ['aleatoria', 'tabular', 'heuristica', 'sarsa_pre', 'sarsa_user', 'DQN', 'PPO', 'A2C'];

export class Catalogo {
  constructor(dados) {
    this.dados = dados;
    this.modelos = { DQN: [], PPO: [], A2C: [] };
    for (const m of dados.modelos) if (this.modelos[m.algo]) this.modelos[m.algo].push(m);
    for (const a of Object.keys(this.modelos)) this.modelos[a].sort((x, y) => x.semente - y.semente);
    this.sel = {};   // semente escolhida por algoritmo (padrão: a de maior retorno descontado nas runs finais)
    for (const [a, ms] of Object.entries(this.modelos)) {
      if (!ms.length) continue;
      const melhor = ms.reduce((b, m) => ((m.metricas?.ret_desc ?? -Infinity) > (b.metricas?.ret_desc ?? -Infinity) ? m : b), ms[0]);
      this.sel[a] = melhor.semente; this.melhor = { ...(this.melhor ?? {}), [a]: melhor.semente };
    }
    this.usuario = null;
  }

  /** Ids disponíveis, na ordem de apresentação. */
  ids() {
    return ORDEM.filter((id) => {
      if (id === 'sarsa_pre') return !!this.dados.sarsa;
      if (id === 'sarsa_user') return !!this.usuario;
      if (this.modelos[id]) return this.modelos[id].length > 0;
      return true;
    });
  }
  familia(id) { return this.modelos[id] ? id : id.startsWith('sarsa') ? 'SARSA' : id; }
  cor(id) { return corAgente(this.familia(id)); }
  rotulo(id, semente) {
    if (this.modelos[id]) return semente === undefined ? id : `${id} (s${semente})`;
    return t('ag.' + id);
  }
  descricao(id) { return t('ag.' + id + '_d'); }
  tipo(id) { return this.modelos[id] ? 'mlp' : id.startsWith('sarsa') ? 'sarsa' : id; }
  sementes(algo) { return this.modelos[algo] ?? []; }

  /** Especificação serializável (para o Web Worker) do agente `id` (e semente, para as redes). */
  espec(id, semente = this.sel[id]) {
    if (this.modelos[id]) {
      const m = this.modelos[id].find((x) => x.semente === semente) ?? this.modelos[id][0];
      return { tipo: 'mlp', dados: m.dados };
    }
    if (id === 'sarsa_pre') return { tipo: 'sarsa', dados: this.dados.sarsa.dados };
    if (id === 'sarsa_user') return { tipo: 'sarsa', dados: this.usuario.modelo };
    return { tipo: id };
  }
  politica(id, env, semente) { return criarPolitica(this.espec(id, semente), env); }

  definirUsuario(modelo, meta = {}) { this.usuario = { modelo, ...meta }; }
}
