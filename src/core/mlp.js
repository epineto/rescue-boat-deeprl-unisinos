// Inferência de MLPs exportados do stable-baselines3 (DQN, PPO, A2C). Sem treino: só a passagem direta.
// MLP inference for models exported from stable-baselines3 (see tools/exportar_para_lab.py).
//
// Formato JSON do modelo: {algo, saida: 'q'|'logits', ativacao: 'relu'|'tanh',
//   camadas: [{entrada, saida, W: [[...]] (saida × entrada), b: [...]}], ...metadados}
// A ação é o argmax da saída (política determinística): valores Q no DQN; logits da política no PPO e no A2C.

const ATIVACOES = { relu: (x) => (x > 0 ? x : 0), tanh: Math.tanh };

export function criarMLP(modelo) {
  const act = ATIVACOES[modelo.ativacao];
  if (!act) throw new Error(`ativação desconhecida: ${modelo.ativacao}`);
  const camadas = modelo.camadas.map((c) => {
    const W = new Float64Array(c.saida * c.entrada);
    c.W.forEach((linha, i) => { if (linha.length !== c.entrada) throw new Error('forma de W inconsistente'); W.set(linha, i * c.entrada); });
    if (c.b.length !== c.saida) throw new Error('forma de b inconsistente');
    return { n: c.entrada, m: c.saida, W, b: Float64Array.from(c.b) };
  });
  for (let i = 1; i < camadas.length; i++) if (camadas[i].n !== camadas[i - 1].m) throw new Error('camadas incompatíveis');
  const bufs = camadas.map((c) => new Float64Array(c.m));

  /** Saídas da última camada (Q ou logits) para uma observação. Devolve uma cópia. */
  function saidas(obs) {
    let x = obs;
    camadas.forEach((c, l) => {
      const y = bufs[l], ultima = l === camadas.length - 1;
      for (let i = 0; i < c.m; i++) {
        let s = c.b[i];
        const off = i * c.n;
        for (let j = 0; j < c.n; j++) s += c.W[off + j] * x[j];
        y[i] = ultima ? s : act(s);
      }
      x = y;
    });
    return Float64Array.from(x);
  }
  function agir(obs) {
    const q = saidas(obs);
    let b = 0;
    for (let i = 1; i < q.length; i++) if (q[i] > q[b]) b = i;
    return b;
  }
  return { saidas, agir, nParametros: camadas.reduce((a, c) => a + c.n * c.m + c.m, 0), camadas: camadas.length };
}
