import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { BarcoContinuo } from '../src/core/env.js';
import { TileCoder, SarsaSemiGradiente, sarsaParaJson, sarsaDeJson, situacao } from '../src/core/sarsa.js';
import { criarMLP } from '../src/core/mlp.js';

const ref = JSON.parse(readFileSync(new URL('./reference.json', import.meta.url), 'utf8'));

test('tile coding: índices idênticos aos do Python em 5 configurações × 200 pontos (inclusive nas bordas 0 e 1)', () => {
  for (const c of ref.tile) {
    const tc = new TileCoder(c.n_tiles, c.n_til);
    assert.equal(tc.tam, c.tam);
    c.z.forEach((z, i) => assert.deepEqual([...tc.indices(z)], c.indices[i], `config ${c.n_tiles} ponto ${i}`));
  }
});

test('tile coding: cada entrada ativa exatamente nTil características, todas dentro do vetor de pesos', () => {
  const tc = new TileCoder([8, 8, 3], 8);
  for (let k = 0; k < 100; k++) {
    const idx = tc.indices([Math.random(), Math.random(), Math.random()]);
    assert.equal(new Set(idx).size, 8);
    for (const i of idx) assert.ok(i >= 0 && i < tc.tam);
  }
});

test('situação das pessoas: índice 0..26 a partir dos one-hot', () => {
  const o = new Float32Array(13);
  o.set([1, 0, 0, 1, 0, 0, 1, 0, 0], 2);
  assert.equal(situacao(o), 0);
  o.set([0, 0, 1, 0, 1, 0, 1, 0, 0], 2);
  assert.equal(situacao(o), 2 * 9 + 1 * 3 + 0);
  o.set([0, 0, 1, 0, 0, 1, 0, 0, 1], 2);
  assert.equal(situacao(o), 26);
});

test('Sarsa semi-gradiente: 1.500 atualizações reproduzem os pesos e os valores Q do Python', () => {
  const r = ref.sarsa;
  const ag = new SarsaSemiGradiente({ nTiles: r.cfg.n_tiles, nTil: r.cfg.n_til, alpha: r.cfg.alpha, gamma: r.cfg.gamma });
  const env = new BarcoContinuo();
  const obs = new Float32Array(13);
  let idx = new Int32Array(r.cfg.n_til), idx2 = new Int32Array(r.cfg.n_til);
  env.reset(r.semente_env);
  let sit = ag.feat(env.observacao(obs), idx);
  let maxDelta = 0, terminais = 0;
  r.acao.forEach((a, t) => {
    const o = env.step(a, obs);
    assert.equal(o.terminated || o.truncated, r.fim_episodio[t], `fim do episódio no passo ${t}`);
    if (o.terminated) terminais++;
    const sit2 = ag.feat(obs, idx2);
    const d = ag.atualiza(sit, idx, a, o.recompensa, o.terminated, sit2, idx2, r.acao2[t]);
    maxDelta = Math.max(maxDelta, Math.abs(d - r.deltas[t]));
    let s2 = sit2;
    if (o.terminated || o.truncated) { env.reset(); s2 = ag.feat(env.observacao(obs), idx2); }
    [idx, idx2] = [idx2, idx];
    sit = s2;
  });
  assert.ok(terminais >= 1, 'a sequência de referência inclui episódios concluídos com sucesso');
  assert.ok(maxDelta < 1e-9, `erro máximo do erro TD = ${maxDelta}`);
  let nz = 0, soma = 0;
  for (const v of ag.w) { if (v !== 0) nz++; soma += v; }
  assert.equal(nz, r.n_nao_nulos);
  assert.ok(Math.abs(soma - r.soma_w) < 1e-8, `Σw ${soma} vs ${r.soma_w}`);
  for (const [i, v] of r.amostra) assert.ok(Math.abs(ag.w[i] - v) < 1e-9, `w[${i}]`);
  r.obs_q.forEach((o, k) => {
    const q = ag.qValores(Float32Array.from(o));
    q.forEach((v, a) => assert.ok(Math.abs(v - r.q[k][a]) < 1e-9));
  });
});

test('serialização esparsa em base64 preserva a política (float32)', () => {
  const ag = new SarsaSemiGradiente({ nTiles: [6, 6, 3], nTil: 4, alpha: 0.3 });
  for (let i = 0; i < ag.w.length; i += 7) ag.w[i] = Math.fround(Math.sin(i));
  const j = JSON.parse(JSON.stringify(sarsaParaJson(ag)));
  const ag2 = sarsaDeJson(j);
  assert.equal(j.n_nao_nulos, Math.ceil(ag.w.length / 7) - (Math.sin(0) === 0 ? 1 : 0));
  for (let i = 0; i < ag.w.length; i++) assert.equal(ag2.w[i], ag.w[i]);
});

test('treino no navegador: aprende a resgatar nas 3 pessoas em poucos passos (limiar frouxo e semente fixa)', async () => {
  const ag = new SarsaSemiGradiente({ nTiles: [8, 8, 3], nTil: 8, alpha: 0.3, eps0: 0.3, eps1: 0.02, semente: 0 });
  const pontos = [];
  const { curva, episodios } = await ag.treinar({ passos: 300000, freqAval: 50000, nEpAval: 20, aoPonto: (p) => pontos.push(p) });
  assert.equal(curva.length, 6);
  assert.ok(episodios.length > 1000);
  assert.deepEqual(curva, pontos);
  const final = curva[curva.length - 1];
  // piloto em Python (resultados/piloto_calibracao.md): sucesso ≈ 0,99 com 400 mil passos nesta configuração; aqui só se exige aprendizado claro
  assert.ok(final.ret_desc > curva[0].ret_desc + 30, `retorno descontado ${curva[0].ret_desc} → ${final.ret_desc}`);
  assert.ok(final.sucesso >= 0.5, `sucesso final ${final.sucesso}`);
});

test('inferência de MLP idêntica à do PyTorch (MLPs sintéticos, ReLU e tanh, 40 observações cada)', () => {
  for (const c of ref.mlp_sintetico) {
    const m = criarMLP(c.modelo);
    c.obs.forEach((o, i) => {
      const y = m.saidas(Float32Array.from(o));
      y.forEach((v, k) => assert.ok(Math.abs(v - c.saidas[i][k]) < 1e-5, `saída ${k} da obs ${i}: ${v} vs ${c.saidas[i][k]}`));
      assert.equal(m.agir(Float32Array.from(o)), c.acoes[i]);
    });
  }
});
