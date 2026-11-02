import test from 'node:test';
import assert from 'node:assert/strict';
import { traduzHipotese, traduzVeredito, classeVeredito, textoHp } from '../src/ui/resultados.js';
import { lerNumero, fmtNum } from '../src/core/numeros.js';

test('tradução dos textos de hipoteses.csv para EN (e identidade em PT)', () => {
  assert.equal(traduzHipotese('H1: DQN vs PPO (AUC do retorno descontado)', 'en'), 'H1: DQN vs PPO (AUC of the discounted return)');
  assert.equal(traduzHipotese('H3: DQN vs heuristica (retorno descontado final)', 'en'), 'H3: DQN vs heuristic (final discounted return)');
  assert.equal(traduzHipotese('H1: DQN vs PPO (AUC do retorno descontado)', 'pt'), 'H1: DQN vs PPO (AUC do retorno descontado)');
  assert.equal(traduzVeredito('confirmada (abaixo da referência)', 'en'), 'confirmed (below the reference)');
  assert.equal(traduzVeredito('inconclusiva (direção favorável, p ≥ 0,05)', 'en'), 'inconclusive (favorable direction, p ≥ 0.05)');
  assert.equal(traduzVeredito('refutada (direção contrária)', 'en'), 'refuted (opposite direction)');
  assert.equal(classeVeredito('confirmada'), 'v-conf');
  assert.equal(classeVeredito('refutada (acima da referência)'), 'v-refut');
  assert.equal(classeVeredito('inconclusiva (sem diferença)'), 'v-inc');
});

test('textoHp: lista chave=valor e arranjos', () => {
  assert.equal(textoHp({ learning_rate: 0.001, net_arch: [128, 128] }), 'learning_rate=0.001, net_arch=[128,128]');
});

test('leitura e formatação de números nos dois idiomas', () => {
  assert.equal(lerNumero('0,25'), 0.25);
  assert.equal(lerNumero('0.25'), 0.25);
  assert.equal(lerNumero('1.234,5'), 1234.5);
  assert.equal(lerNumero('1,234.5'), 1234.5);
  assert.ok(Number.isNaN(lerNumero('abc')));
  assert.equal(fmtNum(1234.5, 1, 'pt'), '1.234,5');
  assert.equal(fmtNum(1234.5, 1, 'en'), '1,234.5');
});

import { lerParametros } from '../src/ui/treinar.js';

test('formulário de treino: leitura de parâmetros com vírgula ou ponto e validação de intervalos', () => {
  const ok = { nx: '8', ny: '8', ni: '3', ntil: '8', alfa: '0,3', eps0: '0.3', eps1: '0,02', gama: '0,99', sem: '0', passos: '300000' };
  const r = lerParametros(ok);
  assert.deepEqual(r.cfg.nTiles, [8, 8, 3]);
  assert.equal(r.cfg.alpha, 0.3);
  assert.equal(r.cfg.eps0, 0.3);
  assert.equal(r.passos, 300000);
  for (const [campo, valor] of [['nx', '0'], ['nx', '2,5'], ['alfa', '5'], ['alfa', 'abc'], ['eps0', '1,5'], ['gama', '0,2'], ['passos', '10'], ['ntil', '99']]) {
    assert.ok(lerParametros({ ...ok, [campo]: valor }).erro, `${campo}=${valor} deveria ser rejeitado`);
  }
});
