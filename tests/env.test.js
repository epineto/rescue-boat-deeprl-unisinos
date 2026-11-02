import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { BarcoContinuo, PARAMS_PADRAO, MAPA_PADRAO } from '../src/core/env.js';
import { criarRng } from '../src/core/rng.js';

const ref = JSON.parse(readFileSync(new URL('./trajetoria_referencia.json', import.meta.url), 'utf8'));

test('parâmetros e mapa do JS coincidem com os da trajetória de referência (Python)', () => {
  assert.deepEqual({ ...PARAMS_PADRAO }, ref.parametros);
  assert.deepEqual(MAPA_PADRAO, ref.mapa);
});

test('mulberry32: primeiros valores idênticos à implementação de referência', () => {
  // valores calculados pela implementação Python (Mulberry32(0) e Mulberry32(12345))
  const r0 = criarRng(0), r1 = criarRng(12345);
  const ref0 = JSON.parse(readFileSync(new URL('./rng_referencia.json', import.meta.url), 'utf8'));
  assert.deepEqual([r0.u(), r0.u(), r0.u()], ref0.semente0);
  assert.deepEqual([r1.u(), r1.u(), r1.u()], ref0.semente12345);
});

test('trajetórias de referência: posição, intensidade, recompensa, situação e término (tolerância 1e-9)', () => {
  assert.ok(ref.trajetorias.length >= 2);
  let maxErro = 0;
  for (const tr of ref.trajetorias) {
    const env = new BarcoContinuo(ref.parametros, ref.mapa);
    env.reset(tr.semente);
    assert.ok(Math.abs(env.inten - tr.intensidade_inicial) < 1e-12);
    tr.acoes.forEach((a, k) => {
      const o = env.step(a), e = tr.passos[k];
      for (const [v, w] of [[env.x, e.x], [env.y, e.y], [env.inten, e.intensidade], [o.recompensa, e.recompensa]]) {
        maxErro = Math.max(maxErro, Math.abs(v - w));
        assert.ok(Math.abs(v - w) < 1e-9, `${tr.nome} passo ${k}: ${v} vs ${w}`);
      }
      assert.deepEqual(env.status, e.status, `${tr.nome} passo ${k}: situação`);
      assert.equal(o.terminated, e.terminated);
      assert.equal(o.truncated, e.truncated);
      assert.equal(env.toques, e.toques);
    });
  }
  assert.ok(maxErro < 1e-9);
});

test('a mesma semente dá o mesmo ruído, independentemente da ação (consumo fixo de 2 pares por passo)', () => {
  const a = new BarcoContinuo(), b = new BarcoContinuo();
  a.reset(7); b.reset(7);
  for (let k = 0; k < 20; k++) { a.step(0); b.step(k % 4); }
  assert.equal(a.rng.estado, b.rng.estado);
  assert.equal(a.inten, b.inten);
});

test('observação: 13 componentes em [0, 1], float32, posição normalizada e one-hot das pessoas', () => {
  const env = new BarcoContinuo();
  const o = env.reset(3);
  assert.ok(o instanceof Float32Array);
  assert.equal(o.length, 13);
  assert.equal(o[0], Math.fround(2.5 / 6));
  assert.equal(o[1], Math.fround(5.5 / 6));
  assert.deepEqual([...o.slice(2, 11)], [1, 0, 0, 1, 0, 0, 1, 0, 0]);
  assert.equal(o[11], 0);
  for (let k = 0; k < 150; k++) { const r = env.step(k % 4); for (const v of r.obs) assert.ok(v >= 0 && v <= 1); if (r.terminated || r.truncated) break; }
});

test('regras: colisão com bloco, limites do mapa, truncamento e erro em parâmetro desconhecido', () => {
  assert.throws(() => new BarcoContinuo({ inexistente: 1 }), TypeError);
  const env = new BarcoContinuo({ sigma_mov: 0, corr_sigma: 0 });
  env.reset(1);
  env.x = 3.9; env.y = 1.5; // linha 1, coluna 3; o bloco ocupa a coluna 4 da mesma linha
  env.step(2); // ir a leste por 0,7 cairia em x = 4,6 (dentro do bloco): o barco não se move
  assert.equal(env.x, 3.9);
  assert.equal(env.y, 1.5);
  const e2 = new BarcoContinuo();
  e2.reset(0);
  for (let k = 0; k < 40; k++) e2.step(2);
  assert.ok(e2.x < 6 && e2.x >= 0);
  const e3 = new BarcoContinuo();
  e3.reset(0);
  let fim = null;
  for (let k = 0; k < 100 && !fim; k++) { const r = e3.step(0); if (r.terminated || r.truncated) fim = r; }
  assert.ok(fim && fim.truncated && !fim.terminated && e3.passos === 100);
});
