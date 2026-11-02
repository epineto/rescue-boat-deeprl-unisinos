import test from 'node:test';
import assert from 'node:assert/strict';
import { carregarDados } from '../src/core/dados.js';
import { criarMLP } from '../src/core/mlp.js';
import { sarsaDeJson } from '../src/core/sarsa.js';
import { compararAgentes } from '../src/core/comparar.js';
import { BarcoContinuo } from '../src/core/env.js';
import { avaliar, resumo } from '../src/core/avaliar.js';
import { criarPolitica } from '../src/core/politicas.js';
import { lerDados, temDados } from './util.js';

if (!temDados()) test('modelos exportados presentes em data/', { todo: 'executar tools/exportar_para_lab.py' }, () => {});
else {
  const D = await carregarDados(lerDados);
  const P = D.paridade;

  test('paridade: observações fixas e saídas do PyTorch estão em data/paridade.json', () => {
    assert.ok(P && P.observacoes.length >= 20);
    for (const o of P.observacoes) { assert.equal(o.length, 13); for (const v of o) assert.ok(v >= 0 && v <= 1); }
  });

  test('inferência em JavaScript idêntica à do PyTorch/SB3 em todos os modelos exportados (saídas dentro de 1e-4; mesma ação)', () => {
    assert.ok(D.modelos.length > 0, 'nenhum modelo exportado');
    for (const m of D.modelos) {
      const ref = P.modelos[m.id];
      assert.ok(ref, `paridade de ${m.id}`);
      const rede = criarMLP(m.dados);
      let maxDif = 0, divergencias = 0;
      P.observacoes.forEach((o, i) => {
        const y = rede.saidas(Float32Array.from(o));
        assert.equal(y.length, ref.saidas[i].length);
        y.forEach((v, k) => { maxDif = Math.max(maxDif, Math.abs(v - ref.saidas[i][k])); });
        const ordenado = [...ref.saidas[i]].sort((a, b) => b - a);
        const empate = ordenado[0] - ordenado[1] < 1e-5;
        if (!empate && rede.agir(Float32Array.from(o)) !== ref.acoes[i]) divergencias++;
      });
      assert.ok(maxDif < 1e-4, `${m.id}: diferença máxima ${maxDif}`);
      assert.equal(divergencias, 0, `${m.id}: ações divergentes`);
    }
  });

  test('Sarsa exportado: valores Q idênticos aos do Python (pesos em float32)', () => {
    if (!D.sarsa) return;
    const ag = sarsaDeJson(D.sarsa.dados);
    assert.equal(P.sarsa.id, D.sarsa.id);
    P.observacoes.forEach((o, i) => {
      const q = ag.qValores(Float32Array.from(o));
      q.forEach((v, a) => assert.ok(Math.abs(v - P.sarsa.q[i][a]) < 1e-6, `obs ${i} ação ${a}: ${v} vs ${P.sarsa.q[i][a]}`));
    });
  });

  test('desempenho em JavaScript dos modelos exportados é compatível com as métricas das execuções (episódios com sementes 10000+)', () => {
    for (const m of D.modelos) {
      const mt = m.metricas;
      if (!mt || mt.ret_desc === undefined) continue;
      const env = new BarcoContinuo();
      const av = resumo(avaliar(criarPolitica({ tipo: 'mlp', dados: m.dados }, env), 100, 10000, 0.99, {}, env));
      // a avaliação do experimento usou o mesmo ambiente e as mesmas sementes: coincide a menos de episódios em que cos/sin diferem na última casa
      assert.ok(Math.abs(av.ret_desc - mt.ret_desc) < 12 + 3 * Math.max(av.ret_desc_ic95, 1), `${m.id}: JS ${av.ret_desc} vs Python ${mt.ret_desc}`);
      assert.ok(Math.abs(av.sucesso - mt.sucesso) <= 0.12, `${m.id}: sucesso JS ${av.sucesso} vs Python ${mt.sucesso}`);
    }
  });

  test('comparação em N episódios (núcleo do Web Worker): mesmas sementes, resultados reprodutíveis e progresso', async () => {
    const agentes = [{ id: 'heuristica', espec: { tipo: 'heuristica' } }, { id: 'aleatoria', espec: { tipo: 'aleatoria' } }, ...D.modelos.slice(0, 2).map((m) => ({ id: m.id, espec: { tipo: 'mlp', dados: m.dados } }))];
    let chamadas = 0;
    const r1 = await compararAgentes({ agentes, n: 20, semente0: 10000, gamma: 0.99, aoProgresso: () => { chamadas++; } });
    const r2 = await compararAgentes({ agentes, n: 20, semente0: 10000, gamma: 0.99 });
    assert.ok(chamadas > 0);
    assert.equal(r1.n, 20);
    assert.deepEqual(r1.resultados.heuristica.vetores.ret, r2.resultados.heuristica.vetores.ret);
    assert.equal(r1.resultados.heuristica.resumo.sucesso, 1);
    for (const a of agentes) assert.equal(r1.resultados[a.id].vetores.ret.length, 20);
    const parada = await compararAgentes({ agentes, n: 100, semente0: 1, deveParar: () => true, ceder: async () => {} });
    assert.ok(parada.parcial && parada.n < 100);
  });
}
