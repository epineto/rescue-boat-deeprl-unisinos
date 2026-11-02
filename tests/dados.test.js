import test from 'node:test';
import assert from 'node:assert/strict';
import { parseCSV, carregarDados, agregarBusca, curvaMedia, contarAprendizes, ALGOS } from '../src/core/dados.js';
import { primeiroSucesso } from '../src/ui/resultados.js';
import { ic95 } from '../src/core/estat.js';
import { lerDados, temDados } from './util.js';

test('parseCSV: aspas, vírgulas e aspas escapadas dentro de campos, CRLF, números e vazios', () => {
  const r = parseCSV('a,b,c\r\n1,"x, y",\r\n2.5,"{""k"": 1}",z\r\n\r\n');
  assert.deepEqual(r, [{ a: 1, b: 'x, y', c: null }, { a: 2.5, b: '{"k": 1}', c: 'z' }]);
  assert.deepEqual(parseCSV(''), []);
  assert.deepEqual(parseCSV('a\n-1e-3\n+2\nabc1\n'), [{ a: -0.001 }, { a: 2 }, { a: 'abc1' }]);
});

test('ic95 reproduz a regra do Python (t de Student para n ≤ 11; 2,0 até 30; 1,96 acima)', () => {
  const v3 = [1, 2, 3];
  assert.ok(Math.abs(ic95(v3) - 4.303 * 1 / Math.sqrt(3)) < 1e-12);
  const v5 = [10, 12, 11, 13, 9];
  const sd = Math.sqrt(v5.reduce((a, x) => a + (x - 11) ** 2, 0) / 4);
  assert.ok(Math.abs(ic95(v5) - 2.776 * sd / Math.sqrt(5)) < 1e-12);
  assert.ok(Number.isNaN(ic95([1])));
});

test('curvaMedia: média, IC95% e extremos por passo', () => {
  const c = curvaMedia([{ passos: 10, v: 1 }, { passos: 10, v: 3 }, { passos: 20, v: 5 }, { passos: 20, v: null }], 'v');
  assert.deepEqual(c.passos, [10, 20]);
  assert.deepEqual(c.media, [2, 5]);
  assert.deepEqual(c.min, [1, 5]);
  assert.deepEqual(c.max, [3, 5]);
  assert.equal(c.ic[1], 0);
});

// ---- experimento suplementar (orçamento maior): carregamento com arquivos sintéticos ----------------------------------------------------
const CAB_RUNS = 'algo,config,semente,passos,ret,ret_desc,sucesso,passos_ep,toca_destrocos,tempo_s,ret_desc_400mil,sucesso_400mil,entropia,frac_patamar,sucesso_treino,hp';
const ARQ_EXTRA = {
  'extra_runs.csv': `${CAB_RUNS}\nA2C,a2c_x,200,2000000,-100,-63.4,0,100,0,10,-63.4,0,1.38,0.001,0,"{""n_envs"": 16}"\nPPO,ppo_x,200,2000000,190,120,0.9,50,0.1,20,1,0,0.17,0,1,"{""n_envs"": 8}"\n`,
  'extra_resumo.csv': 'algo,config,n,passos,n_aprenderam,ret_desc\nA2C,a2c_x,1,2000000,0,-63.4\nPPO,ppo_x,1,2000000,1,120\n',
  'extra_curvas.csv': 'algo,config,semente,passos,passos_reais,ret,ret_desc,sucesso,passos_ep\nPPO,ppo_x,200,400000,400000,0,1,0,100\nPPO,ppo_x,200,500000,500000,190,120,0.9,50\n',
  'cartpole_runs.csv': 'algo,config,semente,passos,ret,ret_min,frac_ep_475,tempo_s\nPPO,ppo_x,200,100000,500,500,1,90\n',
  'cartpole_resumo.csv': 'algo,config,n,ret,ret_min_sem,ret_max_sem,frac_ep_475,tempo_s\nPPO,ppo_x,1,500,500,500,1,90\n',
};
const FALSO = (extraIdx, omitir = []) => {
  const arquivos = { 'final_resumo.csv': 'agente,ret_desc,sucesso\nDQN,1,1\n', 'linhas_de_base.csv': 'agente,ret_desc,sucesso\ntabular,1,1\n', ...ARQ_EXTRA };
  for (const o of omitir) delete arquivos[o];
  const indice = { versao: 1, completo: true, modelos: [], sarsa: null, arquivos: [...Object.keys(arquivos), 'indice.json'], ...(extraIdx ? { extra: extraIdx } : {}) };
  return async (nome) => { if (nome === 'indice.json') return JSON.stringify(indice); if (!(nome in arquivos)) throw new Error('404 ' + nome); return arquivos[nome]; };
};

test('carregarDados: sem o bloco extra no índice, D.extra é null (a aba mostra "indisponível")', async () => {
  assert.equal((await carregarDados(FALSO(null))).extra, null);
});

test('carregarDados: com o bloco extra, lê runs (hp como objeto), resumo, curvas e CartPole', async () => {
  const D0 = await carregarDados(FALSO({ passos: 2000000, passos_comparacao: 400000, limiar_aprendeu: 0.5 }));
  assert.equal(D0.extra.runs.length, 2);
  assert.deepEqual(D0.extra.runs[0].hp, { n_envs: 16 });
  assert.equal(D0.extra.runs[1].sucesso, 0.9);
  assert.equal(D0.extra.curvas.length, 2);
  assert.equal(D0.extra.cartpoleResumo[0].ret, 500);
  assert.equal(D0.extra.info.passos, 2000000);
});

test('carregarDados: arquivo obrigatório do extra ausente → extra null, sem quebrar o resto; CartPole ausente é tolerado', async () => {
  assert.equal((await carregarDados(FALSO({ passos: 1 }, ['extra_curvas.csv']))).extra, null);
  const D1 = await carregarDados(FALSO({ passos: 1 }, ['cartpole_runs.csv', 'cartpole_resumo.csv']));
  assert.equal(D1.extra.cartpoleResumo, null);
  assert.equal(D1.base.tabular.sucesso, 1);
});

test('contarAprendizes: conta runs com sucesso final ≥ limiar', () => {
  assert.deepEqual(contarAprendizes([{ sucesso: 0 }, { sucesso: 0.49 }, { sucesso: 0.5 }, { sucesso: 1 }]), { aprenderam: 2, n: 4 });
  assert.deepEqual(contarAprendizes([{ sucesso: 0.3 }], 0.25), { aprenderam: 1, n: 1 });
  assert.deepEqual(contarAprendizes([]), { aprenderam: 0, n: 0 });
});

test('primeiroSucesso: primeira avaliação com sucesso ≥ 50% e a anterior; null se nunca', () => {
  const cv = [{ passos: 300, sucesso: 0.4 }, { passos: 100, sucesso: 0 }, { passos: 200, sucesso: 0.1 }, { passos: 400, sucesso: 0.5 }, { passos: 500, sucesso: 1 }];
  assert.deepEqual(primeiroSucesso(cv), { passo: 400, anterior: 300 });
  assert.deepEqual(primeiroSucesso([{ passos: 100, sucesso: 1 }]), { passo: 100, anterior: null });
  assert.equal(primeiroSucesso([{ passos: 100, sucesso: 0.2 }]), null);
});

if (!temDados()) test('dados exportados presentes em data/', { todo: 'executar tools/exportar_para_lab.py' }, () => {});
else {
  const D = await carregarDados(lerDados);

  test('data/: índice, linhas de base e resumo final presentes e coerentes', () => {
    const i = D.indice;
    assert.equal(i.versao, 1);
    assert.equal(i.completo, true);
    assert.ok(!('provisorio' in i));
    assert.ok(Math.abs(i.v_estrela_t2 - 161.93) < 0.01);
    for (const a of ['tabular', 'heuristica', 'aleatoria']) {
      assert.ok(D.base[a], a);
      assert.ok(D.base[a].ret_desc < 200 && D.base[a].sucesso >= 0 && D.base[a].sucesso <= 1);
    }
    assert.ok(D.base.tabular.ret_desc > 100 && D.base.aleatoria.ret_desc < 0);
    for (const r of D.finalResumo) assert.ok(r.agente && Number.isFinite(r.ret_desc) && r.sucesso >= 0 && r.sucesso <= 1, r.agente);
    for (const a of Object.keys(i.melhores)) assert.ok(ALGOS.includes(a));
  });

  test('data/: o índice lista todos os arquivos que existem e só eles; os opcionais ausentes viram null', () => {
    for (const f of D.indice.arquivos) assert.doesNotReject(() => lerDados(f));
    for (const campo of ['hipoteses', 'auc']) assert.ok(D[campo] === null || Array.isArray(D[campo]));
  });

  test('data/: a agregação feita no navegador reproduz busca_resumo.csv (média, DP e IC95% entre sementes)', () => {
    if (!D.buscaRuns || !D.buscaResumo) return;
    const agg = agregarBusca(D.buscaRuns);
    assert.equal(agg.length, D.buscaResumo.length);
    for (const r of D.buscaResumo) {
      const a = agg.find((x) => x.algo === r.algo && x.config === r.config);
      assert.ok(a, `${r.algo} ${r.config}`);
      assert.equal(a.n, r.n);
      assert.ok(Math.abs(a.ret_desc - r.ret_desc) < 1e-5);
      if (Number.isFinite(r.ret_desc_ic95)) assert.ok(Math.abs(a.ret_desc_ic95 - r.ret_desc_ic95) < 1e-4, `${r.config}: IC ${a.ret_desc_ic95} vs ${r.ret_desc_ic95}`);
      assert.ok(Math.abs(a.sucesso - r.sucesso) < 1e-5);
    }
  });

  test('data/: a configuração escolhida em cada algoritmo é a de maior retorno descontado (ou empata), conforme o critério declarado', () => {
    if (!D.buscaResumo) return;
    for (const [a, m] of Object.entries(D.indice.melhores)) {
      const sub = D.buscaResumo.filter((r) => r.algo === a);
      const escolhida = sub.find((r) => r.config === m.config);
      assert.ok(escolhida, `${a}: configuração escolhida ausente da busca`);
      const max = Math.max(...sub.map((r) => r.ret_desc));
      assert.ok(escolhida.ret_desc >= max - 1e-9 || escolhida.escolhida === 1, `${a}: ${m.config}`);
    }
  });

  test('data/: curvas têm passos crescentes por execução e métricas finitas', () => {
    for (const curvas of [D.curvasBusca, D.curvasFinais]) {
      if (!curvas) continue;
      const por = new Map();
      for (const l of curvas) { const k = `${l.algo}|${l.config}|${l.semente}`; if (!por.has(k)) por.set(k, []); por.get(k).push(l); }
      for (const ls of por.values()) {
        for (let i = 1; i < ls.length; i++) assert.ok(ls[i].passos > ls[i - 1].passos);
        for (const l of ls) assert.ok(Number.isFinite(l.ret_desc) && l.sucesso >= 0 && l.sucesso <= 1);
      }
    }
  });

  test('data/: finais e busca usam o mesmo conjunto de colunas e todas as sementes esperadas por configuração', () => {
    if (!D.buscaRuns) return;
    const por = new Map();
    for (const r of D.buscaRuns) por.set(`${r.algo}|${r.config}`, (por.get(`${r.algo}|${r.config}`) ?? 0) + 1);
    if (D.indice.completo) for (const [k, n] of por) assert.equal(n, D.indice.protocolo.sementes_busca.length, k);
  });

  test('data/extra: resumo = agregação das runs (média, IC95% entre sementes, nº que aprenderam) e coerente com o índice', () => {
    const E = D.extra;
    assert.ok(E, 'bloco extra ausente: executar tools/exportar_para_lab.py --so-extra');
    assert.equal(E.info.passos, 2000000);
    const agg = agregarBusca(E.runs);
    assert.equal(agg.length, E.resumo.length);
    for (const r of E.resumo) {
      const a = agg.find((x) => x.config === r.config);
      assert.ok(a, r.config);
      assert.equal(a.n, r.n);
      assert.deepEqual(E.info.sementes[r.config], E.runs.filter((x) => x.config === r.config).map((x) => x.semente).sort((p, q) => p - q));
      assert.ok(Math.abs(a.ret_desc - r.ret_desc) < 1e-5);
      if (r.n > 1) assert.ok(Math.abs(a.ret_desc_ic95 - r.ret_desc_ic95) < 1e-3, `${r.config}: IC ${a.ret_desc_ic95} vs ${r.ret_desc_ic95}`);
      assert.equal(contarAprendizes(E.runs.filter((x) => x.config === r.config), E.info.limiar_aprendeu).aprenderam, r.n_aprenderam);
    }
  });

  test('data/extra: cada run tem curva até o orçamento e o valor aos 400 mil passos é o da curva', () => {
    const E = D.extra;
    for (const r of E.runs) {
      const cv = E.curvas.filter((l) => l.config === r.config && l.semente === r.semente).sort((a, b) => a.passos - b.passos);
      assert.equal(cv.length, 20, `${r.config} s${r.semente}`);
      assert.equal(cv.at(-1).passos, r.passos);
      const c4 = cv.find((l) => l.passos === E.info.passos_comparacao);
      assert.ok(Math.abs(c4.ret_desc - r.ret_desc_400mil) < 1e-5 && Math.abs(c4.sucesso - r.sucesso_400mil) < 1e-5);
      assert.ok(r.sucesso >= 0 && r.sucesso <= 1 && r.entropia > 0 && r.entropia <= Math.log(4) + 1e-9);
    }
  });

  test('data/extra: afirmações do texto da aba conferidas nos dados (A2C não aprende; PPO aprende em parte e só depois dos 400 mil passos)', () => {
    const E = D.extra;
    const a2c = E.runs.filter((r) => r.algo === 'A2C');
    assert.equal(a2c.length, 3);
    assert.deepEqual(contarAprendizes(a2c), { aprenderam: 0, n: 3 });
    assert.ok(E.curvas.filter((l) => l.algo === 'A2C').every((l) => l.sucesso === 0), 'A2C: sucesso 0 em toda a curva');
    const ppo = E.runs.filter((r) => r.config === 'ppo_8envs');
    assert.deepEqual(contarAprendizes(ppo), { aprenderam: 1, n: 3 });
    const aprendiz = ppo.find((r) => r.sucesso >= 0.5);
    assert.ok(aprendiz.sucesso_400mil < 0.5, 'a semente que aprende ainda não tinha aprendido aos 400 mil passos');
    const pr = primeiroSucesso(E.curvas.filter((l) => l.config === 'ppo_8envs' && l.semente === aprendiz.semente));
    assert.deepEqual(pr, { passo: 500000, anterior: 400000 });
    assert.ok(ppo.filter((r) => r.sucesso < 0.5).every((r) => Math.abs(r.ret_desc + 21.4) < 0.1), 'platô de uma pessoa entregue: retorno descontado ≈ −21,4');
    const clip = E.runs.filter((r) => r.config === 'ppo_clip0.3_8envs');
    assert.equal(clip.length, 1);
    assert.deepEqual(contarAprendizes(clip), { aprenderam: 1, n: 1 });
  });

  test('data/extra: sanidade no CartPole (PPO e A2C com 16 ambientes resolvem em todas as sementes; A2C padrão só em parte)', () => {
    const cp = D.extra.cartpoleResumo;
    assert.ok(cp && cp.length === 5);
    for (const r of cp.filter((x) => x.algo === 'PPO' || x.config === 'a2c_16envs')) assert.ok(r.ret_min_sem >= 475 && r.ret === 500, r.config);
    const padrao = cp.find((x) => x.config === 'a2c_padrao');
    assert.ok(padrao.ret_min_sem < 475 && padrao.ret_max_sem >= 475);
    for (const r of cp) {
      const sem = D.extra.cartpoleRuns.filter((x) => x.config === r.config);
      assert.equal(sem.length, r.n);
      assert.ok(Math.abs(sem.reduce((acc, x) => acc + x.ret, 0) / sem.length - r.ret) < 1e-4);
    }
  });
}
