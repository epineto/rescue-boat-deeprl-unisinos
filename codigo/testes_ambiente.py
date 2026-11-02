"""Testes do ambiente contínuo do Trabalho 3 (compatíveis com pytest; também executáveis com ``python testes_ambiente.py``).

Uso:
    python testes_ambiente.py                      # executa todos os testes
    python testes_ambiente.py --gerar-referencia   # (re)grava ../resultados/trajetoria_referencia.json e executa os testes
"""
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
import barco_continuo as B  # noqa: E402

REF = AQUI.parent / "resultados" / "trajetoria_referencia.json"
# ambiente sem nenhum ruído e sem correnteza média: dinâmica exatamente calculável à mão
SEM_RUIDO = dict(passo=0.55, r_embarque=0.55, r_abrigo=0.6, max_passos=200, sigma_mov=0.0, corr_sigma=0.0, corr_sigma_i=0.0, corr_sigma_0=0.0, corr_media=0.0, corr_vmax=0.5)


def _env_sem_ruido(**kw):
    env = B.RescueBoatContEnv(**{**SEM_RUIDO, **kw})
    env.reset(seed=1)
    return env


def _aprox(a, b, tol=1e-9):
    assert abs(a - b) <= tol, (a, b)


# ---------------------------------------------------------------------------------------------
def test_espacos_e_observacao():
    env = B.RescueBoatContEnv()
    obs, info = env.reset(seed=0)
    assert env.observation_space.shape == (13,) and env.observation_space.dtype == np.float32
    assert env.action_space.n == 4
    assert obs.dtype == np.float32 and obs.shape == (13,) and env.observation_space.contains(obs)
    _aprox(obs[0], 2.5 / 6, 1e-6)
    _aprox(obs[1], 5.5 / 6, 1e-6)
    assert list(obs[2:11]) == [1, 0, 0] * 3 and obs[11] == 0
    assert 0.0 <= obs[12] <= 1.0
    for _ in range(300):
        obs, r, te, tr, info = env.step(env.action_space.sample())
        assert env.observation_space.contains(obs)
        if te or tr:
            env.reset()


def test_gerador_portavel_valores_conhecidos():
    # valores calculados com a implementação de referência em JavaScript (Node 20)
    g = B.Mulberry32(0)
    esperado = [0.26642920868471265, 0.00032974570058286, 0.22327202744781971, 0.14620214793831110, 0.46732782293111086]
    for e in esperado:
        _aprox(g.uniforme(), e, 1e-15)
    g = B.Mulberry32(12345)
    esperado = [0.97972826776094735, 0.30675226449966431, 0.48420542152598500, 0.81793441250920296, 0.50942836934700608]
    for e in esperado:
        _aprox(g.uniforme(), e, 1e-15)
    z0, z1 = B.Mulberry32(12345).par()
    r = math.sqrt(-2 * math.log(1 - 0.97972826776094735))
    _aprox(z0, r * math.cos(2 * math.pi * 0.30675226449966431), 1e-12)
    _aprox(z1, r * math.sin(2 * math.pi * 0.30675226449966431), 1e-12)
    # as normais têm média ~0 e desvio ~1
    g = B.Mulberry32(7)
    z = np.array([g.par() for _ in range(20000)]).ravel()
    assert abs(z.mean()) < 0.03 and abs(z.std() - 1) < 0.03


def test_determinismo_por_semente():
    def roda(seed, n=120):
        env = B.RescueBoatContEnv()
        obs, _ = env.reset(seed=seed)
        rng = np.random.default_rng(99)
        out = [obs.copy()]
        for a in rng.integers(0, 4, n):
            obs, r, te, tr, _ = env.step(int(a))
            out.append(np.append(obs, r))
            if te or tr:
                break
        return np.concatenate(out)
    assert np.array_equal(roda(5), roda(5))
    assert not np.array_equal(roda(5), roda(6))
    env = B.RescueBoatContEnv()
    o1, _ = env.reset(seed=3)
    for _ in range(10):
        env.step(2)
    o2, _ = env.reset(seed=3)
    assert np.array_equal(o1, o2)
    # sem semente: primeiro reset sorteia do np_random seedado do Gymnasium (reproduzível)
    e1, e2 = B.RescueBoatContEnv(), B.RescueBoatContEnv()
    e1.reset(seed=None)
    e2.reset(seed=None)
    assert e1.np_random is not None


def test_movimento_sem_ruido_e_paredes():
    env = _env_sem_ruido()
    assert (env.x, env.y) == (2.5, 5.5)
    env.step(0)                                   # N: y diminui 0,55
    _aprox(env.y, 4.95)
    _aprox(env.x, 2.5)
    env.step(2)                                   # L
    _aprox(env.x, 3.05)
    env.step(1)                                   # S
    env.step(1)                                   # S encosta no chão (recorte em 6 - 1e-9)
    assert env.y < 6.0 and env.y > 5.9
    for _ in range(20):                           # parede oeste
        env.step(3)
    assert env.x == 0.0


def test_colisao_com_blocos():
    env = _env_sem_ruido()
    env.x, env.y = 2.5, 4.4                       # ao lado do bloco (4,1): x∈[1,2), y∈[4,5)
    env.step(3)                                   # O: x=1,95 cairia dentro do bloco, logo não move
    assert (env.x, env.y) == (2.5, 4.4)
    env.x, env.y = 3.8, 4.4                       # bloco (4,4): x∈[4,5)
    env.step(2)                                   # L: x=4,35 dentro do bloco
    assert (env.x, env.y) == (3.8, 4.4)
    env.x, env.y = 1.5, 3.7                       # acima do bloco (4,1): descer para y=4,25 entra no bloco
    env.step(1)
    assert (env.x, env.y) == (1.5, 3.7)
    env.x, env.y = 1.5, 2.3                       # abaixo do bloco (1,1) (y∈[1,2)): subir para y=1,75 entra no bloco
    env.inten = 0.0
    env.step(0)
    assert (env.x, env.y) == (1.5, 2.3)
    env.x, env.y = 1.5, 2.6                       # subir para y=2,05 é permitido (fora do bloco)
    env.step(0)
    _aprox(env.y, 2.05)
    # nunca se termina dentro de um bloco, mesmo com ruído, correnteza e ações aleatórias
    env = B.RescueBoatContEnv()
    env.reset(seed=11)
    rng = np.random.default_rng(0)
    for _ in range(5000):
        _, _, te, tr, _ = env.step(int(rng.integers(4)))
        assert not env._bloq[int(env.y)][int(env.x)]
        if te or tr:
            env.reset()


def test_correnteza_e_intensidade():
    env = _env_sem_ruido(corr_media=0.0, corr_theta=0.1)
    env.inten = 1.0
    env.x, env.y = 4.5, 3.2
    env.step(0)                                   # N: y=2,65 (na faixa) + arrasto 0,5·I = 0,5 → 3,15
    _aprox(env.y, 2.65 + 0.5)
    _aprox(env.inten, 1.0 + 0.1 * (0.0 - 1.0))    # AR(1) sem ruído: I ← I + θ(μ − I)
    env.inten = 0.0
    env.x, env.y = 4.5, 3.2
    env.step(0)                                   # I=0: sem arrasto, sobe 0,55
    _aprox(env.y, 3.2 - 0.55)
    env.x, env.y = 4.5, 3.9                       # fora da faixa: sem arrasto, mesmo com I=1
    env.inten = 1.0
    env.step(0)
    _aprox(env.y, 3.9 - 0.55)
    env.x, env.y = 4.5, 1.9                       # a posição após a fase 1 (2,45) está na faixa: há arrasto
    env.inten = 1.0
    env.step(1)
    _aprox(env.y, 1.9 + 0.55 + 0.5)
    env.x, env.y = 3.5, 1.2                       # S termina em 1,75 (fora da faixa; coluna 3 livre): sem arrasto
    env.inten = 1.0
    env.step(1)
    _aprox(env.y, 1.75)
    # a intensidade é sempre recortada em [0, 1], mesmo com ruído enorme
    env = B.RescueBoatContEnv(corr_sigma_i=2.0)
    env.reset(seed=1)
    for _ in range(300):
        env.step(2)
        assert 0.0 <= env.inten <= 1.0
    # a intensidade é visível na observação (último componente)
    obs, *_ = env.step(2)
    _aprox(float(obs[12]), env.inten, 1e-6)


def test_embarque_por_raio_e_capacidade():
    env = _env_sem_ruido()
    env.x, env.y = 0.95, 0.5                      # P1 em (0,5; 0,5); mover O → x=0,40, a 0,10 de P1
    env.step(3)
    assert env.status == [1, 0, 0] and env.aboard == 1
    env = _env_sem_ruido()
    env.x, env.y = 0.5, 1.6                       # após mover L: (1,05; 1,6), longe de P1
    env.step(2)
    assert env.status == [0, 0, 0]
    env = _env_sem_ruido(r_embarque=0.3)          # raio menor: x=0,65 está a 0,15 de P1 → embarca
    env.x, env.y = 1.2, 0.5
    env.step(3)
    assert env.status[0] == 1
    env = _env_sem_ruido(r_embarque=0.1)          # raio ainda menor: 0,15 > 0,1 → não embarca
    env.x, env.y = 1.2, 0.5
    env.step(3)
    assert env.status[0] == 0
    # capacidade: com 2 a bordo, a terceira pessoa continua ilhada
    env = _env_sem_ruido()
    env.status, env.aboard = [1, 1, 0], 2
    env.x, env.y = 5.5, 2.3                       # P3 em (5,5; 1,5): subir 0,55 → y=1,75 a 0,25 de P3 (dentro do raio)
    env.inten = 0.0
    env.step(0)
    assert env.status == [1, 1, 0] and env.aboard == 2
    env = _env_sem_ruido()
    env.status, env.aboard = [1, 0, 0], 1         # com 1 a bordo ainda cabe mais um
    env.x, env.y = 5.5, 2.3
    env.inten = 0.0
    env.step(0)
    assert env.status == [1, 0, 1] and env.aboard == 2


def test_desembarque_e_recompensas():
    env = _env_sem_ruido()
    env.status, env.aboard = [1, 1, 0], 2
    env.x, env.y = 2.5, 4.6
    _, r, te, tr, info = env.step(1)              # S: y=5,15, a 0,35 de H (raio 0,6) → entrega as duas
    _aprox(r, -1 + 2 * 50)
    assert env.status == [2, 2, 0] and env.aboard == 0 and not te and not tr
    env.status, env.aboard = [2, 2, 1], 1
    env.x, env.y = 2.5, 4.6
    _, r, te, tr, info = env.step(1)              # entrega a terceira: −1 + 50 + 100
    _aprox(r, -1 + 50 + 100)
    assert te and not tr and info["sucesso"]
    env = _env_sem_ruido()                        # sem ninguém a bordo, passar pelo abrigo não paga nada
    env.x, env.y = 2.5, 4.6
    _, r, te, tr, _ = env.step(1)
    _aprox(r, -1.0)


def test_destrocos_so_ao_entrar():
    env = _env_sem_ruido()
    env.x, env.y = 2.5, 4.4
    _, r, *_ = env.step(0)                        # y=3,85: a 0,35 do centro (2,5; 3,5) → entra
    _aprox(r, -10.0)
    assert env.toques == 1
    _, r, *_ = env.step(0)                        # y=3,30: ainda dentro → passo comum
    _aprox(r, -1.0)
    assert env.toques == 1
    _, r, *_ = env.step(0)                        # y=2,75: fora (e na faixa, sem arrasto com I=0)
    _aprox(r, -1.0)
    env.x, env.y = 2.5, 2.9
    env.step(1)                                   # S: y=3,45 (antes a 0,6 do centro) → entra de novo
    assert env.toques == 2
    env2 = _env_sem_ruido()                       # o canto da célula do destroço fica livre (disco inscrito na célula)
    env2.x, env2.y = 2.1, 2.5
    env2.step(1)                                  # S → (2,1; 3,05): distância ao centro (2,5; 3,5) ≈ 0,60 > 0,5
    assert env2.toques == 0


def test_truncamento():
    env = _env_sem_ruido(max_passos=5)
    for k in range(5):
        _, r, te, tr, info = env.step(1)          # encosta no chão
        assert not te
        assert tr == (k == 4)
    assert info.get("TimeLimit.truncated") is True
    env = B.RescueBoatContEnv()
    env.reset(seed=0)
    n = 0
    tr = False
    while not tr:
        _, _, te, tr, _ = env.step(1)
        n += 1
        assert not te
    assert n == B.PARAMS_PADRAO["max_passos"]


def test_trajetoria_montada_a_mao_resgate_completo():
    """Rota feita à mão, sem ruído e sem correnteza, que resgata P3 e P2 e entrega ambas (recompensa conferida)."""
    env = _env_sem_ruido()
    env.inten = 0.0
    seq = []
    # sobe pela coluna 5 (x=5,05 depois de 5 passos L), contornando os destroços e blocos
    seq += [2] * 5                                # L: x = 2,5 + 5·0,55 = 5,25
    seq += [0] * 2                                # N: y = 5,5 − 1,10 = 4,40 (célula (4,5), livre)
    seq += [0] * 2                                # N: y = 3,30 (linha 3, x=5,25 fora dos destroços)
    seq += [0] * 2                                # N: y = 2,20 (faixa; I=0 → sem arrasto)
    seq += [0] * 1                                # N: y = 1,65 (a 0,15 de P3 em (5,5; 1,5) e a 0,25 em x) → embarca
    total, ok = 0.0, False
    for a in seq:
        _, r, te, tr, info = env.step(a)
        total += r
    assert env.status == [0, 0, 1], env.status
    assert env.y < 2.0 and env.aboard == 1
    _aprox(total, -len(seq) + 0.0)                # nenhum evento de recompensa além de −1 por passo (sem destroços)
    assert env.toques == 0


def test_check_env_do_sb3():
    from stable_baselines3.common.env_checker import check_env
    check_env(B.RescueBoatContEnv(), warn=True)


def test_mdp_discreto_do_t2():
    env = B.RescueBoatEnv()
    assert env.nS == 832 and env.nA == 4
    V, pi, Q, it = B.iteracao_valor(env, gamma=0.99)
    _aprox(float(V[env.s0]), 161.93, 5e-3)
    for s in range(env.nS):
        for a in range(4):
            _aprox(sum(p for p, *_ in env.P[s][a]), 1.0, 1e-12)


def test_politicas_de_referencia():
    env = B.RescueBoatContEnv()
    tab = B.resumo(B.avaliar(B.PoliticaTabularDiscretizada(), 30, env=env))
    heu = B.resumo(B.avaliar(B.PoliticaHeuristica(env), 30, env=env))
    ale = B.resumo(B.avaliar(B.politica_aleatoria(0), 30, env=env))
    assert tab["sucesso"] >= 0.9 and heu["sucesso"] >= 0.9
    assert ale["sucesso"] <= 0.1
    assert tab["ret"] > ale["ret"] + 100 and heu["ret"] > ale["ret"] + 100


def test_renderizacao():
    env = B.RescueBoatContEnv(render_mode="ansi")
    env.reset(seed=0)
    txt = env.render()
    assert "B" in txt and "P1" in txt and "I=" in txt
    env = B.RescueBoatContEnv(render_mode="rgb_array")
    env.reset(seed=0)
    img = env.render()
    assert img.ndim == 3 and img.shape[2] == 3 and img.dtype == np.uint8


def test_tile_coding_e_sarsa():
    tc = B.TileCoder((12, 12, 4), 8)
    idx = tc.indices(np.array([[0.3, 0.7, 0.5]]))
    assert len(idx) == 8 and len(set(idx.tolist())) == 8 and idx.min() >= 0 and idx.max() < tc.tam
    assert np.array_equal(idx, tc.indices(np.array([[0.3, 0.7, 0.5]])))
    for z in ([0, 0, 0], [1, 1, 1], [0.999999, 0.5, 0.0]):
        i2 = tc.indices(np.array([z], dtype=float))
        assert i2.min() >= 0 and i2.max() < tc.tam
    ag = B.SarsaSemiGradiente(semente=0)
    curva, log, t = ag.treinar(4000, n_ep_aval=3, freq_aval=2000)
    assert len(curva) == 2 and log.shape[1] == 5


def test_fabrica_sb3_curta():
    for algo, hp in (("DQN", dict(learning_starts=50, buffer_size=2000)), ("PPO", dict(n_steps=64, batch_size=32)), ("A2C", {})):
        r = B.treinar(algo, hp, 0, 600, n_ep_aval=2, n_ep_final=3, freq_aval=300)
        assert len(r["curva"]) >= 2 and r["final"]["passos"] > 0


def test_velocidade_minima():
    v = B.velocidade(30000)
    print(f"  velocidade: {v:,.0f} passos/s")
    assert v >= 20000


def test_trajetoria_de_referencia_reproduzivel():
    if not REF.exists():
        print("  (sem arquivo de referência: pulado; rode --gerar-referencia)")
        return
    ref = json.loads(REF.read_text())
    for tr in ref["trajetorias"]:
        env = B.RescueBoatContEnv(**ref["parametros"])
        obs, info = env.reset(seed=tr["semente"])
        _aprox(env.inten, tr["intensidade_inicial"], 1e-12)
        for k, a in enumerate(tr["acoes"]):
            obs, r, te, trc, info = env.step(a)
            p = tr["passos"][k]
            _aprox(env.x, p["x"], 1e-12)
            _aprox(env.y, p["y"], 1e-12)
            _aprox(env.inten, p["intensidade"], 1e-12)
            _aprox(r, p["recompensa"], 1e-12)
            assert list(info["status"]) == p["status"] and te == p["terminated"] and trc == p["truncated"]
        assert len(tr["acoes"]) == len(tr["passos"])


# ---------------------------------------------------------------------------------------------
def gerar_referencia():
    """Grava duas trajetórias determinísticas (ações fixas) com estado e recompensa esperados a cada passo."""
    env0 = B.RescueBoatContEnv()
    heur = B.PoliticaHeuristica(env0)
    trajs = []

    def grava(nome, semente, acoes_fn, n_max):
        env = B.RescueBoatContEnv()
        obs, _ = env.reset(seed=semente)
        tr = {"nome": nome, "semente": semente, "intensidade_inicial": env.inten, "acoes": [], "passos": []}
        for k in range(n_max):
            a = acoes_fn(env, obs, k)
            obs, r, te, trc, info = env.step(a)
            tr["acoes"].append(int(a))
            tr["passos"].append({"x": env.x, "y": env.y, "intensidade": env.inten, "recompensa": r,
                                 "status": list(info["status"]), "terminated": bool(te), "truncated": bool(trc),
                                 "toques": info["toques"]})
            if te or trc:
                break
        trajs.append(tr)

    grava("heuristica_resgate_completo", 123, lambda env, obs, k: heur(env, obs), 200)
    rng = np.random.default_rng(7)
    sequencia = rng.integers(0, 4, 80).tolist()
    grava("acoes_aleatorias_fixas", 7, lambda env, obs, k: sequencia[k], 80)
    g = B.Mulberry32(12345)
    out = {
        "descricao": ("Trajetórias de referência do RescueBoatContEnv: ações fixas + semente → posição, intensidade, recompensa, "
                      "situação das pessoas e flags a cada passo. Gerador: mulberry32 + Box-Muller (ver docstring de barco_continuo.py). "
                      "Tolerância sugerida na comparação em JavaScript: 1e-9."),
        "parametros": B.PARAMS_PADRAO,
        "mapa": B.MAPA_PADRAO,
        "gerador": {"semente": 12345, "uniformes": [B.Mulberry32(12345).uniforme()] + [g.uniforme() for _ in range(0)],
                    "primeiros_uniformes": [], "par_normais": None},
        "trajetorias": trajs,
    }
    g = B.Mulberry32(12345)
    out["gerador"] = {"semente": 12345, "primeiros_uniformes": [g.uniforme() for _ in range(8)],
                      "par_normais_apos_semente": list(B.Mulberry32(12345).par())}
    REF.parent.mkdir(exist_ok=True)
    REF.write_text(json.dumps(out, indent=1))
    print("gravado:", REF, "| trajetórias:", [(t["nome"], len(t["acoes"])) for t in trajs])


if __name__ == "__main__":
    if "--gerar-referencia" in sys.argv:
        gerar_referencia()
    testes = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    falhas = 0
    for nome, f in testes:
        t0 = time.time()
        try:
            f()
            print(f"OK    {nome} ({time.time() - t0:.1f} s)")
        except Exception as e:  # noqa: BLE001
            falhas += 1
            import traceback
            traceback.print_exc()
            print(f"FALHA {nome}: {e!r}")
    print(f"\n{len(testes) - falhas}/{len(testes)} testes passaram")
    sys.exit(1 if falhas else 0)
