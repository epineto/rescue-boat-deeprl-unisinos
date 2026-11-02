"""Gera tests/reference.json: valores de referência calculados pelo Python (reference/python/barco_continuo.py) que os testes
em JavaScript (npm test) reproduzem. Reexecutar só se o ambiente Python mudar.

Uso:  python tools/gen_reference.py

Conteúdo:
  mdp_t2        V*(s0) e política ótima do MDP discreto do Trabalho 2 (832 estados)
  tile          índices do tile coding para várias configurações e 200 pontos em [0, 1]^3
  sarsa         sequência de ações e pesos após 1.500 atualizações semi-gradiente do Sarsa (verifica a regra de atualização)
  mlp_sintetico MLPs aleatórios (ReLU e tanh) e as saídas do PyTorch, para testar a inferência sem depender dos dados exportados
"""
import json
import sys
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "reference" / "python"))
import barco_continuo as B  # noqa: E402


def f32(x):
    """float32 → float com 9 dígitos significativos (ida e volta exata em float32)."""
    return float(f"{float(x):.9g}")


def ref_mdp():
    tab = B.PoliticaTabularDiscretizada()
    return {"nS": int(tab.env.nS), "v0": float(tab.v0), "pi": [int(a) for a in tab.pi]}


def ref_tile():
    rng = np.random.default_rng(0)
    saida = []
    for n_tiles, n_til in (((6, 6, 3), 8), ((8, 8, 3), 8), ((12, 12, 4), 8), ((5, 7, 4), 6), ((10, 10, 5), 4)):
        tc = B.TileCoder(n_tiles, n_til)
        z = rng.random((200, 3))
        z[0] = (0, 0, 0)
        z[1] = (1, 1, 1)
        z[2] = (0.5, 0.999999, 0.0000001)
        saida.append({"n_tiles": list(n_tiles), "n_til": n_til, "tam": int(tc.tam),
                      "z": z.tolist(), "indices": [[int(i) for i in tc.indices(zi[None, :])] for zi in z]})
    return saida


def ref_sarsa(passos=1500):
    cfg = dict(n_tiles=(8, 8, 3), n_til=8, alpha=0.3, gamma=0.99)
    ag = B.SarsaSemiGradiente(semente=0, **cfg)
    env = B.RescueBoatContEnv()
    heur = B.PoliticaHeuristica(env)
    rng = np.random.default_rng(3)
    obs, _ = env.reset(seed=5)
    sit, idx = ag._feat(obs)

    def escolhe(e, o):
        return int(rng.integers(4)) if rng.random() < 0.25 else heur(e, o)

    a = escolhe(env, obs)
    acao, acao2, resets, deltas = [], [], [], []
    for _ in range(passos):
        obs2, r, term, trunc, info = env.step(a)
        sit2, idx2 = ag._feat(obs2)
        a2 = escolhe(env, obs2)
        alvo = r if term else r + ag.gamma * ag.w[sit2][a2, idx2].sum()
        delta = alvo - ag.w[sit][a, idx].sum()
        ag.w[sit][a, idx] += ag.alpha * delta
        acao.append(a)
        acao2.append(a2)
        deltas.append(float(delta))
        fim = term or trunc
        resets.append(bool(fim))
        if fim:
            obs2, _ = env.reset()
            sit2, idx2 = ag._feat(obs2)
            a2 = escolhe(env, obs2)
        sit, idx, a = sit2, idx2, a2
    nz = np.flatnonzero(ag.w.ravel())
    amostra = nz[np.linspace(0, len(nz) - 1, 40).astype(int)]
    obs_q = []
    e2 = B.RescueBoatContEnv()
    o, _ = e2.reset(seed=11)
    for k in range(12):
        obs_q.append(o.tolist())
        o, *_ = e2.step(heur(e2, o))
    return {"cfg": {"n_tiles": list(cfg["n_tiles"]), "n_til": cfg["n_til"], "alpha": cfg["alpha"], "gamma": cfg["gamma"]},
            "semente_env": 5, "acao": acao, "acao2": acao2, "fim_episodio": resets, "deltas": deltas,
            "soma_w": float(ag.w.sum()), "n_nao_nulos": int(len(nz)),
            "amostra": [[int(i), float(ag.w.ravel()[i])] for i in amostra],
            "obs_q": obs_q, "q": [[float(ag.q(*ag._feat(np.array(o, dtype=np.float32)))[a]) for a in range(4)] for o in obs_q],
            "n_sucessos": int(sum(1 for d, f in zip(deltas, resets) if f)),
            }


def ref_mlp():
    import torch
    torch.manual_seed(0)
    rng = np.random.default_rng(1)
    obs = rng.random((40, 13)).astype(np.float32)
    saida = []
    for ativ, camadas in (("relu", [13, 16, 16, 4]), ("tanh", [13, 24, 4])):
        mods = []
        for i in range(len(camadas) - 1):
            mods.append(torch.nn.Linear(camadas[i], camadas[i + 1]))
            if i < len(camadas) - 2:
                mods.append(torch.nn.ReLU() if ativ == "relu" else torch.nn.Tanh())
        rede = torch.nn.Sequential(*mods)
        lin = [m for m in mods if isinstance(m, torch.nn.Linear)]
        with torch.no_grad():
            y = rede(torch.from_numpy(obs)).numpy()
        saida.append({"modelo": {"algo": "SINTETICO", "saida": "q", "ativacao": ativ,
                                 "camadas": [{"entrada": m.in_features, "saida": m.out_features,
                                              "W": [[f32(v) for v in l] for l in m.weight.detach().numpy()],
                                              "b": [f32(v) for v in m.bias.detach().numpy()]} for m in lin]},
                      "obs": obs.tolist(), "saidas": y.tolist(), "acoes": [int(a) for a in y.argmax(1)]})
    return saida


if __name__ == "__main__":
    ref = {"mdp_t2": ref_mdp(), "tile": ref_tile(), "sarsa": ref_sarsa(), "mlp_sintetico": ref_mlp()}
    (RAIZ / "tests" / "reference.json").write_text(json.dumps(ref, separators=(",", ":")))
    print("tests/reference.json:", (RAIZ / "tests" / "reference.json").stat().st_size, "bytes; V*(s0) =", round(ref["mdp_t2"]["v0"], 4),
          "| atualizações Sarsa:", len(ref["sarsa"]["acao"]), "| episódios concluídos:", ref["sarsa"]["n_sucessos"])
