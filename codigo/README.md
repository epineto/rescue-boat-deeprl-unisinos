# Código do experimento

Cópia literal dos arquivos do projeto do Trabalho 3 usados para gerar os resultados deste repositório.

| Arquivo | Função |
|---|---|
| `barco_continuo.py` | Ambiente `RescueBoatContEnv` (Gymnasium), MDP discreto do Trabalho 2, linhas de base (tabular discretizada, heurística, aleatória), Sarsa semi-gradiente com *tile coding* e utilitários de avaliação |
| `experimentos_t3.py` | Protocolo completo: linhas de base, busca de hiperparâmetros (DQN, PPO, A2C e Sarsa), execuções finais, curvas, hipóteses H1 a H3 e tabelas CSV |
| `experimento_extra_t3.py` | Experimento suplementar: PPO e A2C com 2 milhões de passos e sanidade da fábrica de agentes no CartPole-v1 |
| `testes_ambiente.py` | Testes do ambiente e do MDP |
| `requirements_t3.txt` | Versões das bibliotecas usadas (Python 3.12, CPU) |

```bash
pip install -r requirements_t3.txt
python testes_ambiente.py          # testes do ambiente
python experimentos_t3.py          # protocolo completo (cerca de 100 min em 4 núcleos); RAPIDO=1 para uma versão reduzida de poucos minutos
```

Os arquivos gerados (CSV e JSON) estão em [`../resultados_completos/`](../resultados_completos/). Os arquivos por execução (`runs/`) não fazem parte deste repositório; os modelos finais do SB3 (5 sementes por algoritmo) estão em [`../resultados_completos/modelos/`](../resultados_completos/modelos/).
