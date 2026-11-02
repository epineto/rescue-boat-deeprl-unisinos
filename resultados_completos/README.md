# Resultados completos do Trabalho 3 (experimento de referência)

Arquivos lidos pelo notebook `Trabalho3-RL-Barco-Continuo.ipynb` (modos `rapido` e `salvos`). Foram gerados pelo experimento
completo (`codigo/experimentos_t3.py`, ~100 min em 4 núcleos de CPU) e pelo experimento suplementar
(`codigo/experimento_extra_t3.py`, PPO/A2C com 2 milhões de passos e sanidade no CartPole-v1). Nenhum arquivo foi editado à mão.

No Colab, o notebook procura esta pasta ao lado do próprio arquivo; se ela não existir, baixa os arquivos de
`https://github.com/epineto/rescue-boat-deeprl-unisinos` (pasta `resultados_completos/`). Também é possível enviar a pasta pelo painel de arquivos do Colab.

## Conteúdo

| Arquivo | Descrição |
|---|---|
| `info.json` | versões das bibliotecas, orçamentos, sementes, critério de escolha, parâmetros do ambiente e tempos da execução de referência |
| `melhores_configs.json` | configuração escolhida na busca para cada algoritmo, com os hiperparâmetros |
| `linhas_de_base.csv` | referências sem treino: tabular discretizada (política ótima do T2), heurística (BFS) e aleatória (500 episódios) |
| `busca_hiperparametros.csv` | **todas** as runs da busca: 4 algoritmos x 6 configurações x 3 sementes, 200 mil passos (72 linhas) |
| `busca_resumo_com_hp.csv` | resumo por configuração (média, desvio, IC95%, posição, configuração escolhida, hiperparâmetros) |
| `curvas_busca.csv`, `curvas_finais.csv` | curvas de aprendizado (avaliações periódicas da política gulosa) da busca e das runs finais |
| `resultado_final.csv` | runs finais: melhor configuração de cada algoritmo, 5 sementes novas, 400 mil passos (20 linhas) |
| `resultado_final_resumo.csv` | resumo das runs finais com as referências |
| `hipoteses.csv`, `auc_curvas.csv` | testes das hipóteses H1 a H3 e áreas sob as curvas de aprendizado |
| `trajetoria_referencia.json` | trajetórias de referência do ambiente (paridade com a implementação em JavaScript do laboratório) |
| `extra_*.csv`, `cartpole_*.csv`, `info_extra.json` | experimento suplementar: orçamento de 2 milhões de passos e sanidade da fábrica de agentes no CartPole-v1 |
| `modelos/final_<ALGO>_<config>_s<semente>.zip` | modelos treinados das runs finais (DQN, PPO e A2C; 5 sementes cada), usados nas trajetórias |

## Esquema dos CSV principais

`busca_hiperparametros.csv` e `resultado_final.csv`: `algo, config, semente, passos, ret` (retorno não descontado médio),
`ret_desc` (retorno descontado, gama = 0,99), `ret_desc_ic95_ep` (IC95% entre episódios), `sucesso`, `passos_ep`,
`toca_destrocos`, `tempo_s`, `hp` (hiperparâmetros em JSON). A avaliação usa a política gulosa em episódios com sementes
fixas (10000, 10001, ...), as mesmas para todos os agentes. O IC95% entre sementes (t de Student) é calculado pelo notebook.

## Como refazer

No notebook, `MODO = "completo"` (ou `MODO_T3=completo` na linha de comando) repete o protocolo inteiro em
`resultados_refeitos/`; `MODO = "rapido"` executa uma demonstração curta ao vivo e lê estes arquivos para as tabelas e figuras.
