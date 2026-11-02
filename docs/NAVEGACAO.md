# Navegação do laboratório (roteiro para a apresentação)

Cada seção tem uma âncora fixa que pode ser aberta direto (`…/#ambiente`, etc.). A barra do topo troca de seção; **PT/EN** troca o idioma e o botão de lua/sol troca o tema.

| Âncora | O que é | O que mostrar em 20 s |
|---|---|---|
| `#intro` | Problema, parâmetros, agentes e números-chave | Ler as 3 linhas do problema e apontar o episódio da heurística em andamento e o quadro do MDP |
| `#ambiente` | Controle manual do barco | Clicar no mapa, mover com as setas até a correnteza (o medidor sobe e desce), mostrar a observação de 13 números e entregar uma pessoa ao abrigo |
| `#politicas` | Políticas lado a lado | **Reproduzir** com a heurística, DQN, PPO e A2C no mesmo episódio; trocar a semente de treino de uma rede; ligar "Continuar em novos episódios"; **Comparar** com N = 100 e ler as médias com IC95% |
| `#treinar` | Sarsa no navegador | **Treinar** com os padrões (curva sobe em segundos); trocar para a grade 12 × 12 × 4 com α = 0,1 (aprende devagar) e comparar; **Reproduzir episódio** e ir a Políticas para ver o agente ao lado das redes |
| `#resultados` | Explorador dos resultados | Resultado final (barras com sementes), Busca completa (filtrar por algoritmo, ordenar, mostrar sementes), Curvas (faixa de IC95% e linhas de referência), Hipóteses com a ressalva de baixo poder, Orçamento maior (A2C 0 de 3 e PPO 1 de 3 com 2 milhões de passos) |
| `#sobre` | Como rodar, testes, aulas, limitações | Mostrar a tabela Aula → seção e as limitações |

Dica: abrir o laboratório já na seção desejada pelo link com âncora, por exemplo `https://epineto.github.io/rescue-boat-deeprl-unisinos/#politicas`.
