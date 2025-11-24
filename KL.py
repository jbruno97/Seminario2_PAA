import networkx as nx

# Função auxiliar para calcular o Custo Total de Comunicação (Cut Ponderado)
def calcular_cut_ponderado(graph, particao_A, particao_B):
    """
    Calcula o custo total de comunicação (soma dos pesos das arestas cortadas).
    Este custo simula a latência agregada de consultas cross-shard.
    """
    cut_cost = 0
   
    for u, v, data in graph.edges(data=True):
        weight = data.get('weight', 1) # Pega o peso da aresta (default é 1 se não houver peso)
       
        # Verifica se os nós estão em partições diferentes
        if (u in particao_A and v in particao_B) or (u in particao_B and v in particao_A):
            cut_cost += weight
           
    return cut_cost

# 1. Criação do Grafo Ponderado (Entidades e Frequência de Consultas)
G = nx.Graph()

# Definimos duas comunidades de dados (A e B)
# Arestas internas são de baixo custo (peso 1)
G.add_edges_from([
    ('U1', 'U2', {'weight': 1}), ('U1', 'U3', {'weight': 1}), ('U2', 'U3', {'weight': 1}),
    ('P4', 'P5', {'weight': 1}), ('P4', 'P6', {'weight': 1}), ('P5', 'P6', {'weight': 1})
])

# Arestas de alto custo (simulando consultas frequentes entre clusters)
# Estas arestas DEVEM ser internas, mas estão cruzando na partição inicial ruim.
G.add_edges_from([
    ('U2', 'P4', {'weight': 10}), # Alto custo de comunicação!
    ('U3', 'P5', {'weight': 10}),
])

# 2. Partição Inicial Ruim (Sharding não otimizado)
# Shard 1 (A): {U1, U2, P4}
# Shard 2 (B): {U3, P5, P6}
A_inicial = {'U1', 'U2', 'P4'}
B_inicial = {'U3', 'P5', 'P6'}
particao_inicial = (A_inicial, B_inicial)

print("--- Cenário: Otimização de Sharding com Grafo Ponderado ---")
print(f"Número Total de Entidades (Nós): {G.number_of_nodes()}")
print(f"Custo Máximo de Comunicação por Aresta: 10 (U2-P4 e U3-P5)")
print("-" * 60)

# Custo de comunicação inicial (Cut Ponderado)
cut_inicial = calcular_cut_ponderado(G, A_inicial, B_inicial)
print(f"Custo Inicial de Comunicação Cross-Shard: {cut_inicial}")
print(f"Shard 1 (A): {A_inicial}")
print(f"Shard 2 (B): {B_inicial}")
print("-" * 60)


# 3. Aplicação da Heurística Kernighan–Lin para Refinamento
# O KL vai trocar os nós para minimizar o cut PONDERADO.
particao_melhorada = nx.algorithms.community.kernighan_lin_bisection(
    G,
    partition=particao_inicial,
    max_iter=5 # Rodadas de refinamento
)

A_final, B_final = particao_melhorada

# Custo de comunicação após o refinamento
cut_final = calcular_cut_ponderado(G, A_final, B_final)

print("--- Resultados do Refinamento Kernighan-Lin ---")
print(f"Novo Custo de Comunicação Cross-Shard (Final): {cut_final}")
print(f"Shard Otimizado 1 (A): {A_final}")
print(f"Shard Otimizado 2 (B): {B_final}")
print("-" * 60)

# Análise do Ganho
if cut_final < cut_inicial:
    reducao_percentual = (1 - cut_final / cut_inicial) * 100
    print(f" Sucesso! Redução no Custo de Comunicação (Cut) de {cut_inicial} para {cut_final}.")
    print(f"Isso representa uma redução de {reducao_percentual:.2f}% na latência de consultas cross-shard.")
else:
    print(f" Convergência: A partição não foi melhorada (Cut={cut_inicial}).")