def prim_alg(G):
    
    list_of_edges = list(G.edges(data = True))
    number_of_nodes = len(list(G.nodes()))

    nodes = [0]
    minn = 'inf'
    min_edge = list_of_edges[0]
    tree = []

    while len(nodes) < number_of_nodes:
        minn = 'inf'
        for edge in list_of_edges:
            if (edge[0] in nodes) ^ (edge[1] in nodes) and (minn == 'inf' or edge[2]['weight'] < minn):
                minn = edge[2]['weight']
                min_edge = edge
        list_of_edges.remove(min_edge)
        tree.append(min_edge)
        if min_edge[0] in nodes:
            nodes.append(min_edge[1])
        else:
            nodes.append(min_edge[0])

    return tree

def fw_alg(G):

    list_of_edges = list(G.edges(data = True))
    number_of_nodes = len(list(G.nodes()))

    graph = {i : {j : 'inf' for j in range(number_of_nodes)} for i in range(number_of_nodes)}
    for edge in list_of_edges:
        graph[edge[0]][edge[1]] = edge[2]['weight']
    
    for node in range(number_of_nodes):
        line = []
        row = []
        for i in range(number_of_nodes):
            if graph[node][i] != 'inf':
                line.append(i)
        for i in range(number_of_nodes):
            if graph[i][node] != 'inf':
                row.append(i)
        
        for i in line:
            for j in row:
                if graph[node][i] != 'inf' and graph[j][node] != 'inf' and \
                    (graph[j][i] == 'inf' or graph[node][i] + graph[j][node] < graph[j][i]):
                    graph[j][i] = graph[node][i] + graph[j][node]
        
        graph[node][node] == 0
    
    return graph
