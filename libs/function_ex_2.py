import heapq
from collections import defaultdict

def floyd_warshall(graph, n):
    """
    Compute shortest paths between all pairs of nodes using Floyd-Warshall.
    
    Args:
        graph (dict): Adjacency list representation of the graph.
        n (int): Number of nodes.
    
    Returns:
        dict: Shortest paths as a dictionary { (u, v): distance }.
    """
    dist = { (u, v): float('inf') for u in range(n) for v in range(n) }
    for u in range(n):
        dist[(u, u)] = 0
    for u in graph:
        for v, weight in graph[u]:
            dist[(u, v)] = weight
    
    for k in range(n):
        for i in range(n):
            for j in range(n):
                if dist[(i, j)] > dist[(i, k)] + dist[(k, j)]:
                    dist[(i, j)] = dist[(i, k)] + dist[(k, j)]
    
    return dist

def closeness_centrality(graph, n):
    """
    Calculate closeness centrality for all nodes.
    
    Args:
        graph (dict): Adjacency list representation of the graph.
        n (int): Number of nodes.
    
    Returns:
        dict: Closeness centrality for each node.
    """
    dist = floyd_warshall(graph, n)
    centrality = {}
    
    for u in range(n):
        total_distance = sum(dist[(u, v)] for v in range(n) if dist[(u, v)] < float('inf'))
        if total_distance > 0:
            centrality[u] = (n - 1) / total_distance
        else:
            centrality[u] = 0.0  # Node is isolated
    
    return centrality

def betweenness_centrality(graph, n):
    """
    Calculate betweenness centrality for all nodes.
    
    Args:
        graph (dict): Adjacency list representation of the graph.
        n (int): Number of nodes.
    
    Returns:
        dict: Betweenness centrality for each node.
    """
    centrality = { u: 0 for u in range(n) }
    
    for source in range(n):
        stack = []
        predecessors = { u: [] for u in range(n) }
        shortest_paths = { u: 0 for u in range(n) }
        shortest_paths[source] = 1
        distance = { u: float('inf') for u in range(n) }
        distance[source] = 0
        queue = []
        heapq.heappush(queue, (0, source))
        
        while queue:
            _, u = heapq.heappop(queue)
            stack.append(u)
            for v, weight in graph.get(u, []):
                if distance[v] > distance[u] + weight:
                    distance[v] = distance[u] + weight
                    heapq.heappush(queue, (distance[v], v))
                    predecessors[v] = [u]
                    shortest_paths[v] = shortest_paths[u]
                elif distance[v] == distance[u] + weight:
                    predecessors[v].append(u)
                    shortest_paths[v] += shortest_paths[u]
        
        dependencies = { u: 0 for u in range(n) }
        while stack:
            u = stack.pop()
            for p in predecessors[u]:
                dependencies[p] += (shortest_paths[p] / shortest_paths[u]) * (1 + dependencies[u])
            if u != source:
                centrality[u] += dependencies[u]
    
    for u in centrality:
        centrality[u] /= 2  # Undirected graph adjustment
    
    return centrality
