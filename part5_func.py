# Girvan-Newman Algorithm for Community Detection

import heapq
from collections import defaultdict, deque
import matplotlib.pyplot as plt
import numpy as np
import copy
import networkx as nx   
import pandas as pd
import itertools

def find_connected_components(graph, method="dfs_recursive"):
    """
    Find all connected components in an undirected graph using DFS.

    Args:
        graph (dict): Adjacency list representing the graph.
        method (str): 'dfs_recursive' or 'dfs_iterative'.

    Returns:
        list: List of connected components.
    """
    visited = set()
    components = []

    # Recursive version
    # Source: https://www.geeksforgeeks.org/depth-first-search-or-dfs-for-a-graph/
    def dfs_recursive(node, component):
        """
        Recursive Depth-First Search (DFS) function to explore a connected component in the graph.
        
        This function visits all nodes reachable from the given 'node' and adds them to the same 
        connected component. It uses recursion to traverse all unvisited neighbors.
        """
        visited.add(node)  # Mark the current node as visited
        component.append(node)  # Add the current node to the component
        
        #print(f"Step {step_counter}:")
        #print(f"  Current Node: {node}")
        #print(f"  Visited Nodes: {visited}")
        #print(f"  Current Component: {component}")
        #print("----------------------------------")
        #step_counter += 1
    
        # Recursively visit all unvisited neighbors
        for neighbor in graph[node]:
            if neighbor not in visited:  # Process only unvisited nodes
                dfs_recursive(neighbor, component)

    # Iterative version
    def dfs_iterative(start):
        """
        Iterative Depth-First Search using an explicit stack.
        """
        stack = [start]
        component = []  # Local list to store the component nodes

        while stack:
            node = stack.pop()  # Pop the most recently added node (LIFO)
            if node not in visited:
                visited.add(node)
                component.append(node)
                
                # Add unvisited neighbors to the stack
                for neighbor in reversed(graph[node]):
                    if neighbor not in visited:
                        stack.append(neighbor)
        return component

    # Which one to use? Both give same results, O notation is the same but Recursive is more intuitive but may have some problem in larger dataset due to recursion limit that on Py is around 1000 calls

    # Explore all nodes
    for node in graph:
        if node not in visited:
            if method == "dfs_recursive":
                component = []
                dfs_recursive(node, component)
                components.append(component)
            elif method == "dfs_iterative":
                component = dfs_iterative(node)
                components.append(component)
            else:
                raise ValueError("Invalid method. Use 'dfs_recursive' or 'dfs_iterative'.")

    return components


# Source: https://www.geeksforgeeks.org/breadth-first-search-or-bfs-for-a-graph/
def bfs_shortest_paths(graph, source):
    """
    Perform Breadth-First Search (BFS) to calculate the shortest paths from a given source node to all other nodes.
    BFS explores vertices level by level, using a queue data structure. This implementation also tracks the parents 
    of each node to reconstruct shortest paths.
    """
    queue = deque([source]) # Initialize a queue for BFS, starting with the source node

    distances = {node: float('inf') for node in graph} # Set the distance to all nodes as infinity (unvisited)
    distances[source] = 0 # Set the distance to the source node as 0 (starting point).

    # Initialize the parents dictionary:
    # - Tracks all parents for each node along shortest paths.
    parents = defaultdict(list) # parents dict tracks all parents for each node along shortest paths
    
    #step = 0

    while queue: # until all vertices are reachable
        # Dequeue the current node
        current = queue.popleft() # remove from queue

        #print(f"Step {step}:")
        #print(f"  Current Node: {current}")
        #print(f"  Queue: {list(queue)}")
        #print(f"  Distances: {distances}")
        #print(f"  Parents: {dict(parents)}")

        # Explore all adjacent vertices of the current node
        for neighbor in graph[current]:
            if distances[neighbor] == float('inf'): # so not visited
                distances[neighbor] = distances[current] + 1 # update as one level deeper than current node
                queue.append(neighbor) # add to queue to process later

            if distances[neighbor] == distances[current] + 1: # if distance is exactly one level deeper than the current node
                parents[neighbor].append(current) # Add the current node as a parent of the neighbor

    return distances, parents

# Source: https://symbio6.nl/en/blog/analysis/betweenness-centrality (only for the concept)
from collections import defaultdict

def calculate_betweenness_bfs(graph):
    """
    Calculate edge betweenness centrality for all edges in an undirected network.

    Betweenness centrality measures the "bridgeness" of an edge by evaluating the fraction
    of shortest paths between all pairs of nodes in the network that pass through that edge.
    """

    betweenness = defaultdict(int) # Dict to store BC values for edges

    for node in graph:
        #print(f"\nProcessing Source Node: {node}")
        distances, parents = bfs_shortest_paths(graph, node) # BFS return shortest paths and their parent relationships
        #print(f"  BFS Distances from {node}: {distances}")
        #print(f"  BFS Parents from {node}: {dict(parents)}")
        node_flow = {n: 1 for n in graph} # Initialize flow for all nodes as 1 (default contribution to shortest paths)

        nodes_by_distance = sorted(distances, key=distances.get, reverse=True) # Process nodes with farthest nodes first
        #print(f"  Nodes by Distance (Reverse): {nodes_by_distance}")

        # Backtrack from farthest nodes to distribute flow across edges
        for target in nodes_by_distance: 
            for parent in parents[target]:
                # Define the edge between parent and target
                edge = tuple(sorted((parent, target)))  # Define edge between parent and target. Sort to avoid duplicates
                
                # Distribute flow proportionally across all shortest paths to the target
                flow = node_flow[target] / len(parents[target])  # Split flow equally
                betweenness[edge] += flow  # Add flow to the edge's betweenness
                node_flow[parent] += flow  # Pass flow back to the parent node
                #print(f"    Updated Betweenness for Edge {edge}: {betweenness[edge]}")
                #print(f"    Flow Distribution: Node {parent} Flow: {node_flow[parent]}")
                
    #for edge, score in betweenness.items():
        #print(f"  Edge {edge}: {score}")

    return betweenness

# Dijkstra's Algorithm 
def dijkstra_adj_list(graph, start_node):
    """
    Dijkstra's Algorithm to find the shortest paths from a start node
    in an adjacency list representation of a graph.
    """
    distances = {node: float('inf') for node in graph}
    distances[start_node] = 0
    previous_nodes = {node: None for node in graph}
    priority_queue = [(0, start_node)]

    while priority_queue:
        current_distance, current_node = heapq.heappop(priority_queue)
        for neighbor in graph[current_node]:
            weight = 1
            distance = current_distance + weight
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                previous_nodes[neighbor] = current_node
                heapq.heappush(priority_queue, (distance, neighbor))

    return distances, previous_nodes

# Betweenness Centrality Calculation: Dijkstra based
def calculate_betweenness_dijkstra(graph):
    """
    Calculate edge betweenness centrality using Dijkstra's Algorithm.
    """
    betweenness = defaultdict(float)

    for node in graph:
        #print(f"\n[DEBUG] Running Dijkstra from source node: {node}")
        distances, parents = dijkstra_adj_list(graph, node)
        node_flow = {n: 1 for n in graph}

        # Sort nodes by distance from the source (farthest nodes processed first)
        nodes_by_distance = sorted(distances, key=distances.get, reverse=True)
        #print(f"[DEBUG] Nodes by distance (farthest to closest): {nodes_by_distance}")

        for target in nodes_by_distance:
            if parents[target] is not None:
                edge = tuple(sorted((parents[target], target)))  # Avoid duplicates
                flow = node_flow[target] / 1  # In unweighted graphs, flow = 1
                betweenness[edge] += flow
                node_flow[parents[target]] += flow
                #print(f"[DEBUG] Edge {edge} receives flow: {flow:.2f}")

 #   print("\n[DEBUG] Final Betweenness Centrality:")
  #  for edge, centrality in betweenness.items():
   #     print(f"  Edge {edge}: {centrality:.2f}")

    return betweenness
    
# Source: https://memgraph.github.io/networkx-guide/algorithms/community-detection/girvan-newman/
# Girvan-Newman Algorithm
def custom_girvan_newman(graph, betweenness_method="bfs", component_method="dfs_recursive"):
    """
    Girvan-Newman algorithm for community detection.
    Args:
        graph (dict): Input graph as adjacency list.
        betweenness_method (str): 'bfs' or 'dijkstra' for edge betweenness.
        component_method (str): 'dfs_recursive' or 'dfs_iterative' for connected components.
    Returns:
        tuple: (communities, removed_edges)
    """
    graph_copy = copy.deepcopy({node: list(neighbors) for node, neighbors in graph.adjacency()})
    removed_edges = []
    iteration = 0

    while True:
        print(f"\n[DEBUG] Iteration {iteration}: Calculating betweenness centrality...")

        # Calculate betweenness centrality with choosed method
        if betweenness_method == "bfs":
            betweenness = calculate_betweenness_bfs(graph_copy)
        elif betweenness_method == "dijkstra":
            betweenness = calculate_betweenness_dijkstra(graph_copy)
        else:
            raise ValueError("Invalid betweenness method. Choose 'bfs' or 'dijkstra'.")

        # Stop if no edges remain
        if not betweenness:
            break

        # Remove the edge with the highest betweenness
        edge_to_remove = max(betweenness, key=betweenness.get)
        graph_copy[edge_to_remove[0]].remove(edge_to_remove[1])
        graph_copy[edge_to_remove[1]].remove(edge_to_remove[0])
        removed_edges.append(edge_to_remove)
        print(f"[DEBUG] Removed edge: {edge_to_remove}")

        # Find connected components
        communities = find_connected_components(graph_copy, method=component_method)
        print(f"[DEBUG] Communities: {communities}")

        # Terminate when more than one community is detected
        if len(communities) > 1:
            break

        iteration += 1

    return communities, removed_edges

# Visualization
def visualize_communities(graph, communities, removed_edges=[]):
    """
    Visualize the graph and its communities.
    """
    # Assign colors to nodes based on their community
    colors = plt.cm.rainbow(np.linspace(0, 1, len(communities)))
    color_map = {}
    for i, community in enumerate(communities):
        for node in community:
            color_map[node] = colors[i]

    pos = nx.spring_layout(graph)

    plt.figure(figsize=(10, 10))
    nx.draw_networkx_nodes(graph, pos, node_size=500, 
                           node_color=[color_map[node] for node in graph.nodes()],
                           edgecolors='black')
    nx.draw_networkx_labels(graph, pos, font_size=12, font_color='white')

    # Draw edges excluding removed edges
    edges_to_draw = [(u, v) for u, v in graph.edges() if (u, v) not in removed_edges and (v, u) not in removed_edges]
    nx.draw_networkx_edges(graph, pos, edgelist=edges_to_draw, width=2, alpha=0.5)

    plt.title("Girvan-Newman Communities")
    plt.axis("off")
    plt.show()

# Graph Class (Adjacency List Implementation)
class Graph:
    def __init__(self, edges):
        self.graph = defaultdict(set)
        self.edges = edges
        self.nodes = set()
        for u, v in edges:
            self.add_edge(u, v)

    def add_edge(self, u, v):
        self.graph[u].add(v)
        self.graph[v].add(u)
        self.nodes.update([u, v])

    def degree(self, node):
        return len(self.graph[node])

    def adjacency_matrix(self):
        nodes = sorted(self.nodes)
        n = len(nodes)
        node_idx = {node: i for i, node in enumerate(nodes)}
        adj_matrix = np.zeros((n, n))
        for u in nodes:
            for v in self.graph[u]:
                adj_matrix[node_idx[u], node_idx[v]] = 1
        return adj_matrix, nodes
    
    def adjacency(self):
        for node, neighbors in self.graph.items():
            yield node, list(neighbors)

def calculate_modularity(adj_matrix, communities):
    """
    It evaluates the modularity of the resulting communities.
    """
    m = np.sum(adj_matrix) / 2
    Q = 0.0
    for community in communities:
        # For every pair of nodes in the same community i,j
        for i in community:
            for j in community:
                A_ij = adj_matrix[i, j] # Adjacency matrix value
                k_i = np.sum(adj_matrix[i]) # Degree of node i
                k_j = np.sum(adj_matrix[j]) # Degree of node j
                Q += A_ij - (k_i * k_j) / (2 * m) # Modularity formula numerator
    return Q / (2 * m) # Normalize by 2m

# Test the Girvan-Newman Algorithm

import time
from networkx.algorithms.community import girvan_newman as nx_girvan_newman

# --- Custom Girvan-Newman ---
def test_girvan_newman(graph, custom_method, component_method):
    adj_matrix, nodes = graph.adjacency_matrix()  # Extract the adjacency matrix
    print(f"\n ⚠️--- Testing with Custom Method: {custom_method} | Component Method: {component_method} ---")
    
    # Measure time for Custom Girvan-Newman
    try:
        start_time = time.time()
        custom_communities, custom_removed_edges = custom_girvan_newman(graph, custom_method, component_method)
        end_time = time.time()
        custom_time = end_time - start_time
        print(f"⏱️ Custom Girvan-Newman Time: {custom_time:.4f} seconds")
        print("Custom Communities:", custom_communities)
        print("Modularity:", calculate_modularity(adj_matrix, custom_communities))
        print("Custom Removed Edges:", custom_removed_edges)
    except NotImplementedError as e:
        print(f"Custom Girvan-Newman: {e}")
        custom_time = None

    # Measure time for NetworkX Girvan-Newman
    print("\n ⚠️Running NetworkX Girvan-Newman Algorithm...")
    start_time = time.time()
    nx_gen = nx_girvan_newman(nx.Graph(graph.graph))  # Generate community splits
    nx_communities = next(iter(nx_gen))  # Extract first split
    end_time = time.time()
    nx_time = end_time - start_time
    nx_communities = [sorted(list(c)) for c in nx_communities]  # Sort nodes in each community
    print(f"⏱️ NetworkX Girvan-Newman Time: {nx_time:.4f} seconds")
    print("NetworkX Communities:", nx_communities)

    # Comparison
    if custom_time:
        print("\n--- Comparison Results ---")
        if sorted([sorted(c) for c in custom_communities]) == sorted(nx_communities):
            print("✅ Communities match!")
        else:
            print("❌ Communities do NOT match!")
        print(f"⏱️ Time Difference: {abs(custom_time - nx_time):.4f} seconds")

# Spectral Clustering

# KMeans and Kmeans++ for Section 2.3 for HW4
# Source: https://github.com/emanueleiacca/ADM-HW4/blob/main/functions/functions.py#L329
def initialize_centroids(data, k, method="random",seed=42):
    """
    Initialize centroids using the chosen method.
    Parameters:
        - data: NumPy array of data points.
        - k: Number of clusters.
        - method: "random" for basic initialization or "kmeans++" for K-me ()ans++ initialization.
    """
    if method == "random":
        np.random.seed(seed)  # Set the random seed for reproducibility
        # Randomly select k unique indices
        indices = np.random.choice(data.shape[0], k, replace=False)
        return data[indices]

    elif method == "kmeans++":
        np.random.seed(seed)
        # K-means++ initialization
        centroids = [data[np.random.choice(data.shape[0])]]  # First centroid randomly chosen
        for _ in range(1, k):
            # Compute distances from nearest centroid for all points
            distances = np.min([np.linalg.norm(data - centroid, axis=1) for centroid in centroids], axis=0)
            # Compute probabilities proportional to squared distances
            probabilities = distances ** 2 / np.sum(distances ** 2)
            # Choose next centroid based on probabilities
            next_centroid_index = np.random.choice(data.shape[0], p=probabilities)
            centroids.append(data[next_centroid_index])
        return np.array(centroids)

    else:
        raise ValueError("Invalid method. Choose 'random' or 'kmeans++'.")

def compute_distance(point, centroids):
    """Compute the distance of a point to all centroids and return the nearest one."""
    distances = np.linalg.norm(centroids - point, axis=1)
    return np.argmin(distances)  # Return the index of the closest centroid

def assign_clusters(data, centroids):
    """Assign each point to the nearest centroid."""
    clusters = []
    for point in data:
        cluster_id = compute_distance(point, centroids)
        clusters.append(cluster_id)
    return np.array(clusters)

def update_centroids(data, clusters, k):
    """Update centroids as the mean of points in each cluster."""
    new_centroids = []
    for cluster_id in range(k):
        cluster_points = data[clusters == cluster_id]
        if len(cluster_points) > 0:
            new_centroids.append(cluster_points.mean(axis=0))
        else:  # Handle empty cluster
            new_centroids.append(np.zeros(data.shape[1]))
    return np.array(new_centroids)

def kmeans(data, k, method="random", max_iterations=100, tolerance=1e-4, seed = 42):
    """
    K-means clustering algorithm with option for basic or K-means++ initialization.
    Parameters:
        - data: NumPy array of data points.
        - k: Number of clusters.
        - method: "random" for basic K-means or "kmeans++" for K-means++.
        - max_iterations: Maximum number of iterations.
        - tolerance: Convergence tolerance.
    """
    # Initialize centroids
    centroids = initialize_centroids(data, k, method=method)

    for iteration in range(max_iterations):
        # Assign clusters
        clusters = assign_clusters(data, centroids)

        # Update centroids
        new_centroids = update_centroids(data, clusters, k)

        # Check for convergence
        if np.all(np.abs(new_centroids - centroids) < tolerance):
            print(f"Converged at iteration {iteration}")
            break

        centroids = new_centroids

    return centroids, clusters

from sklearn.preprocessing import normalize
import numpy as np
from collections import defaultdict

# Source: https://rahuljain788.medium.com/implementing-spectral-clustering-from-scratch-a-step-by-step-guide-9643e4836a76
def spectral_clustering(graph, k):
    """
    Spectral Clustering on a graph using normalized Laplacian and custom K-means.
    """
    # Adjacency and Degree Matrices
    adj_matrix, nodes = graph.adjacency_matrix()
    degrees = np.diag(adj_matrix.sum(axis=1))

    # Normalized Laplacian
    D_inv_sqrt = np.diag(1.0 / np.sqrt(np.diag(degrees)))
    D_inv_sqrt = np.nan_to_num(D_inv_sqrt)  # Handle division by zero
    L = np.eye(len(nodes)) - D_inv_sqrt @ adj_matrix @ D_inv_sqrt

    # Eigenvalue Decomposition
    eigenvalues, eigenvectors = np.linalg.eigh(L)
    sorted_indices = np.argsort(eigenvalues)  # Sort eigenvalues
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]
    
    print("Sorted Eigenvalues:", eigenvalues[:k+1])

    # Select and normalize eigenvectors corresponding to smallest non-trivial eigenvalues
    k_smallest_eigenvectors = eigenvectors[:, 1:k+1]  # Skip the first trivial eigenvector
    k_smallest_eigenvectors = normalize(k_smallest_eigenvectors, axis=1)

    # Run K-Means Clustering
    centroids, cluster_assignments = kmeans(k_smallest_eigenvectors, k, method="kmeans++", seed=42)

    # Group Nodes into Communities
    communities = defaultdict(list)
    for i, cluster_id in enumerate(cluster_assignments):
        communities[cluster_id].append(nodes[i])

    return list(communities.values())

# Louvain Algorithm

# Source: https://users.ece.cmu.edu/~lowt/papers/Louvain_accepted.pdf

def louvain_cluster(adj_matrix, max_iter=10):
    """
    Simplified version of the Louvain clustering algorithm using NumPy. 
    The algorithm aims to detect communities in a graph by iteratively optimizing the modularity of the graph. 
    """
    n = adj_matrix.shape[0]  # Number of nodes
    degrees = np.sum(adj_matrix, axis=1)
    inv_m = 1.0 / np.sum(degrees)  # 2m for modularity normalization
    communities = np.arange(n)  # Initialize each node in its own community

    def modularity_gain(node, target_comm, curr_comm):
        """Compute the modularity gain of moving 'node' to 'target_comm'."""
        k_i = degrees[node]
        delta_q = 0.0
        # Direct Contributions
        for neighbor, weight in enumerate(adj_matrix[node]): # It iterates over all neighbors of the node
            if weight > 0:
                # Depending by where the neighbor belongs, add edge weight to the modularity gain
                if communities[neighbor] == target_comm:
                    delta_q += weight
                if communities[neighbor] == curr_comm:
                    delta_q -= weight
        # Indirect Contributions
        # Compute the sum of degrees of nodes in the target and current communities
        sum_in_target = np.sum(degrees[communities == target_comm])
        sum_in_curr = np.sum(degrees[communities == curr_comm])
        # Adjusted formula
        delta_q -= k_i * (sum_in_target - k_i) * inv_m
        delta_q += k_i * (sum_in_curr - k_i) * inv_m
        return delta_q

    # Iterative Community Refinement
    for iteration in range(max_iter): # Either max_iter or until no nodes are moved
        moved = False
        for node in range(n): # for each node
            curr_comm = communities[node]
            max_gain = 0
            best_comm = curr_comm
 
            # Evaluate modularity gain for moving it to each neighboring community
            for neighbor, weight in enumerate(adj_matrix[node]):
                if weight > 0 and communities[neighbor] != curr_comm:
                    target_comm = communities[neighbor]
                    gain = modularity_gain(node, target_comm, curr_comm)
                    if gain > max_gain:
                        max_gain = gain
                        best_comm = target_comm

            # Reassign the node to the best community
            if best_comm != curr_comm:
                communities[node] = best_comm
                moved = True

        if not moved:  # Stop if no nodes were moved
            break

    return extract_communities(communities)

def extract_communities(communities):
    """Group nodes by their community assignments."""
    community_groups = defaultdict(list)
    for node, comm in enumerate(communities):
        community_groups[comm].append(node)
    return list(community_groups.values())

# Additional Metrics

def lambiotte_coefficient(adj_matrix, communities):
    """
    Compute the Lambiotte coefficient for each node.
    """
    n = adj_matrix.shape[0]
    node_importance = {}

    for community in communities: # For each community
        for node in community: # For each node in the community
            k_in = np.sum([adj_matrix[node, neighbor] for neighbor in community]) # Internal degree
            k_total = np.sum(adj_matrix[node])  # Total degree
            L = k_in / k_total if k_total > 0 else 0 # Lambiotte formula # Avoid division by zero
            node_importance[node] = L
    
    return node_importance

def clauset_parameter(adj_matrix, communities):
    """
    Compute Clauset's parameter for each community.
    """
    community_quality = {}

    for idx, community in enumerate(communities): # For each community
        E_in = sum(adj_matrix[i, j] for i in community for j in community if i != j) / 2  # Internal edges
        E_out = sum(adj_matrix[i, j] for i in community for j in range(adj_matrix.shape[0]) if j not in community) # External edges
        
        Q = E_in / (E_in + E_out) if (E_in + E_out) > 0 else 0  # Clauset's formula # Avoid division by zero
        community_quality[idx] = Q

    return community_quality


# Pre implemented functions for Louvain and Spectral Clustering

import numpy as np
from community import community_louvain  # python-louvain package
from sklearn.cluster import SpectralClustering
import networkx as nx
from collections import defaultdict
import time

# Pre-implemented Spectral Clustering
def pre_implemented_spectral(graph, k):
    adj_matrix = nx.to_numpy_array(graph)
    nodes = list(graph.nodes())

    sc = SpectralClustering(n_clusters=k, affinity='precomputed', random_state=42)
    labels = sc.fit_predict(adj_matrix)
    
    communities = defaultdict(list)
    for i, label in enumerate(labels):
        communities[label].append(nodes[i])
    return list(communities.values())

# Pre-implemented Louvain Method
def pre_louvain_method(nx_graph):
    partition = community_louvain.best_partition(nx_graph)
    grouped = defaultdict(list)
    for node, comm in partition.items():
        grouped[comm].append(node)
    return list(grouped.values())

# Compare method and evaluate metrics
from tabulate import tabulate 

def compare_methods(custom_method, pre_method, name):
    print(f"\n--- Comparing {name} ---")
    print("Custom Communities:", custom_method)
    print("Pre-Implemented Communities:", pre_method)

    if sorted([sorted(c) for c in custom_method]) == sorted([sorted(c) for c in pre_method]):
        print(f"✅ {name} Communities Match!")
    else:
        print(f"❌ {name} Communities Do NOT Match!")

def compare_communities_overlap(custom, pre, name):

    print(f"\n--- {name} Overlap Comparison ---")

    overlap_data = []

    for i, custom_comm in enumerate(custom):
        overlap_scores = []
        for j, pre_comm in enumerate(pre):
            overlap = len(set(custom_comm) & set(pre_comm))
            overlap_scores.append((j, overlap))

        overlap_scores = sorted(overlap_scores, key=lambda x: -x[1])
        best_match = overlap_scores[0]
        
        overlap_data.append([
            f"Custom {i}", 
            f"Pre-Implemented {best_match[0]}",
            len(custom_comm),  # Custom community size
            len(pre[best_match[0]]),  # Best match size
            best_match[1]  # Overlap count
        ])

    headers = ["Custom Community", "Best Match", "Custom Size", "Best Match Size", "Overlap Nodes"]
    print(tabulate(overlap_data, headers=headers, tablefmt="grid"))

# --- Metrics Calculation Function ---
def evaluate_community_metrics(adj_matrix, communities, title):
    """
    Calculate and display Lambiotte Coefficient for nodes and Clauset's Parameter for communities.
    :param adj_matrix: NumPy adjacency matrix of the graph.
    :param communities: List of detected communities.
    :param title: Title for the display output.
    """
    n = adj_matrix.shape[0]
    node_degree = np.sum(adj_matrix, axis=1)

    # --- Calculate Lambiotte Coefficient ---
    lambiotte_coeff = {}
    for node in range(n):
        community = next(c for c in communities if node in c)
        internal_edges = np.sum(adj_matrix[node][community])
        lambiotte_coeff[node] = internal_edges / node_degree[node] if node_degree[node] > 0 else 0

    # --- Calculate Clauset's Parameter ---
    clauset_param = {}
    for i, community in enumerate(communities):
        internal_edges = sum(
            adj_matrix[node_i][node_j] for node_i in community for node_j in community if node_i != node_j
        )
        external_edges = sum(
            adj_matrix[node_i][node_j] for node_i in community for node_j in range(n) if node_j not in community
        )
        clauset_param[i] = internal_edges / (internal_edges + external_edges) if (internal_edges + external_edges) > 0 else 0

    # --- Display Results in Tabular Format ---
    print(f"\n--- {title} ---")
    print("\nLambiotte Coefficient (Node Importance):")
    lambiotte_table = [[node, f"{lambiotte_coeff[node]:.4f}"] for node in sorted(lambiotte_coeff.keys())]
    print(tabulate(lambiotte_table, headers=["Node", "Lambiotte Coefficient"], tablefmt="grid"))

    print("\nClauset's Parameter (Community Strength):")
    clauset_table = [[f"Community {i}", f"{clauset_param[i]:.4f}"] for i in clauset_param.keys()]
    print(tabulate(clauset_table, headers=["Community", "Clauset's Parameter"], tablefmt="grid"))

# Part 2 of the HW

from collections import defaultdict

def df_to_adjacency_list(df):
    """
    Convert a DataFrame of flight data to an adjacency list aggregated at the city level.
    """
    graph = defaultdict(set) 

    for _, row in df.iterrows():
        origin, destination = row["Origin_city"], row["Destination_city"]
        if origin != destination:  # Avoid self-loops
            graph[origin].add(destination)
            graph[destination].add(origin)  # Ensure undirected graph

    return {city: list(neighbors) for city, neighbors in graph.items()}

def print_graph(graph, sample_size=10):
    """
    Print a sample of the graph's adjacency list.
    """
    print(f"Adjacency List Sample (First {sample_size} Nodes):")
    print("-" * 50)
    sampled_nodes = list(graph.keys())[:sample_size]
    edges = sum(len(neighbors) for neighbors in graph.values())

    for node in sampled_nodes:
        print(f"{node}: {graph[node]}")
    print("-" * 50)
    print(f"Total nodes in the sample: {len(sampled_nodes)}")
    print(f"Original graph size: {len(graph)} nodes, {edges} edges")

def operation_test(n):
    start = time.time()
    result = 0
    for _ in range(n):
        result += 1  
    end = time.time()
    return n / (end - start)

def df_to_adjacency_matrix(df, weight_col="Passengers"):
    """
    Convert airport DataFrame to an adjacency matrix.
    """
    import pandas as pd
    cities = pd.concat([df["Origin_city"], df["Destination_city"]]).unique()
    city_to_idx = {city: idx for idx, city in enumerate(cities)}
    n = len(cities)
    adj_matrix = np.zeros((n, n))

    for _, row in df.iterrows():
        origin, destination = row["Origin_city"], row["Destination_city"]
        weight = row[weight_col]
        i, j = city_to_idx[origin], city_to_idx[destination]
        adj_matrix[i, j] += weight
        adj_matrix[j, i] += weight  # Ensure symmetry

    return adj_matrix, city_to_idx

def convert_communities_to_readable(communities):
    """
    Converts a nested community structure to a more human-readable format.

    Parameters:
        communities (list of lists): Detected communities as nested lists.

    Returns:
        list of lists: Flattened, grouped, and understandable format.
    """
    readable_communities = []
    for group in communities:
        if isinstance(group[0], list):
            for subgroup in group:
                readable_communities.append(subgroup)
        else:
            readable_communities.append(group)
    return readable_communities


def plot_communities_air(adj_matrix, communities):
    """
    Plot the graph with communities highlighted, ignoring single-node communities.
    Enhance clarity by separating communities and adjusting visual elements.

    Parameters:
        adj_matrix (numpy.ndarray): Adjacency matrix of the graph.
        communities (list of lists): Community structure, each list contains node indices in that community.
    """
    # Filter out single-node communities
    filtered_communities = [community for community in communities if len(community) > 1]

    # Create a graph from the adjacency matrix
    G = nx.from_numpy_array(adj_matrix)

    # Filter nodes in multi-node communities
    nodes_in_multi_node_communities = set(node for community in filtered_communities for node in community)
    G_filtered = G.subgraph(nodes_in_multi_node_communities)

    # Assign colors to nodes based on their community
    community_colors = {node: i for i, community in enumerate(filtered_communities) for node in community}

    # Create node colors array for the filtered graph
    node_colors = [community_colors.get(node, -1) for node in G_filtered.nodes]

    # Increase spacing between nodes
    pos = nx.spring_layout(G_filtered, k=0.3, seed=42)

    # Highlight edges within and between communities
    intra_edges = []
    inter_edges = []
    for u, v in G_filtered.edges:
        if community_colors[u] == community_colors[v]:
            intra_edges.append((u, v))
        else:
            inter_edges.append((u, v))

    # Plot the graph
    plt.figure(figsize=(14, 10))
    nx.draw_networkx_edges(G_filtered, pos, edgelist=intra_edges, alpha=0.8, edge_color="gray")
    nx.draw_networkx_edges(G_filtered, pos, edgelist=inter_edges, alpha=0.2, edge_color="black")
    nx.draw_networkx_nodes(
        G_filtered,
        pos,
        node_color=node_colors,
        cmap=plt.cm.tab10,
        node_size=300,
    )
    nx.draw_networkx_labels(G_filtered, pos, font_size=8, font_color="black", font_weight="bold")

    # Add a title
    plt.title("Enhanced Community Structure Visualization", fontsize=16)
    plt.axis("off")
    plt.show()

import folium
from folium.plugins import MarkerCluster
import random

def plot_communities_with_colored_connections(df, adj_matrix, readable_communities, city_to_index):
    """
    Plot a lightweight interactive map with communities, colored connections, and a clear legend.
    """
    # Reverse mapping: index -> city
    index_to_city = {v: k for k, v in city_to_index.items()}

    # Prepare city-community mapping
    community_data = []
    for i, community in enumerate(readable_communities):
        for city_index in community:
            city_name = index_to_city.get(city_index)
            if city_name:
                community_data.append({'City': city_name, 'Community': i})

    community_df = pd.DataFrame(community_data)

    # Extract unique cities with their coordinates
    city_coords = df[['Origin_city', 'Org_airport_lat', 'Org_airport_long']].drop_duplicates()
    city_coords.rename(columns={'Origin_city': 'City', 'Org_airport_lat': 'Latitude', 'Org_airport_long': 'Longitude'}, inplace=True)

    # Merge city coordinates with communities
    city_df = city_coords.merge(community_df, on='City', how='inner')

    # Create a Folium map
    folium_map = folium.Map(location=[37.0902, -95.7129], zoom_start=4)  # Centered on the USA
    marker_cluster = MarkerCluster().add_to(folium_map)

    # Assign random colors to multi-node communities
    multi_node_communities = {i: c for i, c in enumerate(readable_communities) if len(c) > 1}
    community_colors = {
        i: f"#{''.join(random.choices('89ABCDEF', k=6))}" for i in multi_node_communities
    }

    # Use gray for single-node communities
    for i, community in enumerate(readable_communities):
        if len(community) == 1:
            community_colors[i] = "gray"

    # Add city markers
    for _, row in city_df.iterrows():
        popup_info = f"City: {row['City']}<br>Community: {row['Community']}"
        folium.CircleMarker(
            location=[row['Latitude'], row['Longitude']],
            radius=5,
            color=community_colors[row['Community']],
            fill=True,
            fill_opacity=0.8,
            popup=popup_info
        ).add_to(marker_cluster)

    # Add colored connections
    for i, row1 in city_df.iterrows():
        for j, row2 in city_df.iterrows():
            if adj_matrix[city_to_index[row1['City']], city_to_index[row2['City']]] > 0:
                community_color = community_colors[row1['Community']]
                folium.PolyLine(
                    locations=[
                        [row1['Latitude'], row1['Longitude']],
                        [row2['Latitude'], row2['Longitude']],
                    ],
                    color=community_color,
                    weight=0.8,
                    opacity=0.3,  # Adjust transparency
                ).add_to(folium_map)

    # Add a legend excluding single-node communities
    legend_html = """
    <div style="position: fixed; 
                bottom: 50px; left: 50px; width: 200px; height: auto; 
                background-color: white; z-index:9999; font-size:14px;
                border:2px solid grey; padding: 10px;">
    <strong>Community Colors</strong><br>
    <em>(Gray: Single-node communities)</em><br>
    """
    for i, color in community_colors.items():
        if len(readable_communities[i]) > 1:  # Exclude single-node communities
            legend_html += f"<div style='display:inline-block; width:15px; height:15px; background-color:{color}; margin-right:5px;'></div> Community {i}<br>"
    legend_html += "</div>"

    folium_map.get_root().html.add_child(folium.Element(legend_html))

    return folium_map

def analyze_flight_network(adj_matrix, city_labels, city1, city2):
    """
    Analyze the flight network, identify communities, and answer key questions.
    """
    communities = louvain_cluster(adj_matrix) # Perform Louvain clustering
    num_communities = len(communities)

    # Map communities to city labels
    labeled_communities = [
        [city_labels[node] for node in community]
        for community in communities
    ]
    # Determine if city1 and city2 are in the same community
    city1_idx = city_labels.index(city1)
    city2_idx = city_labels.index(city2)
    same_community = any(
        city1_idx in community and city2_idx in community for community in communities
    )
    print(f"\nCity {city1} and City {city2} belong to the same community: {same_community}\n")

# LLM Functions

# Visualize the communities
def visualize_communities(graph, communities, title):
    # Assign colors based on communities
    node_colors = {}
    for i, community in enumerate(communities):
        for node in community:
            node_colors[node] = i
    colors = [node_colors[node] for node in graph.nodes()]
    
    # Layout and drawing
    pos = nx.spring_layout(graph)
    nx.draw(graph, pos, node_color=colors, with_labels=True, cmap=plt.cm.rainbow)
    plt.title(title)
    plt.show()

def analyze_communities(graph, communities):
    print("=== Community Analysis ===")
    
    # Number of communities
    num_communities = len(communities)
    print(f"Total Communities Detected: {num_communities}\n")
    
    # Display nodes in each community
    for i, community in enumerate(communities, start=1):
        print(f"Community {i}:")
        print(f"  Nodes: {sorted(community)}")
        print(f"  Size: {len(community)}\n")
    
    # Calculate modularity
    modularity = nx.algorithms.community.quality.modularity(
        graph,
        communities
    )
    print(f"Modularity Score: {modularity:.4f}")

# part 3

def prepare_flight_graph(df, flight_date):
    """
    Prepare a weighted graph filtered by a specific flight date.
+    """
    filtered_df = df[df['Fly_date'] == flight_date]
    
    if filtered_df.empty:
        raise ValueError("[ERROR] No flights available on the given date.")
    
    graph = defaultdict(set)  # Graph adjacency list (node: set of neighbors)
    weights = {}  # Edge weights (tuple(node1, node2): weight)
    
    for _, row in filtered_df.iterrows():
        # populate graph
        origin = row['Origin_airport']
        dest = row['Destination_airport']
        # and weights
        distance = row['Distance']
        
        graph[origin].add(dest)  # Add destination as a neighbor of origin
        weights[(origin, dest)] = distance  # Add distance as weight for the edge
    
    return graph, weights

def get_city_airports(df, city, column):
    """
    Get a list of airports for a specific city.    
    """
    # Filter dataset by city and extract unique airport codes
    return df[df[column] == city][column.replace('city', 'airport')].unique().tolist()

def reconstruct_path(previous_nodes, start, end):
    """
    Reconstructs the shortest path from start to end using previous_nodes.
    """
    print(f"\n🔄 [DEBUG] Reconstructing path from '{start}' to '{end}'")
    path = []
    current_node = end
    
    # Backtrack from end to start using the 'previous_nodes' map
    while current_node is not None:
        path.insert(0, current_node)
        current_node = previous_nodes.get(current_node)
    
    # Verify if a valid path exists
    if not path or path[0] != start:
        return "No route found"
    
    # Return if it does
    formatted_path = " → ".join(path)
    print(f"✅ [DEBUG] Path Found: {formatted_path}")
    return formatted_path

# I am just gonna explain the changes made from the previous version
def dijkstra_adj_list_weighted(graph, weights, start_node):
    """
    Dijkstra's Algorithm for weighted graphs.
    """
    all_nodes = set(graph.keys()) # created to include nodes from both graph and weights
    for (u, v) in weights.keys():
        all_nodes.update([u, v]) # Ensures nodes referenced in weights but missing from graph are also included in the algorithm
    # This avoids the issue where certain nodes are skipped because they were not part of graph.keys().
    # Add missing nodes to the adjacency list
    for node in all_nodes:
        if node not in graph:
            graph[node] = set()
    
    # Validate that each edge in weights exists in the adjacency list
    for (u, v), weight in weights.items():
        if v not in graph[u]:
            graph[u].add(v)
        if u not in graph[v]:
            graph[v].add(u)

    distances = {node: float('inf') for node in all_nodes}
    previous_nodes = {node: None for node in all_nodes}

    distances[start_node] = 0
    priority_queue = [(0, start_node)]

    while priority_queue:
        current_distance, current_node = heapq.heappop(priority_queue)

        if current_distance > distances[current_node]:
            continue

        for neighbor in graph.get(current_node, []):
            if neighbor not in distances:
                print(f"[WARNING] Neighbor node '{neighbor}' not found in distances. Skipping.")
                continue

            weight = weights.get((current_node, neighbor), float('inf')) # we actyally get the weight from the weights dictionary, instead of using a fixed value
            distance = current_distance + weight

            if distance < distances[neighbor]:
                distances[neighbor] = distance
                previous_nodes[neighbor] = current_node
                heapq.heappush(priority_queue, (distance, neighbor))

    return distances, previous_nodes

def compute_best_routes_between_cities(df, origin_city, destination_city, flight_date):
    """
    Compute the best routes between airports in two cities on a given date.
    """
    
    filtered_df = df[df['Fly_date'] == flight_date]
    print(f"[DEBUG] Filtered dataset contains {len(filtered_df)} flights on {flight_date}.")
    
    origin_airports = get_city_airports(filtered_df, origin_city, 'Origin_city')
    destination_airports = get_city_airports(filtered_df, destination_city, 'Destination_city')
    
    if not origin_airports or not destination_airports:
        return print("[ERROR] No flights available for the given cities on the specified date.")
        
    airport_pairs = list(itertools.product(origin_airports, destination_airports))
    print(f"[DEBUG] Generated {len(airport_pairs)} airport pairs to evaluate.")
    
    graph, weights = prepare_flight_graph(filtered_df, flight_date)
    
    results = []
    for origin, destination in airport_pairs:
        print(f"\n🔄 [DEBUG] Processing pair: {origin} → {destination}")
        
        if origin not in graph or destination not in graph: 
            print(f"[WARNING] Origin '{origin}' or destination '{destination}' is not in the graph.")
            results.append({
                "Origin_city_airport": origin,
                "Destination_city_airport": destination,
                "Best_route": "No route found",
                "Total_distance": "N/A"
            })
            continue
        
        distances, previous_nodes = dijkstra_adj_list_weighted(graph, weights, origin) # apply Dijkstra's algorithm
        path = reconstruct_path(previous_nodes, origin, destination)
        
        if path == "No route found":
            results.append({
                "Origin_city_airport": origin,
                "Destination_city_airport": destination,
                "Best_route": "No route found",
                "Total_distance": "N/A"
            })
        else:
            total_distance = distances.get(destination, float('inf'))
            results.append({
                "Origin_city_airport": origin,
                "Destination_city_airport": destination,
                "Best_route": path,
                "Total_distance": total_distance
            })
    
    return pd.DataFrame(results)

def plot_best_route_on_map(df, best_route_df):
    """
    Plot only the best route on a Folium map.
    """
    # Extract the best route information
    origin_airport = best_route_df.iloc[0]['Origin_city_airport']
    destination_airport = best_route_df.iloc[0]['Destination_city_airport']
    best_route = best_route_df.iloc[0]['Best_route'].split(" → ")

    # Create a base map centered on the first airport
    start_coords = df.loc[df['Origin_airport'] == origin_airport, ['Org_airport_lat', 'Org_airport_long']].iloc[0]
    m = folium.Map(location=[start_coords['Org_airport_lat'], start_coords['Org_airport_long']], zoom_start=5)
    
    # Plot each airport in the best route
    for i in range(len(best_route)):
        airport = best_route[i]
        
        if i == 0:
            # Starting airport
            coords = df.loc[df['Origin_airport'] == airport, ['Org_airport_lat', 'Org_airport_long']].iloc[0]
            folium.Marker(
                location=[coords['Org_airport_lat'], coords['Org_airport_long']],
                popup=f"Start: {airport}",
                icon=folium.Icon(color='green', icon='plane')
            ).add_to(m)
        elif i == len(best_route) - 1:
            # Destination airport
            coords = df.loc[df['Destination_airport'] == airport, ['Dest_airport_lat', 'Dest_airport_long']].iloc[0]
            folium.Marker(
                location=[coords['Dest_airport_lat'], coords['Dest_airport_long']],
                popup=f"End: {airport}",
                icon=folium.Icon(color='red', icon='flag')
            ).add_to(m)
        else:
            # Intermediate airports
            coords = df.loc[df['Origin_airport'] == airport, ['Org_airport_lat', 'Org_airport_long']].iloc[0]
            folium.Marker(
                location=[coords['Org_airport_lat'], coords['Org_airport_long']],
                popup=airport,
                icon=folium.Icon(color='blue', icon='circle')
            ).add_to(m)

    # Draw the route
    route_coords = []
    for i in range(len(best_route) - 1):
        start_airport = best_route[i]
        end_airport = best_route[i + 1]
        
        start_coords = df.loc[df['Origin_airport'] == start_airport, ['Org_airport_lat', 'Org_airport_long']].iloc[0]
        end_coords = df.loc[df['Destination_airport'] == end_airport, ['Dest_airport_lat', 'Dest_airport_long']].iloc[0]
        
        route_coords.append(([start_coords['Org_airport_lat'], start_coords['Org_airport_long']],
                             [end_coords['Dest_airport_lat'], end_coords['Dest_airport_long']]))
    
    for start, end in route_coords:
        folium.PolyLine([start, end], color='red', weight=4, opacity=0.7).add_to(m)
    
    return m
