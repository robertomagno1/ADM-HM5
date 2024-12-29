"""
This script provides functions to calculate various centrality measures in a flight network.
Centrality measures include:
    - Degree Centrality: Reflects the number of connections (in and out) a node has.
    - Closeness Centrality: Measures how easily a node can reach all other nodes.
    - Betweenness Centrality: Quantifies the number of shortest paths passing through a node.
    - PageRank: Evaluates the relative importance of a node based on incoming links.

Each function operates on a directed graph, where nodes represent airports and edges
represent flights between them. These measures help analyze network structures and
the importance of specific nodes.
"""


from collections import defaultdict, deque
import matplotlib.pyplot as plt

# =====================================================
# Centrality Computation Functions
# =====================================================

def calculate_degree_centrality(flight_network, airport):
    """
    Compute the degree centrality of a given airport in the flight network.
    
    Degree centrality measures how connected a node is by calculating the sum of its 
    in-degree and out-degree, normalized by the total number of other nodes in the graph.

    Parameters:
        flight_network: A directed graph representing the flight network.
        airport: The node representing the airport for which centrality is calculated.

    Returns:
        A float representing the normalized degree centrality.
    """
    out_degree = flight_network.out_degree(airport)  # Outgoing connections
    in_degree = flight_network.in_degree(airport)  # Incoming connections
    total_nodes = flight_network.number_of_nodes() - 1  # Exclude the node itself
    return (out_degree + in_degree) / (2 * total_nodes) if total_nodes > 0 else 0.0

def calculate_closeness_centrality(flight_network, airport):
    """
    Compute the closeness centrality for a given airport in the flight network.
    
    Closeness centrality measures how easily a node can reach all other nodes in the graph. 
    It is the reciprocal of the sum of shortest path distances from the node to all others.

    Parameters:
        flight_network: A directed graph representing the flight network.
        airport: The node representing the airport for which centrality is calculated.

    Returns:
        A float representing the closeness centrality.
    """
    if airport not in flight_network:
        return 0.0

    distances = bfs_shortest_paths(flight_network, airport)  # Get shortest path distances
    reachable_nodes = [dist for dist in distances.values() if dist < float('inf')]

    if len(reachable_nodes) <= 1:
        return 0.0  # Closeness is undefined for isolated nodes

    reachable_sum = sum(reachable_nodes)
    return (len(reachable_nodes) - 1) / reachable_sum

def bfs_shortest_paths(graph, start_node):
    """
    Perform a Breadth-First Search (BFS) to calculate the shortest paths 
    from a start node to all other nodes in the graph.

    Parameters:
        graph: A directed graph.
        start_node: The starting node for the BFS traversal.

    Returns:
        A dictionary mapping each node to its shortest path distance from the start node.
    """
    distances = {node: float('inf') for node in graph.nodes()}
    distances[start_node] = 0
    queue = deque([start_node])

    while queue:
        current = queue.popleft()
        for neighbor in graph.successors(current):  # Visit all outgoing edges
            if distances[neighbor] == float('inf'):  # First visit
                distances[neighbor] = distances[current] + 1
                queue.append(neighbor)
    return distances

def calculate_betweenness_centrality(flight_network, airport):
    """
    Compute the betweenness centrality of a given airport.
    
    Betweenness centrality measures the extent to which a node lies on the shortest paths
    between pairs of other nodes in the graph.

    Parameters:
        flight_network: A directed graph representing the flight network.
        airport: The node representing the airport for which centrality is calculated.

    Returns:
        A float representing the betweenness centrality.
    """
    total_paths = 0
    passing_paths = 0

    for src in flight_network.nodes():
        if src == airport:
            continue

        paths, parents = calculate_shortest_path_dependencies(flight_network, src)

        for dest in flight_network.nodes():
            if dest == airport or dest == src or paths[dest] == 0:
                continue

            path_count = count_paths_through_node(dest, airport, parents)
            passing_paths += path_count  # Increment paths passing through the airport
            total_paths += paths[dest]

    n = flight_network.number_of_nodes()
    return passing_paths / ((n - 1) * (n - 2)) if total_paths > 0 else 0.0


def calculate_shortest_path_dependencies(flight_network, source):
    """
    Compute the shortest path counts and parent relationships for a source node.
    
    This function identifies all shortest paths from the source to other nodes and tracks 
    the parent nodes contributing to those paths.

    Parameters:
        flight_network: A directed graph representing the flight network.
        source: The node from which shortest paths are calculated.

    Returns:
        paths: A dictionary with node-wise shortest path counts.
        parents: A dictionary mapping each node to its parent nodes in the shortest paths.
    """
    paths = defaultdict(int)
    parents = defaultdict(list)
    distances = {node: float('inf') for node in flight_network.nodes()}

    paths[source] = 1
    distances[source] = 0
    queue = deque([source])

    while queue:
        current = queue.popleft()
        for neighbor in flight_network.successors(current):
            distance = distances[current] + 1
            if distance < distances[neighbor]:  # Found a shorter path
                distances[neighbor] = distance
                paths[neighbor] = paths[current]
                parents[neighbor] = [current]
                queue.append(neighbor)
            elif distance == distances[neighbor]:  # Found an equally short path
                paths[neighbor] += paths[current]
                parents[neighbor].append(current)

    return paths, parents


def count_paths_through_node(dest, node, parents):
    """
    Count the number of shortest paths passing through a specific node.

    Parameters:
        dest: The destination node of the paths.
        node: The node to check for path inclusion.
        parents: A dictionary mapping each node to its parent nodes in the shortest paths.

    Returns:
        The number of paths passing through the specified node.
    """
    stack = deque([dest])
    path_count = 0

    while stack:
        current = stack.pop()
        if current == node:  # Node lies on the path
            path_count += 1
        else:
            stack.extend(parents[current])  # Explore parent nodes

    return path_count



def calculate_page_rank(flight_network, airport, damping_factor=0.85, max_iter=100, tolerance=1e-6):
    """
    Compute the PageRank of an airport in the flight network.
    
    PageRank measures the importance of a node based on the number and quality of links 
    directed to it, using a random walk model.

    Parameters:
        flight_network: A directed graph representing the flight network.
        airport: The node representing the airport for which PageRank is calculated.
        damping_factor: The probability of continuing the random walk at each step.
        max_iter: Maximum number of iterations for convergence.
        tolerance: Convergence threshold for rank changes.

    Returns:
        A float representing the PageRank of the specified airport.
    """
    N = flight_network.number_of_nodes()
    ranks = {node: 1 / N for node in flight_network.nodes()}  # Initialize ranks
    sink_nodes = {node for node in flight_network.nodes() if flight_network.out_degree(node) == 0}

    for _ in range(max_iter):
        previous_ranks = ranks.copy()
        sink_rank = damping_factor * sum(previous_ranks[node] for node in sink_nodes) / N
        for node in flight_network.nodes():
            rank_sum = sum(
                previous_ranks[neighbor] / flight_network.out_degree(neighbor)
                for neighbor in flight_network.predecessors(node)  # Incoming edges
            )
            ranks[node] = (1 - damping_factor) / N + damping_factor * (rank_sum + sink_rank)

        # Check for convergence
        if max(abs(ranks[node] - previous_ranks[node]) for node in flight_network.nodes()) < tolerance:
            break

    return ranks.get(airport, 0)
    

# =====================================================
# Analyze Centrality for Single Airport
# =====================================================
def analyze_centrality(flight_network, airport):
    """
    Analyze various centrality measures for a given airport.
    
    Combines degree centrality, closeness centrality, betweenness centrality, and PageRank 
    into a single dictionary for comparison.

    Parameters:
        flight_network: A directed graph representing the flight network.
        airport: The node representing the airport to analyze.

    Returns:
        A dictionary containing the centrality measures for the specified airport.
    """
    return {
        "Airport": airport,
        "Degree Centrality": calculate_degree_centrality(flight_network, airport),
        "Closeness Centrality": calculate_closeness_centrality(flight_network, airport),
        "Betweenness Centrality": calculate_betweenness_centrality(flight_network, airport),
        "PageRank": calculate_page_rank(flight_network, airport),
    }


# =====================================================
# Compare Centralities Across Airports
# =====================================================
def compare_centralities(flight_network):
    import matplotlib.pyplot as plt
    centralities = []

    for airport in tqdm(flight_network.nodes(), desc="Calculating Centralities"):
        centrality = analyze_centrality(flight_network, airport)
        centralities.append(centrality)

    results_df = pd.DataFrame(centralities)

    # Plot distributions
    for col in ["Degree Centrality", "Closeness Centrality", "Betweenness Centrality", "PageRank"]:
        plt.figure(figsize=(10, 6))
        plt.hist(results_df[col], bins=20, color='skyblue', edgecolor='black')
        plt.title(f"{col} Distribution")
        plt.xlabel(col)
        plt.ylabel("Frequency")
        plt.show()

    # Get top 5 airports for each centrality measure
    results = {
        "top_degree": results_df.nlargest(5, "Degree Centrality")[["Airport", "Degree Centrality"]].values.tolist(),
        "top_closeness": results_df.nlargest(5, "Closeness Centrality")[["Airport", "Closeness Centrality"]].values.tolist(),
        "top_betweenness": results_df.nlargest(5, "Betweenness Centrality")[["Airport", "Betweenness Centrality"]].values.tolist(),
        "top_pagerank": results_df.nlargest(5, "PageRank")[["Airport", "PageRank"]].values.tolist(),
    }

    return results


def build_directed_graph(df):
    """
    Build a directed graph from a DataFrame that has at least these columns:
      - 'Origin_airport'
      - 'Destination_airport'.
    We ignore rows where origin == destination (no self-loops).
    
    The result is a Python dictionary with the structure:
      {
        "AIRPORT_1": [list_of_destination_airports],
        "AIRPORT_2": [...],
        ...
      }
    For any 'destination' not already in the dictionary, we add a key pointing to an empty list
    to ensure every airport appears (even if it has no outgoing edges).
    """
    graph = defaultdict(list)
    
    for _, row in df.iterrows():
        origin = row['Origin_airport']
        dest = row['Destination_airport']
        
        # Skip self-loops (no route from an airport to itself)
        if origin != dest:
            graph[origin].append(dest)
        
        # Make sure the destination exists as a key, so it won't be missing
        if dest not in graph:
            graph[dest] = []
    
    # Convert defaultdict(list) into a regular dictionary for clarity
    return dict(graph)


def compute_degree_centrality(graph, airport):
    """
    Compute the degree centrality for a single 'airport'.
    
    Degree Centrality(airport) = in_degree(airport) + out_degree(airport)
    
    - out_degree(airport) is simply the length of graph[airport].
    - in_degree(airport) is the number of adjacency lists that include 'airport'.
    """
    if airport not in graph:
        return 0
    
    # Out-degree: how many destinations are in the airport's adjacency list
    out_deg = len(graph[airport])
    
    # In-degree: how many adjacency lists (i.e., other airports) contain this airport
    in_deg = 0
    for node, neighbors in graph.items():
        in_deg += neighbors.count(airport)
    
    return in_deg + out_deg


def bfs_distances_unweighted(graph, start):
    """
    Perform a BFS (Breadth-First Search) starting from 'start' in an unweighted directed graph.
    Returns a dictionary dist such that dist[node] = number of edges from 'start' to 'node',
    or float('inf') if 'node' is unreachable.
    """
    dist = {n: float('inf') for n in graph}
    dist[start] = 0
    
    queue = deque([start])
    while queue:
        current = queue.popleft()
        for neighbor in graph[current]:
            if dist[neighbor] == float('inf'):
                dist[neighbor] = dist[current] + 1
                queue.append(neighbor)
    return dist


def compute_closeness_centrality(graph, airport):
    """
    Closeness Centrality(airport) = (# reachable nodes) / (sum of distances to reachable nodes).
    
    - We first do a BFS to get the distance from 'airport' to every other node.
    - We ignore any node that is unreachable (distance = inf).
    - If no nodes are reachable, closeness is 0.
    """
    if airport not in graph:
        return 0.0
    
    dist_dict = bfs_distances_unweighted(graph, airport)
    
    # Filter out unreachable nodes and the airport itself
    valid_distances = [d for node, d in dist_dict.items()
                       if d < float('inf') and node != airport]
    
    if not valid_distances:
        return 0.0
    
    return len(valid_distances) / sum(valid_distances)


def brandes_betweenness_all_nodes(graph):
    """
    Use Brandes' algorithm to compute betweenness centrality for every node in a directed,
    unweighted graph. Returns a dict {node: betweenness_value}.
    
    The Betweenness Centrality of a node v is roughly how many shortest paths between other pairs
    (s, t) pass through v. Brandes' method does BFS from each node and back-propagates dependencies.
    """
    from collections import deque
    
    # Initialize betweenness (Cb) of each node to 0
    Cb = {v: 0.0 for v in graph}
    nodes = list(graph.keys())

    for s in nodes:
        # Lists and dictionaries needed by Brandes
        S = []  # stack
        P = {w: [] for w in graph}     # predecessors
        dist = {w: -1 for w in graph}  # distance from s
        sigma = {w: 0 for w in graph}  # number of shortest paths from s to w
        
        # Initialize BFS from s
        dist[s] = 0
        sigma[s] = 1
        queue = deque([s])
        
        # BFS to find shortest paths
        while queue:
            v = queue.popleft()
            S.append(v)
            for w in graph[v]:
                if dist[w] < 0:        # first time we see w
                    dist[w] = dist[v] + 1
                    queue.append(w)
                if dist[w] == dist[v] + 1:
                    sigma[w] += sigma[v]
                    P[w].append(v)
        
        # Accumulate dependencies
        delta = {w: 0.0 for w in graph}
        while S:
            w = S.pop()
            for v in P[w]:
                delta[v] += (sigma[v] / sigma[w]) * (1 + delta[w])
            if w != s:
                Cb[w] += delta[w]
    
    return Cb


def compute_betweenness_for_node(graph, airport):
    """
    Compute Brandes betweenness centrality for all nodes, then return it
    for the single specified 'airport'.
    """
    bc_all = brandes_betweenness_all_nodes(graph)
    return bc_all.get(airport, 0.0)


def compute_pagerank_all(graph, alpha=0.85, max_iter=100, tol=1e-6):
    """
    Calculate PageRank for all nodes in a directed, unweighted graph.
    - alpha: damping factor (0.85 is typical).
    - max_iter: max number of power-iteration steps.
    - tol: minimum difference threshold for convergence.
    Returns a dict {node: pagerank_value}.
    """
    nodes = list(graph.keys())
    N = len(nodes)
    
    # Initialize each node's rank to 1/N
    rank = {n: 1.0 / N for n in nodes}
    
    # Precompute out-degree
    out_degree = {n: len(graph[n]) for n in nodes}

    for _ in range(max_iter):
        new_rank = {n: (1 - alpha)/N for n in nodes}

        # sum of ranks of sink nodes (out_degree=0)
        sink_sum = 0.0
        for n in nodes:
            if out_degree[n] == 0:
                sink_sum += rank[n]
        sink_contrib = alpha * sink_sum / N

        # Distribute rank from each node
        for u in nodes:
            if out_degree[u] > 0:
                contribution = alpha * (rank[u] / out_degree[u])
                for v in graph[u]:
                    new_rank[v] += contribution
        
        # Add sink contribution to everyone
        for v in nodes:
            new_rank[v] += sink_contrib
        
        # Check convergence
        diff = sum(abs(new_rank[n] - rank[n]) for n in nodes)
        rank = new_rank
        if diff < tol:
            break

    return rank


def compute_pagerank_for_node(graph, airport):
    """
    Compute PageRank for all nodes, then return the value for the given 'airport'.
    """
    pr_all = compute_pagerank_all(graph)
    return pr_all.get(airport, 0.0)


def analyze_centrality(df, airport):
    """
    Build the directed graph from 'df' using 'build_directed_graph'.
    Then compute the following for a single 'airport':
       - Degree Centrality
       - Closeness Centrality
       - Betweenness Centrality
       - PageRank
    Returns a dict with these 4 measures for 'airport'.
    """
    # 1) Build the graph
    graph = build_directed_graph(df)

    # 2) Compute the metrics for the single airport
    deg = compute_degree_centrality(graph, airport)
    clo = compute_closeness_centrality(graph, airport)
    bet = compute_betweenness_for_node(graph, airport)
    pr  = compute_pagerank_for_node(graph, airport)

    return {
        "airport": airport,
        "degree_centrality": deg,
        "closeness_centrality": clo,
        "betweenness_centrality": bet,
        "pagerank": pr
    }


def compare_centralities(df):
    """
    1) Build the directed flight graph from 'df'.
    2) Compute:
       - Degree (in+out)
       - Closeness (BFS)
       - Betweenness (Brandes)
       - PageRank (iterative)
       for ALL nodes in the graph.
    3) Plot each distribution as a histogram.
    4) Return the top-5 airports for each metric in a dictionary.
    """
    graph = build_directed_graph(df)
    nodes = list(graph.keys())

    # ---- Degree
    in_deg = {n: 0 for n in nodes}
    for n in nodes:
        for neigh in graph[n]:
            in_deg[neigh] += 1

    degree_dict = {}
    for n in nodes:
        out_deg = len(graph[n])
        degree_dict[n] = out_deg + in_deg[n]

    # ---- Closeness
    closeness_dict = {}
    for n in nodes:
        dist_map = bfs_distances_unweighted(graph, n)
        valid = [d for x, d in dist_map.items() if d < float('inf') and x != n]
        closeness_dict[n] = (len(valid) / sum(valid)) if valid else 0.0

    # ---- Betweenness
    betweenness_dict = brandes_betweenness_all_nodes(graph)

    # ---- PageRank
    pagerank_dict = compute_pagerank_all(graph)

    # Prepare data for plotting
    deg_vals = [degree_dict[n] for n in nodes]
    clos_vals = [closeness_dict[n] for n in nodes]
    bet_vals = [betweenness_dict[n] for n in nodes]
    pr_vals  = [pagerank_dict[n] for n in nodes]

    # Plot histograms
    plt.figure(figsize=(12, 8))

    # Degree Distribution
    plt.subplot(2,2,1)
    plt.hist(deg_vals, bins=30, color='skyblue', edgecolor='black')
    plt.title("Degree Centrality Distribution")

    # Closeness Distribution
    plt.subplot(2,2,2)
    plt.hist(clos_vals, bins=30, color='orange', edgecolor='black')
    plt.title("Closeness Centrality Distribution")

    # Betweenness Distribution
    plt.subplot(2,2,3)
    plt.hist(bet_vals, bins=30, color='green', edgecolor='black')
    plt.title("Betweenness Centrality Distribution")

    # PageRank Distribution
    plt.subplot(2,2,4)
    plt.hist(pr_vals, bins=30, color='purple', edgecolor='black')
    plt.title("PageRank Distribution")

    plt.tight_layout()
    plt.show()

    # Return the top-5 for each metric
    def top5(d):
        return sorted(d.items(), key=lambda x: x[1], reverse=True)[:5]

    return {
        "top_degree":      top5(degree_dict),
        "top_closeness":   top5(closeness_dict),
        "top_betweenness": top5(betweenness_dict),
        "top_pagerank":    top5(pagerank_dict)
    }


## FUNCTION TO CHECK EVENTUALY ANOMALIES ••...............................................
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict, deque

###############################################################################
# 1) Basic Data Checks
###############################################################################

def check_flight_data(df):
    """
    Perform basic checks on the flight DataFrame:
      1) Check for fully duplicated rows.
      2) Identify any rows where Origin_airport == Destination_airport (self-loops).
      3) Check for missing (NaN) values in 'Origin_airport' or 'Destination_airport'.
      4) (Optional) Flag suspicious IATA codes based on their length (<3 or >4 characters).

    Returns:
        list of strings describing any problems found.
    """
    problems = []

    # (1) Duplicated rows:
    duplicate_rows = df[df.duplicated()]
    if not duplicate_rows.empty:
        problems.append(f"Found {len(duplicate_rows)} fully duplicated rows in the DataFrame.")

    # (2) Self-loops: rows where Origin_airport == Destination_airport
    same_route = df[df['Origin_airport'] == df['Destination_airport']]
    if not same_route.empty:
        problems.append(f"Found {len(same_route)} rows where Origin == Destination (self-loops).")

    # (3) NaN in key columns
    missing_origin = df['Origin_airport'].isna().sum()
    missing_destination = df['Destination_airport'].isna().sum()
    if missing_origin > 0:
        problems.append(f"There are {missing_origin} NaN values in 'Origin_airport'.")
    if missing_destination > 0:
        problems.append(f"There are {missing_destination} NaN values in 'Destination_airport'.")

    # (4) (Optional) Check IATA code length (<3 or >4 might be suspicious)
    iata_suspects = df[
        (df['Origin_airport'].str.len() < 3) |
        (df['Origin_airport'].str.len() > 4) |
        (df['Destination_airport'].str.len() < 3) |
        (df['Destination_airport'].str.len() > 4)
    ]
    if not iata_suspects.empty:
        problems.append(f"Found {len(iata_suspects)} rows with IATA codes of abnormal length (<3 or >4).")

    return problems


###############################################################################
# 2) Building a Directed Graph from the Flight Data
###############################################################################

def build_directed_graph(df):
    """
    Build a directed graph from a DataFrame with at least the columns:
      - 'Origin_airport'
      - 'Destination_airport'.
    - Skip any row where Origin == Destination (avoid self-loops).
    - The result is a dictionary: {origin_airport: [list_of_destinations], ...}.

    If a 'destination' is never an origin, we still add an empty list
    to ensure every airport appears in the graph.
    """
    graph = defaultdict(list)
    for _, row in df.iterrows():
        origin = row['Origin_airport']
        dest = row['Destination_airport']
        if origin != dest:  # skip self-loops
            graph[origin].append(dest)
        # Make sure each destination also exists as a key
        if dest not in graph:
            graph[dest] = []
    return dict(graph)


###############################################################################
# 3) Unweighted Centralities (Degree, Closeness, Betweenness, PageRank)
###############################################################################

# (A) Degree (in + out)

def compute_all_degree_centralities(graph):
    """
    Return a dict {node: in_degree + out_degree} for each node in 'graph'.
    out_degree(node) = len(graph[node])
    in_degree(node)  = number of adjacency lists that mention 'node'.
    """
    # Initialize in-degree
    in_degree = {n: 0 for n in graph}
    # Count in-degree by scanning adjacency
    for n, neighbors in graph.items():
        for dst in neighbors:
            in_degree[dst] += 1

    # Combine: degree = in_degree + out_degree
    degree_dict = {}
    for n in graph:
        out_deg = len(graph[n])
        degree_dict[n] = out_deg + in_degree[n]
    return degree_dict


# (B) Closeness: BFS-based

def bfs_unweighted_distances(graph, start):
    """
    BFS to get a dictionary of distances (in edges) from 'start' to each reachable node.
    If a node is unreachable, distance[node] stays float('inf').
    """
    dist = {n: float('inf') for n in graph}
    dist[start] = 0
    queue = deque([start])
    while queue:
        current = queue.popleft()
        for neighbor in graph[current]:
            if dist[neighbor] == float('inf'):
                dist[neighbor] = dist[current] + 1
                queue.append(neighbor)
    return dist

def compute_all_closeness_centralities(graph):
    """
    Return a dict {node: closeness} for each node, where
      closeness(node) = (#reachable_nodes) / (sum_of_distances).
    """
    closeness_dict = {}
    for node in graph:
        dist_map = bfs_unweighted_distances(graph, node)
        valid = [d for x,d in dist_map.items() if d < float('inf') and x != node]
        if valid:
            closeness_dict[node] = len(valid)/sum(valid)
        else:
            closeness_dict[node] = 0.0
    return closeness_dict


# (C) Betweenness: Brandes for all nodes

def brandes_betweenness_centrality(graph):
    """
    Compute betweenness for every node in an unweighted directed graph.
    Returns {node: betweenness_value}.
    """
    from collections import deque
    Cb = {v: 0.0 for v in graph}
    nodes = list(graph.keys())

    for s in nodes:
        S = []
        P = {w: [] for w in graph}
        dist = {w: -1 for w in graph}
        sigma = {w: 0 for w in graph}

        dist[s] = 0
        sigma[s] = 1
        queue = deque([s])

        # BFS from s
        while queue:
            v = queue.popleft()
            S.append(v)
            for w in graph[v]:
                if dist[w] < 0:
                    dist[w] = dist[v] + 1
                    queue.append(w)
                if dist[w] == dist[v] + 1:
                    sigma[w] += sigma[v]
                    P[w].append(v)

        # Accumulate dependencies
        delta = {w: 0.0 for w in graph}
        while S:
            w = S.pop()
            for v in P[w]:
                delta[v] += (sigma[v]/sigma[w]) * (1 + delta[w])
            if w != s:
                Cb[w] += delta[w]

    return Cb


# (D) PageRank: iterative approach

def compute_pagerank(graph, alpha=0.85, max_iter=100, tol=1e-6):
    """
    Compute PageRank for an unweighted directed graph. Returns {node: rank_value}.
    alpha is the damping factor (0.85 is typical).
    """
    nodes = list(graph.keys())
    N = len(nodes)
    rank = {n: 1.0/N for n in nodes}
    out_degree = {n: len(graph[n]) for n in nodes}

    for _ in range(max_iter):
        new_rank = {n: (1 - alpha)/N for n in nodes}

        # sum of ranks of sink nodes
        sink_sum = 0.0
        for n in nodes:
            if out_degree[n] == 0:
                sink_sum += rank[n]
        sink_contrib = alpha * sink_sum / N

        for u in nodes:
            if out_degree[u] > 0:
                share = alpha * (rank[u]/out_degree[u])
                for v in graph[u]:
                    new_rank[v] += share

        # add sink contribution
        for v in nodes:
            new_rank[v] += sink_contrib

        diff = sum(abs(new_rank[n] - rank[n]) for n in nodes)
        rank = new_rank
        if diff < tol:
            break

    return rank


###############################################################################
# 4) Checking for Outliers in Centralities
###############################################################################

def detect_centrality_anomalies(degree_dict, closeness_dict, betweenness_dict, pagerank_dict):
    """
    Examine the four centrality dictionaries and detect possible outliers.
    
    Returns a list of textual warnings for nodes that exceed certain thresholds, e.g.:
      - closeness >= 0.9999
      - betweenness / pagerank / degree > mean + 3*std
    """
    anomalies = []

    # 1) High closeness
    for airport, val in closeness_dict.items():
        if val >= 0.9999:
            anomalies.append(f"[CLOSENESS] {airport} = {val:.4f} (Suspiciously high)")

    # 2) Betweenness Outliers
    bet_vals = list(betweenness_dict.values())
    if len(bet_vals) > 1:
        mean_bet = np.mean(bet_vals)
        std_bet = np.std(bet_vals)
        cutoff_bet = mean_bet + 3 * std_bet
        for airport, val in betweenness_dict.items():
            if val > cutoff_bet:
                anomalies.append(f"[BETWEENNESS] {airport} = {val:.2f} (> 3 std from mean)")

    # 3) PageRank Outliers
    pr_vals = list(pagerank_dict.values())
    if len(pr_vals) > 1:
        mean_pr = np.mean(pr_vals)
        std_pr = np.std(pr_vals)
        cutoff_pr = mean_pr + 3 * std_pr
        for airport, val in pagerank_dict.items():
            if val > cutoff_pr:
                anomalies.append(f"[PAGERANK] {airport} = {val:.4f} (> 3 std from mean)")

    # 4) Degree Outliers
    deg_vals = list(degree_dict.values())
    if len(deg_vals) > 1:
        mean_deg = np.mean(deg_vals)
        std_deg = np.std(deg_vals)
        cutoff_deg = mean_deg + 3 * std_deg
        for airport, val in degree_dict.items():
            if val > cutoff_deg:
                anomalies.append(f"[DEGREE] {airport} = {val} (> 3 std from mean)")

    return anomalies


###############################################################################
# 5) compare_centralities_with_checks
###############################################################################

def compare_centralities_with_checks(df):
    """
    Extended version of a typical "compare_centralities":
      - Build a directed graph from df.
      - Compute the four measures (Degree, Closeness, Betweenness, PageRank) for all nodes.
      - Plot their distributions in histograms.
      - Return the top-5 nodes for each measure.
      - Also run anomaly checks (detect_centrality_anomalies) to flag potential outliers.
    
    Returns a dictionary with:
      {
        "top_degree": [...],
        "top_closeness": [...],
        "top_betweenness": [...],
        "top_pagerank": [...],
        "anomalies": [list_of_anomalies]
      }
    """
    # 1) Build the directed flight graph
    graph = build_directed_graph(df)
    nodes = list(graph.keys())

    # 2) Compute each measure
    #    (A) Degree
    degree_dict = compute_all_degree_centralities(graph=graph)

    #    (B) Closeness
    closeness_dict = compute_all_closeness_centralities(graph=graph)

    #    (C) Betweenness
    betweenness_dict = brandes_betweenness_centrality(graph=graph)

    #    (D) PageRank
    pagerank_dict = compute_pagerank(graph=graph)

    # 3) Prepare for histogram plotting
    deg_vals = [degree_dict[n] for n in nodes]
    clos_vals = [closeness_dict[n] for n in nodes]
    bet_vals = [betweenness_dict[n] for n in nodes]
    pr_vals  = [pagerank_dict[n] for n in nodes]

    # 4) Plot histograms for each distribution
    plt.figure(figsize=(12, 8))

    # (i) Degree Distribution
    plt.subplot(2, 2, 1)
    plt.hist(deg_vals, bins=30, color='skyblue', edgecolor='black')
    plt.title("Degree Centrality Distribution")

    # (ii) Closeness Distribution
    plt.subplot(2, 2, 2)
    plt.hist(clos_vals, bins=30, color='orange', edgecolor='black')
    plt.title("Closeness Centrality Distribution")

    # (iii) Betweenness Distribution
    plt.subplot(2, 2, 3)
    plt.hist(bet_vals, bins=30, color='green', edgecolor='black')
    plt.title("Betweenness Centrality Distribution")

    # (iv) PageRank Distribution
    plt.subplot(2, 2, 4)
    plt.hist(pr_vals, bins=30, color='purple', edgecolor='black')
    plt.title("PageRank Distribution")

    plt.tight_layout()
    plt.show()

    # 5) Identify top-5 for each measure
    def top5(dictionary):
        return sorted(dictionary.items(), key=lambda x: x[1], reverse=True)[:5]

    top_degree      = top5(degree_dict)
    top_closeness   = top5(closeness_dict)
    top_betweenness = top5(betweenness_dict)
    top_pagerank    = top5(pagerank_dict)

    # 6) Check for anomalies / outliers
    anomalies = detect_centrality_anomalies(
        degree_dict,
        closeness_dict,
        betweenness_dict,
        pagerank_dict
    )

    # 7) Return all results
    return {
        "top_degree":      top_degree,
        "top_closeness":   top_closeness,
        "top_betweenness": top_betweenness,
        "top_pagerank":    top_pagerank,
        "anomalies":       anomalies
    }


##########################################################################################
##########################################################################################
##########################################################################################


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from collections import defaultdict, deque

def inspect_airport(df, airport_code):
    """
    Inspect detailed statistics for a single airport (airport_code) within the DataFrame (df).
    Main actions:
      - Count total rows where this airport appears as either Origin or Destination.
      - List the most common destinations (if airport_code is in 'Origin_airport').
      - List the most common origins (if airport_code is in 'Destination_airport').
      - Show basic statistics on 'Passengers', 'Flights', 'Seats' columns, if available.
      - Print a small sample (head) of these rows.

    Parameters:
      df (pd.DataFrame): Flight DataFrame with columns like 'Origin_airport', 'Destination_airport', ...
      airport_code (str): The specific airport code to inspect.
    """
    # Filter rows where 'airport_code' appears as either origin OR destination
    mask = (df['Origin_airport'] == airport_code) | (df['Destination_airport'] == airport_code)
    sub_df = df[mask]

    print(f"\n=== Inspection for Airport: {airport_code} ===")
    print(f"Total rows found: {len(sub_df)}")

    # Count how many times airport_code is the 'Origin'
    sub_df_origin = sub_df[sub_df['Origin_airport'] == airport_code]
    print(f"  - As Origin: {len(sub_df_origin)} rows")
    if len(sub_df_origin) > 0:
        # Show top 10 destinations from this origin
        dest_counts = sub_df_origin['Destination_airport'].value_counts().head(10)
        print("  - Top 10 Destination_airport:")
        print(dest_counts)

    # Count how many times airport_code is the 'Destination'
    sub_df_dest = sub_df[sub_df['Destination_airport'] == airport_code]
    print(f"  - As Destination: {len(sub_df_dest)} rows")
    if len(sub_df_dest) > 0:
        origin_counts = sub_df_dest['Origin_airport'].value_counts().head(10)
        print("  - Top 10 Origin_airport:")
        print(origin_counts)

    # If columns like 'Passengers', 'Flights', 'Seats' exist, compute basic stats
    if 'Passengers' in df.columns:
        mean_pass = sub_df['Passengers'].mean()
        print(f"  - Average Passengers on routes involving {airport_code}: {mean_pass:.2f}")
    if 'Flights' in df.columns:
        mean_flights = sub_df['Flights'].mean()
        print(f"  - Average Flights on routes involving {airport_code}: {mean_flights:.2f}")
    if 'Seats' in df.columns:
        mean_seats = sub_df['Seats'].mean()
        print(f"  - Average Seats on routes involving {airport_code}: {mean_seats:.2f}")

    # Print an example sample of up to 5 rows for convenience
    print("\nExample up to 5 rows:")
    print(sub_df.head(5))


def summarize_routes(df):
    """
    Summarize the flight routes by grouping (Origin_airport, Destination_airport).

    Returns a DataFrame with aggregated stats, e.g.:
    - For each (Origin_airport, Destination_airport),
      sum of 'Passengers', 'Flights', 'Seats', etc.

    By default, it uses sum, but you can customize if you want average or other metrics.
    """
    grouped = (
        df.groupby(['Origin_airport', 'Destination_airport'], as_index=False)
          .agg({
              'Passengers': 'sum',
              'Flights': 'sum',
              'Seats': 'sum'
              # If more columns exist, you can add them with custom .agg(...) logic
          })
    )
    return grouped


def top_connections_for_airport(df_summary, airport_code, top=10):
    """
    Given a summary DataFrame (df_summary) with columns ['Origin_airport','Destination_airport','Passengers','Flights','Seats'],
    extract the top (by 'Passengers') outgoing and incoming connections for 'airport_code'.

    Parameters:
      - df_summary (pd.DataFrame): Usually an aggregated DataFrame from `summarize_routes`.
      - airport_code (str): The specific airport to analyze connections for.
      - top (int): How many top routes to list (based on descending 'Passengers').

    Returns:
      (outgoing_sorted, incoming_sorted) DataFrames, each limited to 'top' rows, sorted by 'Passengers' descending.
    """
    # Outgoing routes
    outgoing = df_summary[df_summary['Origin_airport'] == airport_code]
    outgoing_sorted = outgoing.sort_values(by='Passengers', ascending=False).head(top)
    
    # Incoming routes
    incoming = df_summary[df_summary['Destination_airport'] == airport_code]
    incoming_sorted = incoming.sort_values(by='Passengers', ascending=False).head(top)
    
    return outgoing_sorted, incoming_sorted


def check_degree_from_dataframe(df_clean, airport_code):
    """
    A quick check to see how many unique 'Destination_airport' and 'Origin_airport' connect to airport_code.

    This does not run BFS or compute formal in/out-degree from an adjacency list.
    It simply uses sets to count distinct routes in the DataFrame.

    Print:
      - how many unique destinations (airport_code -> ???)
      - how many unique origins (??? -> airport_code)
      - total (out + in) => might overlap if the same node is both a dest and an origin
    """
    # Outgoing routes from airport_code
    out_mask = (df_clean['Origin_airport'] == airport_code)
    out_dest_set = set(df_clean.loc[out_mask, 'Destination_airport'])
    
    # Incoming routes into airport_code
    in_mask = (df_clean['Destination_airport'] == airport_code)
    in_orig_set = set(df_clean.loc[in_mask, 'Origin_airport'])
    
    print(f"\n=== Connections for {airport_code} ===")
    print(f"Outgoing from {airport_code}: {len(out_dest_set)} unique destinations.")
    print(f"Incoming to {airport_code}: {len(in_orig_set)} unique origins.")
    print(f"Total degree (from sets) = {len(out_dest_set) + len(in_orig_set)} (note that overlap is possible).")


###############################################################################
# Building a minimal adjacency set and extracting a subgraph for visualization
###############################################################################

def build_directed_graph_simple(df):
    """
    A simplified adjacency representation (using sets to avoid duplicate edges).
    We only care that if a row says (Origin_airport -> Destination_airport), we store that in the graph set.
    """
    graph = defaultdict(set)
    for _, row in df.iterrows():
        origin = row["Origin_airport"]
        dest = row["Destination_airport"]
        if origin != dest:
            graph[origin].add(dest)
    return graph


def extract_subgraph(graph, airport, radius=1):
    """
    Extract a subgraph of all nodes that are within 'radius' BFS steps from 'airport'.
    radius=1 => only direct out-neighbors of 'airport'.
    radius=2 => neighbors and neighbors' neighbors, etc.

    NOTE: This is a one-direction BFS (only following outgoing edges).
          If you also want to consider incoming edges, you'd adapt the BFS or treat the graph as undirected.
    """
    visited = set()
    queue = deque([(airport, 0)])
    sub_nodes = set()

    while queue:
        current, dist = queue.popleft()
        if current not in visited:
            visited.add(current)
            sub_nodes.add(current)
            if dist < radius:
                # Add the direct out-neighbors
                for neigh in graph[current]:
                    queue.append((neigh, dist + 1))
    return sub_nodes


def visualize_subgraph(df_clean, airport_code, radius=1):
    """
    Visualize (with NetworkX) a small subgraph of up to 'radius' BFS steps from 'airport_code'.
    Only follows edges in the forward direction (outgoing adjacency).

    1) Build a simple adjacency set from df_clean (build_directed_graph_simple).
    2) Extract the sub_nodes within radius BFS steps.
    3) Create a NetworkX DiGraph, add edges only among those sub_nodes.
    4) Use a layout (spring_layout) and draw the subgraph.
    """
    g = build_directed_graph_simple(df_clean)
    sub_nodes = extract_subgraph(g, airport_code, radius=radius)
    
    # Construct a NetworkX DiGraph from those sub_nodes
    G = nx.DiGraph()
    for n in sub_nodes:
        G.add_node(n)
    for n in sub_nodes:
        for neigh in g[n]:
            if neigh in sub_nodes:
                G.add_edge(n, neigh)
    
    # Draw
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G, k=0.5, seed=42)
    nx.draw(G, pos, with_labels=True,
            node_color='lightblue',
            node_size=700,
            arrowstyle='-|>', arrowsize=12)

    plt.title(f"Subgraph around '{airport_code}' (radius={radius})")
    plt.show()


###############################################################################
# Handling Volume Outliers and Suspicious Nodes
###############################################################################

def detect_volume_outliers(df, column="Passengers", z_threshold=5):
    """
    Detect outliers in a numeric column (default='Passengers') by comparing absolute
    z-score to a 'z_threshold'. Rows that exceed this threshold are flagged as outliers.

    Return the subset of df that has these outlier rows.

    Example:
      outliers_df = detect_volume_outliers(df, column="Passengers", z_threshold=5)
      if not outliers_df.empty:
          print("Found potential volume outliers:")
          print(outliers_df)
    """
    if column not in df.columns:
        return pd.DataFrame()  # Return empty DataFrame if column does not exist
    
    col_vals = df[column]
    mean_val = col_vals.mean()
    std_val = col_vals.std()

    if std_val == 0:
        # If the column has no variability, no outliers based on std
        return pd.DataFrame()

    outliers = df[(col_vals - mean_val).abs() > z_threshold * std_val]
    return outliers


def remove_suspicious_nodes(df_clean, suspicious_nodes):
    """
    Remove all rows (flights) that involve any airport in 'suspicious_nodes'.
    Returns a 'cleaned' df_new.

    Typically used if you identify certain airports (like placeholders or erroneous codes)
    you want to exclude from your analysis.

    Example usage:
      suspicious_airports = ["FVS", "YIP"]  # hypothetical codes
      df_cleaned = remove_suspicious_nodes(df, suspicious_airports)
    """
    mask = (~df_clean['Origin_airport'].isin(suspicious_nodes)) & \
           (~df_clean['Destination_airport'].isin(suspicious_nodes))
    df_new = df_clean[mask]
    return df_new


###############################################################################
# Example Usage (placeholder)
###############################################################################

if __name__ == "__main__":
    # Example pseudo-code usage:
    # df = pd.read_csv("flight_data.csv")
    #
    # # 1) Inspect a single airport
    # inspect_airport(df, "ATL")
    #
    # # 2) Summarize routes
    # route_summary = summarize_routes(df)
    # outg, incg = top_connections_for_airport(route_summary, "ATL", top=10)
    # print("Top outgoing from ATL:\n", outg)
    # print("Top incoming to ATL:\n", incg)
    #
    # # 3) Check degree quickly (based on sets, not BFS)
    # check_degree_from_dataframe(df, "ATL")
    #
    # # 4) Visualize a subgraph around "ATL" with radius=1
    # visualize_subgraph(df, "ATL", radius=1)
    #
    # # 5) Detect volume outliers
    # volume_outs = detect_volume_outliers(df, column="Passengers", z_threshold=5)
    # if not volume_outs.empty:
    #     print("Volume outliers found:\n", volume_outs)
    #
    # # 6) Remove suspicious airports if needed
    # suspicious_airports = ["XYZ", "ABC"]  # hypothetical
    # df_cleaned = remove_suspicious_nodes(df, suspicious_airports)
    #
    # # ... proceed with further analysis ...
    #
    pass
