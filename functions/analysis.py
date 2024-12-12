import matplotlib.pyplot as plt
import numpy as np

def analyse_graph_features(flight_network):
    # Calculating number of nodes and edges
    nodes = set()
    num_edges = 0
    for edge in flight_network.edges:
        num_edges += 1
        nodes.add(edge[0])
        nodes.add(edge[1])
    num_nodes = len(nodes)

    # Calculating density of graph 
    density = 2 * num_edges / (num_nodes * (num_nodes - 1))

    # Calculate in-degree and out-degree for each airport
    in_degree = {}
    out_degree = {}
    for node in flight_network.nodes:
        in_degree[node] = 0
        out_degree[node] = 0
    
    for edge in flight_network.edges:
        scr, dst = edge
        in_degree[dst] += 1
        out_degree[scr] += 1

    # For visualisation we extract the values
    in_degree_vals = list(in_degree.values())
    out_degree_vals = list(out_degree.values())

    # Plotting
    plt.figure(figsize=(12, 6))

    # In
    plt.subplot(1, 2, 1)
    plt.hist(in_degree_vals, bins=30, color='skyblue', edgecolor='black')
    plt.title('In-Degree Distribution')
    plt.xlabel('In-Degree')
    plt.ylabel('Frequency')

    # Out
    plt.subplot(1, 2, 2)
    plt.hist(out_degree_vals, bins=30, color='orange', edgecolor='black')
    plt.title('Out-Degree Distribution')
    plt.xlabel('Out-Degree')
    plt.ylabel('Frequency')

    plt.tight_layout()
    plt.show()

    # Combine degrees to compute hubs
    total_degree = {}
    for node in flight_network.nodes:
        total_degree[node] = in_degree[node] + out_degree[node]
    
    # Find 90th percentile thresholds
    in_degree_90th = np.percentile(in_degree_vals, 90)
    out_degree_90th = np.percentile(out_degree_vals, 90)
    total_degree_90th = np.percentile(list(total_degree.values()), 90)

    # Identify hubs
    in_hubs = [node for node, degree in in_degree.items() if degree > in_degree_90th]
    out_hubs = [node for node, degree in out_degree.items() if degree > out_degree_90th]
    total_hubs = [node for node, degree in total_degree.items() if degree > total_degree_90th]

    # Determine if graph is sparse using Rule of Thumb
    # 6. Determine if the graph is sparse or dense
    graph_type = "Dense" if density >= 0.1 else "Sparse"

    return {
        "num_nodes": num_nodes,
        "num_edges": num_edges,
        "density": density,
        "in_degree": in_degree,
        "out_degree": out_degree,
        "total_degree": total_degree,
        "in_hubs": in_hubs,
        "out_hubs": out_hubs,
        "total_hubs": total_hubs,
        "graph_type": graph_type
    }     
