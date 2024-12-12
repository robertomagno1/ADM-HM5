import functions.analysis as analysis
import pandas as pd

def summarize_graph_features(flight_network):
    # Analyse
    analysis_res = analysis.analyse_graph_features(flight_network)

    # Extract results for easier access
    num_nodes = analysis_res["num_nodes"]
    num_edges = analysis_res["num_edges"]
    density = analysis_res["density"]
    in_hubs = analysis_res["in_hubs"]
    out_hubs = analysis_res["out_hubs"]
    total_hubs = analysis_res["total_hubs"]
    in_degree = analysis_res["in_degree"]
    out_degree = analysis_res["out_degree"]
    total_degree = analysis_res["total_degree"]
    graph_type = analysis_res["graph_type"]


    # Create a df for hubs
    hubs_data = {
        'Node': list(set(in_hubs + out_hubs + total_hubs)),
        'In-Degree': [in_degree[node] for node in set(in_hubs + out_hubs + total_hubs)],
        'Out-Degree': [out_degree[node] for node in set(in_hubs + out_hubs + total_hubs)],
        'Total Degree': [total_degree[node] for node in set(in_hubs + out_hubs + total_hubs)],
    }
    hubs_df = pd.DataFrame(hubs_data)

    # Create a summary report
    print("Graph Summary Report:")
    print(f"Number of airports (nodes): {num_nodes}")
    print(f"Number of flights (edges): {num_edges}")
    print(f"Graph Density: {density:.4f}")
    print(f"Graph Type: {graph_type}")

    print("\nIdentified Hubs (Airports with high degree):")
    print(hubs_df)