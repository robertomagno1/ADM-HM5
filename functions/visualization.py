import matplotlib.pyplot as plt
# Visualization function for the busiest routes 
def visualize_busiest_routes(busiest_routes):
    # Visualize the results
    plt.figure(figsize=(12, 6))
    plt.bar(
        busiest_routes["Origin_city"] + " → " + busiest_routes["Destination_city"],
        busiest_routes["Total_passengers"],
        color="skyblue",
        edgecolor="black"
    )

    plt.title("Busiest Routes by Passenger Traffic", fontsize=16)
    plt.xlabel("Routes", fontsize=12)
    plt.ylabel("Total Passengers", fontsize=12)
    plt.xticks(rotation=45, ha="right", fontsize=10)
    plt.tight_layout()
    plt.show()

import pandas as pd
import matplotlib.pyplot as plt

# Return and visualize the top routes by passenger efficiency.
def top_routes_by_efficiency(route_stats, top_n=10):
    # Sort routes by passenger efficiency in descending order
    top_routes = route_stats.sort_values(by="Avg_pass_per_flight", ascending=False).head(top_n)

    # Visualization
    plt.figure(figsize=(12, 6))
    plt.bar(
        top_routes["Origin_city"] + " → " + top_routes["Destination_city"],
        top_routes["Avg_pass_per_flight"],
        color="lightgreen",
        edgecolor="black"
    )

    plt.title("Top Routes by Passenger Efficiency", fontsize=16)
    plt.xlabel("Routes", fontsize=12)
    plt.ylabel("Average Passengers per Flight", fontsize=12)
    plt.xticks(rotation=45, ha="right", fontsize=10)
    plt.tight_layout()
    plt.show()

    return top_routes

