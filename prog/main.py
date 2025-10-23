import numpy as np
import matplotlib.pyplot as plt
import time
import os
import json
import sys
from datetime import datetime
import networkx as nx

# --- Import the clustering algorithms from your separate file ---
# MODIFIED: Import PAM_Network_RiskAware to run it directly
from clustering_algorithms import CLARA, CLARA_Network, CLARA_Network_RiskAware, PAM_Network_RiskAware

class Logger(object):
    """
    A simple logger class that writes to both the console and a file.
    """
    def __init__(self, filename="log.txt"):
        self.terminal = sys.stdout
        self.log = open(filename, 'w')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()

# --- Helper Functions for Data and Network Graph Handling ---

def build_road_network_graph(all_nodes, connections):
    """
    Builds a NetworkX graph from node and connection data.
    """
    G = nx.Graph()
    node_positions = {node['id']: (node['x'], node['y']) for node in all_nodes}
    for node in all_nodes:
        if not node.get('is_house', False):
            G.add_node(node['id'], pos=(node['x'], node['y']))
    for conn in connections:
        node1_id, node2_id = conn['node1_id'], conn['node2_id']
        if G.has_node(node1_id) and G.has_node(node2_id):
            pos1, pos2 = node_positions[node1_id], node_positions[node2_id]
            distance = np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
            G.add_edge(node1_id, node2_id, weight=distance)
    return G, node_positions

def map_houses_to_road_network(house_nodes, road_nodes):
    """
    Finds the closest road node for each house.
    """
    house_to_road_map = {}
    road_node_coords = np.array([[node['x'], node['y']] for node in road_nodes])
    road_node_ids = [node['id'] for node in road_nodes]
    n_houses = len(house_nodes)
    last_update_time = time.time()
    for i, house in enumerate(house_nodes):
        house_coord = np.array([house['x'], house['y']])
        distances = np.sqrt(np.sum((road_node_coords - house_coord)**2, axis=1))
        house_to_road_map[house['id']] = road_node_ids[np.argmin(distances)]
        if time.time() - last_update_time > 5.0:
            progress = (i + 1) / n_houses * 100
            print(f"    ...mapping progress: {progress:.1f}% ({i+1}/{n_houses} houses)", end='\r')
            last_update_time = time.time()
    print(" " * 80, end='\r')
    return house_to_road_map

def generate_initial_map(house_nodes, road_nodes, node_positions, timestamp):
    """
    Generates and displays a map of all nodes, highlighting pre-defined centroids.
    """
    print("\nGenerating initial map of all nodes...")
    plt.figure(figsize=(12, 10))

    # Plot road nodes as a background network
    road_coords = np.array([node_positions[node['id']] for node in road_nodes if node['id'] in node_positions])
    if road_coords.any():
        plt.scatter(road_coords[:, 0], road_coords[:, 1], c='gray', s=5, alpha=0.5, label='Road Intersections')

    # Plot all house nodes
    house_coords = np.array([[node['x'], node['y']] for node in house_nodes])
    if house_coords.any():
        plt.scatter(house_coords[:, 0], house_coords[:, 1], c='skyblue', s=20, alpha=0.8, label='Houses')

    # Identify and plot pre-defined centroid nodes
    centroid_nodes = [node for node in house_nodes if node.get('is_centroid')]
    if centroid_nodes:
        centroid_coords = np.array([[node['x'], node['y']] for node in centroid_nodes])
        plt.scatter(centroid_coords[:, 0], centroid_coords[:, 1],
                    c='white',          # Big white colored nodes
                    marker='o',
                    s=200,              # Larger size
                    edgecolors='black', # With a black edge to stand out
                    linewidths=1.5,
                    label='Pre-defined Centroids',
                    zorder=10)          # Ensure they are plotted on top

    plt.title('Initial Map of All Nodes', fontsize=16, fontweight='bold')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.gca().set_aspect('equal', adjustable='box')

    initial_map_path = f'initial_node_map_{timestamp}.png'
    plt.savefig(initial_map_path, dpi=150, bbox_inches='tight')
    print(f"Initial map saved as '{initial_map_path}'")
    plt.show()

def calculate_risk_scores(nodes, bubbles):
    """
    Calculates a risk score for each node based on how many bubbles it's inside.
    Returns a dictionary mapping node_id to its integer risk score.
    """
    print("\nCalculating risk scores from fire incident bubbles...")
    risk_scores = {node['id']: 0 for node in nodes}
    node_coords = {node['id']: (node['x'], node['y']) for node in nodes}
    
    for i, bubble in enumerate(bubbles):
        center = np.array([bubble['center_x'], bubble['center_y']])
        radius_sq = bubble['radius'] ** 2 # Use squared radius for efficiency
        
        for node_id, coords in node_coords.items():
            dist_sq = np.sum((np.array(coords) - center) ** 2)
            if dist_sq <= radius_sq:
                risk_scores[node_id] += 1
                
    high_risk_nodes = sum(1 for score in risk_scores.values() if score > 0)
    print(f"Risk calculation complete. Found {high_risk_nodes} nodes within at least one bubble.")
    return risk_scores

# --- NEW: Helper function for plotting risk-aware results ---

def plot_risk_aware_clusters(ax, title, labels, medoids, house_nodes, bubbles, k, cost):
    """
    A dedicated helper function to plot the results of a risk-aware clustering algorithm.
    """
    # Plot bubbles first as background
    for bubble in bubbles:
        circle = plt.Circle((bubble['center_x'], bubble['center_y']), bubble['radius'],
                            color='red', alpha=0.15, ec='none')
        ax.add_patch(circle)

    # Plot clusters
    colors = plt.cm.viridis(np.linspace(0, 1, k))
    X_coords = np.array([[node['x'], node['y']] for node in house_nodes])
    for i in range(k):
        cluster_mask = (labels == i)
        ax.scatter(X_coords[cluster_mask, 0], X_coords[cluster_mask, 1],
                   c=[colors[i]], label=f'Cluster {i+1}', alpha=0.7, s=20)
    
    # Plot Medoids
    medoid_houses = [node for node in house_nodes if node['id'] in medoids]
    medoid_coords = np.array([[node['x'], node['y']] for node in medoid_houses])
    ax.scatter(medoid_coords[:, 0], medoid_coords[:, 1], c='red', marker='X', s=400,
               edgecolors='white', linewidths=2, label='Medoids', zorder=5)
    
    full_title = f'{title}\nFinal Weighted Cost: {cost:.2f}'
    ax.set_title(full_title, fontsize=14, fontweight='bold')
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.set_aspect('equal', adjustable='box')

# --- REVISED: Main Execution Function for Risk-Aware Comparison ---

def run_risk_aware_comparison(json_file_path, k, num_samples, fire_risk_weight, algo_mode="BOTH"):
    """
    Runs a comparison between CLARA and PAM for risk-aware network clustering.
    - algo_mode: "CLARA", "PAM", or "BOTH".
    """
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_filename = f"risk_aware_comparison_log_{timestamp}.txt"
    logger = Logger(log_filename)
    original_stdout = sys.stdout
    sys.stdout = logger

    try:
        # 1. Load Data and perform pre-processing
        with open(json_file_path, 'r') as f: user_data = json.load(f)
        all_nodes = user_data.get('nodes', [])
        connections = user_data.get('connections', [])
        bubbles = user_data.get('bubbles', [])
        house_nodes = [node for node in all_nodes if node.get('is_house', False)]
        road_nodes = [node for node in all_nodes if not node.get('is_house', False)]

        risk_scores = calculate_risk_scores(house_nodes, bubbles)
        road_graph, node_positions = build_road_network_graph(all_nodes, connections)
        house_to_road_map = map_houses_to_road_network(house_nodes, road_nodes)
        sample_size = min(40 + 2 * k, len(house_nodes))

        # --- 2. Algorithm Execution ---
        results = {}

        if algo_mode in ["CLARA", "BOTH"]:
            print("\n\n" + "#" * 70)
            print("### RUNNING: CLARA Risk-Aware Network Clustering ###")
            print("#" * 70)
            clara_risk_network = CLARA_Network_RiskAware(
                k=k, num_samples=num_samples, sample_size=sample_size, use_build=True,
                road_graph=road_graph, node_positions=node_positions, house_to_road_map=house_to_road_map,
                risk_scores=risk_scores, fire_risk_weight=fire_risk_weight
            )
            start_time = time.time()
            clara_risk_network.fit(house_nodes)
            runtime = time.time() - start_time
            if clara_risk_network.medoids_ is not None:
                results['CLARA'] = {
                    'cost': clara_risk_network.best_cost_, 'runtime': runtime,
                    'medoids': clara_risk_network.medoids_, 'labels': clara_risk_network.labels_
                }

        if algo_mode in ["PAM", "BOTH"]:
            print("\n\n" + "#" * 70)
            print("### RUNNING: PAM Risk-Aware Network Clustering ###")
            print("#" * 70)
            print("WARNING: Running PAM on the full dataset can be very slow.")
            pam_risk_network = PAM_Network_RiskAware(
                k=k, use_build=True, road_graph=road_graph, node_positions=node_positions,
                house_to_road_map=house_to_road_map, risk_scores=risk_scores,
                fire_risk_weight=fire_risk_weight, verbose=True
            )
            start_time = time.time()
            pam_risk_network.fit(house_nodes) # Fit on the entire dataset
            runtime = time.time() - start_time
            
            # Convert medoid indices to actual house IDs
            medoid_ids = [house_nodes[i]['id'] for i in pam_risk_network.medoids]
            final_cost = pam_risk_network.calculate_total_cost(house_nodes, pam_risk_network.medoids, pam_risk_network.labels_)
            results['PAM'] = {
                'cost': final_cost, 'runtime': runtime,
                'medoids': medoid_ids, 'labels': pam_risk_network.labels_
            }

        # --- 3. Display Results ---
        print("\n\n" + "=" * 80)
        print("RISK-AWARE NETWORK CLUSTERING FINAL RESULTS")
        print("=" * 80)
        if algo_mode == "BOTH" and 'CLARA' in results and 'PAM' in results:
            print(f"{'Algorithm':<30} | {'Final Weighted Cost':<25} | {'Runtime (seconds)':<20}")
            print("-" * 80)
            print(f"{'CLARA Risk-Aware (Sampled)':<30} | {results['CLARA']['cost']:<25.2f} | {results['CLARA']['runtime']:<20.2f}")
            print(f"{'PAM Risk-Aware (Full Data)':<30} | {results['PAM']['cost']:<25.2f} | {results['PAM']['runtime']:<20.2f}")
        else:
            for algo_name, res in results.items():
                print(f"Algorithm: {algo_name} Risk-Aware")
                print(f"Final Weighted Cost: {res['cost']:.2f}")
                print(f"Runtime: {res['runtime']:.2f} seconds")
                print(f"Final Medoid House IDs: {res['medoids']}")
        print("=" * 80)

        # --- 4. Visualize Final Clustering ---
        if not results:
            print("\nNo results to visualize.")
            return

        print("\nGenerating final visualization...")
        if algo_mode == "BOTH" and 'CLARA' in results and 'PAM' in results:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 11))
            plot_risk_aware_clusters(ax1, 'CLARA Risk-Aware Network Clustering', 
                                     results['CLARA']['labels'], results['CLARA']['medoids'], 
                                     house_nodes, bubbles, k, results['CLARA']['cost'])
            plot_risk_aware_clusters(ax2, 'PAM Risk-Aware Network Clustering', 
                                     results['PAM']['labels'], results['PAM']['medoids'], 
                                     house_nodes, bubbles, k, results['PAM']['cost'])
            plt.tight_layout()
        elif 'CLARA' in results:
            plt.figure(figsize=(12, 10))
            plot_risk_aware_clusters(plt.gca(), 'CLARA Risk-Aware Network Clustering',
                                     results['CLARA']['labels'], results['CLARA']['medoids'],
                                     house_nodes, bubbles, k, results['CLARA']['cost'])
        elif 'PAM' in results:
            plt.figure(figsize=(12, 10))
            plot_risk_aware_clusters(plt.gca(), 'PAM Risk-Aware Network Clustering',
                                     results['PAM']['labels'], results['PAM']['medoids'],
                                     house_nodes, bubbles, k, results['PAM']['cost'])
        
        final_plot_path = f'risk_aware_comparison_result_{timestamp}.png'
        plt.savefig(final_plot_path, dpi=150, bbox_inches='tight')
        print(f"Final plot saved as '{final_plot_path}'")
        plt.show()

    finally:
        sys.stdout = original_stdout
        logger.close()
        print(f"\nAll console output has been saved to '{logger.log.name}'")


if __name__ == "__main__":
    # --- USER CONFIGURATION ---
    JSON_FILE_PATH = 'dataset.json'
    K_CLUSTERS = 10
    NUM_SAMPLES = 3
    FIRE_RISK_WEIGHT = 1.5
    
    # --- NEW: CHOOSE WHICH ALGORITHM(S) TO RUN ---
    # Options:
    # "CLARA" -> Runs only the fast, sampled-based CLARA algorithm.
    # "PAM"   -> Runs the PAM algorithm on the FULL dataset (can be slow).
    # "BOTH"  -> Runs both and shows a side-by-side comparison.
    ALGORITHM_MODE = "BOTH"
    # ----------------------------------------------

    # --- CHOOSE WHICH EXPERIMENT TO RUN ---
    # This single function now handles all risk-aware scenarios.
    run_risk_aware_comparison(
        JSON_FILE_PATH, K_CLUSTERS, NUM_SAMPLES, FIRE_RISK_WEIGHT, ALGORITHM_MODE
    )