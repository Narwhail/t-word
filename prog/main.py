# main.py

import numpy as np
import matplotlib.pyplot as plt
import time
import os
import json
import sys
from datetime import datetime
import networkx as nx

# --- Import the clustering algorithms from your separate file ---
from clustering_algorithms import CLARA, CLARA_Network, CLARA_Network_RiskAware, PAM_Network_RiskAware
class Logger(object):
    """
    A simple logger class that writes to both the console and a file.
    MODIFIED to prevent writing progress bar updates ('\r') to the log file.
    """
    def __init__(self, filename="log.txt"):
        self.terminal = sys.stdout
        self.log = open(filename, 'w', encoding='utf-8')

    def write(self, message):
        # Always write to the terminal to show live progress
        self.terminal.write(message)

        # Write to the log file only if it's not a temporary progress bar line.
        # We identify progress bar lines as those that start with '\r' but do NOT end with '\n'.
        # Final summary lines that overwrite the progress bar DO end with '\n', so they will be logged.
        if not message.startswith('\r') or message.endswith('\n'):
            self.log.write(message)
            self.log.flush()

    def flush(self):
        # This flush method is needed for compatibility.
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()

# --- Helper Functions for Data and Network Graph Handling ---

def _print_progress_bar(iteration, total, prefix='', suffix='', length=50, fill='█'):
    """
    Prints a terminal progress bar. Call in a loop.
    """
    percent = ("{0:.1f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    # Use carriage return '\r' to stay on the same line.
    sys.stdout.write(f'\r{prefix} |{bar}| {percent}% {suffix}')
    sys.stdout.flush()

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
    Includes a progress bar to show the calculation status.
    Returns a dictionary mapping node_id to its integer risk score.
    """
    print("\nCalculating risk scores from fire incident bubbles...")
    risk_scores = {node['id']: 0 for node in nodes}
    node_coords = {node['id']: (node['x'], node['y']) for node in nodes}
    
    n_bubbles = len(bubbles)
    n_nodes = len(nodes)
    total_checks = n_bubbles * n_nodes
    checks_done = 0

    if total_checks == 0:
        print("No bubbles or nodes to process for risk scoring.")
        return risk_scores

    # Initial display of the progress bar
    _print_progress_bar(0, total_checks, prefix='  Progress:', length=50)

    for i, bubble in enumerate(bubbles):
        center = np.array([bubble['center_x'], bubble['center_y']])
        radius_sq = bubble['radius'] ** 2 # Use squared radius for efficiency
        
        for node_id, coords in node_coords.items():
            dist_sq = np.sum((np.array(coords) - center) ** 2)
            if dist_sq <= radius_sq:
                risk_scores[node_id] += 1
            
            # Update progress bar
            checks_done += 1
            if checks_done % 1000 == 0 or checks_done == total_checks: # Update every 1000 checks or on the last check
                 _print_progress_bar(checks_done, total_checks, prefix='  Progress:', suffix='Complete', length=50)

    # Overwrite the progress bar line with a final summary
    summary_line = f"Risk calculation complete. Found {sum(1 for score in risk_scores.values() if score > 0)} nodes within at least one bubble."
    sys.stdout.write('\r' + summary_line.ljust(80) + '\n')
    sys.stdout.flush()
    
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
    num_clusters = len(np.unique(labels))
    if num_clusters == 0:
        print("Warning: No clusters to plot.")
        return
        
    colors = plt.cm.viridis(np.linspace(0, 1, num_clusters))
    X_coords = np.array([[node['x'], node['y']] for node in house_nodes])
    
    for i in range(num_clusters):
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
# In main.py

def run_risk_aware_comparison(json_file_path, k, num_samples, fire_risk_weight, algo_mode="BOTH", sample_size=None, rand_swap_option=False):
    """
    Runs a comparison between CLARA and PAM for risk-aware network clustering.

    Args:
        json_file_path (str): Path to the dataset JSON file.
        k (int): The number of clusters to form.
        num_samples (int): For CLARA, the number of samples to draw.
        fire_risk_weight (float): The weighting factor for risk scores.
        algo_mode (str): Which algorithm(s) to run. Can be "CLARA", "PAM", or "BOTH".
        sample_size (int, optional): The size of each sample for CLARA. 
                                     If None, a default size is calculated. Defaults to None.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    # 1. Setup Logging and Output Directory
    output_folder = f"run_{algo_mode}_{timestamp}"
    os.makedirs(output_folder, exist_ok=True)
    
    log_filename = os.path.join(output_folder, f"run_log_{timestamp}.txt")
    logger = Logger(log_filename)
    original_stdout = sys.stdout
    sys.stdout = logger
    
    print(f"Output for this run will be saved in: '{output_folder}'")

    try:
        # 2. Load Data and Perform Pre-processing
        print(f"\nLoading data from '{json_file_path}'...")
        with open(json_file_path, 'r') as f: user_data = json.load(f)
        all_nodes = user_data.get('nodes', [])
        connections = user_data.get('connections', [])
        bubbles = user_data.get('bubbles', [])
        
        house_nodes = [node for node in all_nodes if node.get('is_house') is True]
        if not house_nodes:
            print("FATAL ERROR: No house nodes were found in the dataset. Cannot proceed.")
            return
            
        road_nodes = [node for node in all_nodes if not node.get('is_house', False)]

        risk_scores = calculate_risk_scores(house_nodes, bubbles)
        print("\nBuilding road network graph...")
        road_graph, node_positions = build_road_network_graph(all_nodes, connections)

        # ======================= START: NEW DIAGNOSTIC CODE =======================
        # Check if the graph is fully connected.
        if not nx.is_connected(road_graph):
            print("\n" + "="*80)
            print("FATAL ERROR: The road network graph is not fully connected.")
            print("This means there are isolated islands of nodes with no path between them.")
            print("The clustering algorithm cannot proceed because distances between these")
            print("islands are infinite.")
            print("-" * 80)

            # nx.connected_components returns a list of sets, where each set is an island of nodes.
            connected_components = list(nx.connected_components(road_graph))
            num_components = len(connected_components)
            
            print(f"Found {num_components} separate components (islands) in the graph.")
            print("Below is a list of the node IDs in each isolated component:")

            # Iterate through each component and print its members.
            for i, component in enumerate(connected_components):
                # Sort the nodes for consistent and readable output
                component_nodes = sorted(list(component))
                print(f"\n--- Component {i + 1} ({len(component_nodes)} nodes) ---")
                
                # Print nodes in chunks to avoid one massive line in the terminal
                nodes_per_line = 10
                for j in range(0, len(component_nodes), nodes_per_line):
                    print(f"  {component_nodes[j:j+nodes_per_line]}")
            
            print("\n" + "="*80)
            print("ACTION REQUIRED: To fix this, you must either:")
            print("  1. Review your dataset and add connections to link these components.")
            print("  2. Filter your dataset to run the analysis on only one component at a time.")
            print("="*80)
            
            # Stop the execution cleanly to prevent the clustering error
            return
        # ======================== END: NEW DIAGNOSTIC CODE ========================

        print("Mapping houses to the nearest road node...")
        house_to_road_map = map_houses_to_road_network(house_nodes, road_nodes)
        
        # 3. Handle CLARA Sample Size Logic
        clara_sample_size = 0
        if algo_mode in ["CLARA", "BOTH"]:
            if sample_size is None:
                # If no sample size is provided, calculate a default one.
                clara_sample_size = min(40 + 2 * k, len(house_nodes) - 1)
                print(f"\nCLARA sample_size not provided. Defaulting to 40 + 2*k = {clara_sample_size}")
            else:
                # Use the user-provided sample size, but ensure it's valid.
                if sample_size >= len(house_nodes):
                    print(f"\nWARNING: Provided sample_size ({sample_size}) is >= number of houses ({len(house_nodes)}).")
                    clara_sample_size = len(house_nodes) - 1
                    print(f"         Adjusting sample_size to {clara_sample_size}.")
                elif sample_size <= k:
                     print(f"\nWARNING: Provided sample_size ({sample_size}) must be greater than k ({k}).")
                     clara_sample_size = k + 1
                     print(f"         Adjusting sample_size to {clara_sample_size}.")
                else:
                    clara_sample_size = sample_size
            print(f"\nUsing CLARA sample_size: {clara_sample_size}")


        # 4. Algorithm Execution
        results = {}

        if algo_mode in ["CLARA", "BOTH"]:
            print("\n\n" + "#" * 70)
            print("### RUNNING: CLARA Risk-Aware Network Clustering ###")
            print("#" * 70)
            clara_risk_network = CLARA_Network_RiskAware(
                k=k, 
                num_samples=num_samples, 
                sample_size=clara_sample_size,
                use_build=True,
                road_graph=road_graph, 
                node_positions=node_positions, 
                house_to_road_map=house_to_road_map,
                risk_scores=risk_scores, 
                fire_risk_weight=fire_risk_weight,
                verbose=True, 
                output_dir=os.path.join(output_folder, 'clara_iterations'),
                rand_swap=rand_swap_option  # <-- ADD THIS LINE
            )
            start_time = time.time()
            clara_risk_network.fit(house_nodes)
            runtime = time.time() - start_time
            print(f"\n### CLARA execution finished in {runtime:.2f} seconds. ###")

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
                k=k, 
                use_build=True, 
                road_graph=road_graph, 
                node_positions=node_positions,
                house_to_road_map=house_to_road_map, 
                risk_scores=risk_scores,
                fire_risk_weight=fire_risk_weight, 
                verbose=True,
                output_dir=os.path.join(output_folder, 'pam_iterations')
            )
            start_time = time.time()
            pam_risk_network.fit(house_nodes)
            runtime = time.time() - start_time
            print(f"\n### PAM execution finished in {runtime:.2f} seconds. ###")
            
            medoid_ids = [house_nodes[i]['id'] for i in pam_risk_network.medoids]
            results['PAM'] = {
                'cost': pam_risk_network.final_cost_, 
                'runtime': runtime,
                'medoids': medoid_ids, 
                'labels': pam_risk_network.labels_
            }

        # 5. Display Final Results
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
                print(f"\n--- {algo_name} Risk-Aware ---")
                print(f"  Final Weighted Cost: {res['cost']:.2f}")
                print(f"  Runtime: {res['runtime']:.2f} seconds")
                print(f"  Final Medoid House IDs: {res['medoids']}")
        print("=" * 80)

        # 6. Visualize Final Clustering
        if not results:
            print("\nNo results to visualize.")
            return

        print("\nGenerating final visualization...")
        num_plots = len(results)
        if num_plots == 2:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 11))
            plot_risk_aware_clusters(ax1, 'CLARA Risk-Aware Network Clustering', 
                                     results['CLARA']['labels'], results['CLARA']['medoids'], 
                                     house_nodes, bubbles, k, results['CLARA']['cost'])
            plot_risk_aware_clusters(ax2, 'PAM Risk-Aware Network Clustering', 
                                     results['PAM']['labels'], results['PAM']['medoids'], 
                                     house_nodes, bubbles, k, results['PAM']['cost'])
            plt.tight_layout()
        elif num_plots == 1:
            algo_name = list(results.keys())[0]
            res = results[algo_name]
            plt.figure(figsize=(12, 10))
            plot_risk_aware_clusters(plt.gca(), f'{algo_name} Risk-Aware Network Clustering',
                                     res['labels'], res['medoids'],
                                     house_nodes, bubbles, k, res['cost'])
        
        final_plot_path = os.path.join(output_folder, f'risk_aware_comparison_result_{timestamp}.png')
        plt.savefig(final_plot_path, dpi=150, bbox_inches='tight')
        print(f"Final plot saved as '{final_plot_path}'")
        plt.show()

    except Exception as e:
        print(f"\nAN ERROR OCCURRED: {e}")
        import traceback
        traceback.print_exc() # Prints detailed traceback for debugging

    finally:
        # 7. Restore Standard Output and Close Logger
        sys.stdout = original_stdout
        logger.close()
        print(f"\nAll console output has been saved to '{logger.log.name}'")
        

if __name__ == "__main__":
    # --- USER CONFIGURATION ---
    # The path to your JSON dataset file.
    JSON_FILE_PATH = 'dataset.json'
    
    # The desired number of clusters (medoids).
    K_CLUSTERS = 3
    
    # --- CLARA Specific Parameters ---
    # The number of different random samples to process.
    NUM_SAMPLES = 3
    
    # Set a specific sample size for CLARA.
    # This value must be greater than K_CLUSTERS.
    # If set to None, a default size of (40 + 2*k) will be calculated automatically.
    CLARA_SAMPLE_SIZE = 400 # Example: a fixed size of 100
    
    # --- Risk-Aware Parameters ---
    # The weighting factor for fire risk.
    FIRE_RISK_WEIGHT = 1.5
    
    # --- General Algorithm Settings ---
    # Determines which algorithm(s) to run.
    # Options: "CLARA", "PAM", or "BOTH".
    ALGORITHM_MODE = "CLARA" 
    # ----------------------------------------------

    # This function call starts the entire process using the parameters defined above.
    run_risk_aware_comparison(
        json_file_path=JSON_FILE_PATH, 
        k=K_CLUSTERS, 
        num_samples=NUM_SAMPLES, 
        fire_risk_weight=FIRE_RISK_WEIGHT, 
        algo_mode=ALGORITHM_MODE,
        sample_size=CLARA_SAMPLE_SIZE,  # Pass the new sample size parameter
        rand_swap_option=True,

    )