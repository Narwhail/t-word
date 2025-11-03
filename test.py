# test_kmedoids.py

import json
import random
import math
import sys
import time
import os
import numpy as np
import matplotlib.pyplot as plt

# --- Helper function to save plots without showing them ---
def save_cluster_plot(clusters, medoids, k, title, filepath):
    """
    Generates and saves a plot of the current cluster state to a file.
    """
    plt.figure(figsize=(12, 10))
    colors = plt.cm.viridis(np.linspace(0, 1, k))

    for i, (medoid_id, points) in enumerate(clusters.items()):
        if not points: continue
        # Ensure medoid_id is a key in the medoids dictionary before accessing
        medoid_obj = next((m for m in medoids if m['id'] == medoid_id), None)
        if medoid_obj is None: continue

        cluster_coords = np.array([[p['x'], p['y']] for p in points])
        plt.scatter(cluster_coords[:, 0], cluster_coords[:, 1],
                    c=[colors[i]], label=f'Cluster {i+1}', alpha=0.7, s=30)

    medoid_coords = np.array([[m['x'], m['y']] for m in medoids])
    plt.scatter(medoid_coords[:, 0], medoid_coords[:, 1], c='red', marker='X', s=400,
                edgecolors='white', linewidths=1.5, label='Medoids', zorder=10)

    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.gca().set_aspect('equal', adjustable='box')
    
    plt.savefig(filepath, dpi=100, bbox_inches='tight')
    plt.close()


class KMedoids:
    """
    A class to perform K-medoid clustering using either the
    Partitioning Around Medoids (PAM) or CLARA algorithm for the swap phase.
    """
    def __init__(self, k=2, max_iter=100, swap_phase_method='pam', num_samples=5, sample_size=None, verbose=False):
        if k < 1: raise ValueError("The number of clusters (k) must be greater than 0.")
        if swap_phase_method not in ['pam', 'clara']:
            raise ValueError("swap_phase_method must be either 'pam' or 'clara'.")
        
        self.k = k
        self.max_iter = max_iter
        self.swap_phase_method = swap_phase_method
        self.verbose = verbose
        
        # --- CLARA specific parameters ---
        self.num_samples = num_samples
        self.sample_size = sample_size

        if self.swap_phase_method == 'clara':
            if self.sample_size is None:
                # Set a default sample size based on literature suggestions if not provided
                self.sample_size = 40 + 2 * self.k
                if self.verbose:
                    print(f"CLARA sample_size not provided. Defaulting to 40 + 2*k = {self.sample_size}")
            if self.num_samples < 1:
                raise ValueError("Number of samples for CLARA must be at least 1.")
        
        self.medoids = []
        self.clusters = {}

    def _euclidean_distance(self, p1, p2):
        return math.sqrt((p1['x'] - p2['x'])**2 + (p1['y'] - p2['y'])**2)

    def _calculate_total_cost(self, data, medoids):
        total_cost = 0
        for point in data:
            cost = min(self._euclidean_distance(point, m) for m in medoids)
            total_cost += cost
        return total_cost

    def _assign_points_to_clusters(self, data, medoids):
        clusters = {medoid['id']: [] for medoid in medoids}
        for point in data:
            closest_medoid = min(medoids, key=lambda medoid: self._euclidean_distance(point, medoid))
            clusters[closest_medoid['id']].append(point)
        return clusters

    def _run_pam_swap_phase(self, data_subset, enable_iter_plotting=False, output_dir=None):
        """
        Runs the iterative PAM swap phase on a given subset of data.
        Returns the best medoids found for this subset.
        """
        current_medoids = random.sample(data_subset, self.k)

        if self.verbose and enable_iter_plotting:
            initial_cost = self._calculate_total_cost(data_subset, current_medoids)
            print(f"Initial cost for this run: {initial_cost:.2f}")
            if output_dir:
                initial_clusters = self._assign_points_to_clusters(data_subset, current_medoids)
                filepath = os.path.join(output_dir, 'iteration_000_initial.png')
                save_cluster_plot(initial_clusters, current_medoids, self.k, 'Initial State (Iteration 0)', filepath)

        for i in range(self.max_iter):
            iter_start_time = time.time()
            best_cost_for_subset = self._calculate_total_cost(data_subset, current_medoids)
            potential_best_medoids = current_medoids[:]
            made_swap = False

            for medoid in current_medoids:
                non_medoids_in_subset = [p for p in data_subset if p not in current_medoids]
                for potential_new_medoid in non_medoids_in_subset:
                    temp_medoids = [m for m in current_medoids if m != medoid] + [potential_new_medoid]
                    new_cost = self._calculate_total_cost(data_subset, temp_medoids)
                    
                    if new_cost < best_cost_for_subset:
                        best_cost_for_subset = new_cost
                        potential_best_medoids = temp_medoids
                        made_swap = True
            
            current_medoids = potential_best_medoids
            iter_runtime = time.time() - iter_start_time

            if self.verbose and enable_iter_plotting:
                print(f"Iteration {i + 1}: Cost = {best_cost_for_subset:.2f}, Swap: {made_swap}, Runtime: {iter_runtime:.4f}s")
                if output_dir:
                    current_clusters = self._assign_points_to_clusters(data_subset, current_medoids)
                    filepath = os.path.join(output_dir, f'iteration_{i+1:03d}.png')
                    title = f'State after Iteration {i+1}'
                    save_cluster_plot(current_clusters, current_medoids, self.k, title, filepath)

            if not made_swap:
                if self.verbose and enable_iter_plotting: print("\nAlgorithm converged for this subset.")
                break
        
        return current_medoids

    def fit(self, data, output_dir=None):
        if not data or len(data) < self.k:
            raise ValueError("Not enough data points to form k clusters.")
        
        if self.swap_phase_method == 'clara' and len(data) <= self.sample_size:
            if self.verbose:
                print(f"Warning: Dataset size ({len(data)}) is not larger than sample_size ({self.sample_size}). Running as standard PAM instead.")
            self.swap_phase_method = 'pam'

        overall_start_time = time.time()

        if self.swap_phase_method == 'pam':
            if self.verbose: print("\n--- Running Standard PAM Algorithm ---")
            self.medoids = self._run_pam_swap_phase(data, enable_iter_plotting=True, output_dir=output_dir)

        elif self.swap_phase_method == 'clara':
            if self.verbose:
                print(f"\n--- Running CLARA Algorithm ---")
                print(f"Number of samples: {self.num_samples}, Sample size: {self.sample_size}")
            
            best_medoids_so_far = None
            min_overall_cost = float('inf')

            for i in range(self.num_samples):
                sample_data = random.sample(data, self.sample_size)
                if self.verbose: print(f"\nProcessing Sample {i+1}/{self.num_samples}...")
                
                # Run PAM on the sample; plotting is disabled for these sub-runs
                sample_medoids = self._run_pam_swap_phase(sample_data, enable_iter_plotting=False)
                
                # Evaluate the cost of the sample's medoids against the ENTIRE dataset
                current_overall_cost = self._calculate_total_cost(data, sample_medoids)
                if self.verbose: print(f"  - Cost of sample's medoids on full dataset: {current_overall_cost:.2f}")

                if current_overall_cost < min_overall_cost:
                    min_overall_cost = current_overall_cost
                    best_medoids_so_far = sample_medoids
                    if self.verbose: print(f"  - Found new best set of medoids! (New minimum cost: {min_overall_cost:.2f})")

            self.medoids = best_medoids_so_far

        self.clusters = self._assign_points_to_clusters(data, self.medoids)
        overall_runtime = time.time() - overall_start_time
        if self.verbose: 
            final_cost = self._calculate_total_cost(data, self.medoids)
            print(f"\nFinal total cost for {self.swap_phase_method.upper()}: {final_cost:.2f}")
            print(f"Total algorithm runtime: {overall_runtime:.4f} seconds.")


def load_data_from_json(filepath):
    try:
        with open(filepath, 'r') as f: return json.load(f)
    except FileNotFoundError:
        print(f"Error: The file '{filepath}' was not found.")
        return None
    except json.JSONDecodeError:
        print(f"Error: The file '{filepath}' is not a valid JSON file.")
        return None

def plot_final_clusters(clusters, medoids, k, method_name, output_dir=None):
    print("\nGenerating final visualization...")
    plt.figure(figsize=(12, 10))
    colors = plt.cm.viridis(np.linspace(0, 1, k))
    
    for i, (medoid_id, points) in enumerate(clusters.items()):
        if not points: continue
        cluster_coords = np.array([[p['x'], p['y']] for p in points])
        plt.scatter(cluster_coords[:, 0], cluster_coords[:, 1], c=[colors[i]], label=f'Cluster {i+1}', alpha=0.7, s=30)

    medoid_coords = np.array([[m['x'], m['y']] for m in medoids])
    plt.scatter(medoid_coords[:, 0], medoid_coords[:, 1], c='red', marker='X', s=400,
                edgecolors='white', linewidths=1.5, label='Medoids', zorder=10)

    plt.title(f'Final K-Medoid ({method_name.upper()}) Clustering Results', fontsize=16, fontweight='bold')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.gca().set_aspect('equal', adjustable='box')
    
    if output_dir:
        filepath = os.path.join(output_dir, 'final_result.png')
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f"Final plot saved to '{filepath}'")
    
    plt.show()


if __name__ == '__main__':
    file_path = 'smb-dataset.json'
    full_data = load_data_from_json(file_path)

    if full_data:
        print(f"--- Successfully loaded data from '{file_path}' ---")
        nodes_data = full_data.get('nodes', [])
        
        if not nodes_data:
            print("Error: No 'nodes' found in the JSON file. Cannot perform clustering.")
            sys.exit()

        print(f"  - Total Nodes: {len(nodes_data)}")
        print("--------------------------------------------------")

        # --- USER CONFIGURATION ---
        K_CLUSTERS = 3
        SHOW_LOGS_AND_SAVE_ITERATIONS = True
        
        # Choose the swap phase method: 'pam' or 'clara'
        SWAP_METHOD = 'clara'  # <-- CHANGE THIS to 'pam' to run the base algorithm

        # --- CLARA-specific parameters (only used if SWAP_METHOD is 'clara') ---
        NUM_SAMPLES = 3
        # If SAMPLE_SIZE is None, a default of 40 + 2*k will be used.
        # Otherwise, specify a size, e.g., 100. It must be > k.
        SAMPLE_SIZE = 320 
        
        # --- Create a unique output folder for this run ---
        output_folder = f"run_{SWAP_METHOD}_{time.strftime('%Y%m%d-%H%M%S')}"
        os.makedirs(output_folder, exist_ok=True)
        print(f"Output will be saved in folder: '{output_folder}'")
        
        print(f"\nInitializing K-medoids with method='{SWAP_METHOD}' for k={K_CLUSTERS} clusters...")
        
        kmedoids_model = KMedoids(
            k=K_CLUSTERS, 
            max_iter=100, 
            swap_phase_method=SWAP_METHOD,
            num_samples=NUM_SAMPLES,
            sample_size=SAMPLE_SIZE,
            verbose=SHOW_LOGS_AND_SAVE_ITERATIONS
        )
        
        kmedoids_model.fit(nodes_data, output_dir=output_folder)

        final_medoids = kmedoids_model.medoids
        final_clusters = kmedoids_model.clusters

        print("\n--- Clustering Results ---")
        print("\nFinal Medoids (Cluster Centers):")
        for medoid in final_medoids:
            print(f"  - Node ID: {medoid['id']}, Coordinates: (x={medoid['x']}, y={medoid['y']})")

        plot_final_clusters(final_clusters, final_medoids, K_CLUSTERS, SWAP_METHOD, output_dir=output_folder)