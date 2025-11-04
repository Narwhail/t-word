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

    medoid_to_color_idx = {m['id']: i for i, m in enumerate(medoids)}

    for medoid_id, points in clusters.items():
        if not points: continue
        
        color_idx = medoid_to_color_idx.get(medoid_id)
        if color_idx is None: continue

        cluster_coords = np.array([[p['x'], p['y']] for p in points])
        plt.scatter(cluster_coords[:, 0], cluster_coords[:, 1],
                    c=[colors[color_idx]], label=f'Cluster {color_idx+1}', alpha=0.7, s=30)

    medoid_coords = np.array([[m['x'], m['y']] for m in medoids])
    plt.scatter(medoid_coords[:, 0], medoid_coords[:, 1], c='red', marker='X', s=400,
                edgecolors='white', linewidths=1.5, label='Medoids', zorder=10)

    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.gca().set_aspect('equal', adjustable='box')
    
    plt.savefig(filepath, dpi=100, bbox_inches='tight')
    plt.close()


class KMedoids:
    """
    A class to perform K-medoid clustering using either the
    Partitioning Around Medoids (PAM) or CLARA algorithm for the swap phase.
    """
    def __init__(self, k=2, max_iter=100, swap_phase_method='pam', 
                 num_samples=5, sample_size=None, clara_plot_scope='sample', verbose=False):
        if k < 1: raise ValueError("The number of clusters (k) must be greater than 0.")
        if swap_phase_method not in ['pam', 'clara']:
            raise ValueError("swap_phase_method must be either 'pam' or 'clara'.")
        if clara_plot_scope not in ['sample', 'full']:
            raise ValueError("clara_plot_scope must be either 'sample' or 'full'.")

        self.k = k
        self.max_iter = max_iter
        self.swap_phase_method = swap_phase_method
        self.verbose = verbose
        
        self.num_samples = num_samples
        self.sample_size = sample_size
        self.clara_plot_scope = clara_plot_scope

        if self.swap_phase_method == 'clara':
            if self.sample_size is None:
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
        if not medoids: return float('inf')
        total_cost = 0
        for point in data:
            cost = min(self._euclidean_distance(point, m) for m in medoids)
            total_cost += cost
        return total_cost

    def _assign_points_to_clusters(self, data, medoids):
        if not medoids: return {}
        clusters = {medoid['id']: [] for medoid in medoids}
        for point in data:
            closest_medoid = min(medoids, key=lambda medoid: self._euclidean_distance(point, medoid))
            clusters[closest_medoid['id']].append(point)
        return clusters
    
    # In the KMedoids class:

    def _print_progress_bar(self, iteration, total, prefix='', suffix='', length=50, fill='█'):
        """
        Call in a loop to create a terminal progress bar.
        """
        percent = ("{0:.1f}").format(100 * (iteration / float(total)))
        filled_length = int(length * iteration // total)
        bar = fill * filled_length + '-' * (length - filled_length)
        # Use carriage return '\r' to stay on the same line.
        sys.stdout.write(f'\r{prefix} |{bar}| {percent}% {suffix}')
        sys.stdout.flush()

    def _run_pam_swap_phase(self, data_subset, initial_medoids=None, enable_iter_plotting=False, 
                            output_dir=None, plot_prefix='iteration', plotting_data=None):
        """
        Runs the iterative PAM swap phase on a given subset of data.
        """
        current_medoids = initial_medoids if initial_medoids else random.sample(data_subset, self.k)
        data_to_plot = plotting_data if plotting_data is not None else data_subset

        if self.verbose and enable_iter_plotting and output_dir:
            if plot_prefix.startswith('clara'):
                parts = plot_prefix.split('_')
                title = f'Initial State (Sample {int(parts[2])})'
            else:
                title = 'Initial State (PAM)'
            
            initial_clusters = self._assign_points_to_clusters(data_to_plot, current_medoids)
            filepath = os.path.join(output_dir, f'{plot_prefix}_000_initial.png')
            save_cluster_plot(initial_clusters, current_medoids, self.k, title, filepath)

        for i in range(self.max_iter):
            iter_start_time = time.time()
            best_cost_for_subset = self._calculate_total_cost(data_subset, current_medoids)
            potential_best_medoids = current_medoids[:]
            made_swap = False

            # --- Progress Bar Integration ---
            non_medoids = [p for p in data_subset if p not in current_medoids]
            total_swaps_to_check = self.k * len(non_medoids)
            swaps_checked = 0
            if self.verbose:
                # Initial display of the progress bar
                self._print_progress_bar(0, total_swaps_to_check, prefix=f'  - Iteration {i+1} Progress:', length=40)
            # --- End Integration ---

            for medoid_idx, medoid in enumerate(current_medoids):
                for non_medoid_idx, potential_new_medoid in enumerate(non_medoids):
                    temp_medoids = [m for m in current_medoids if m != medoid] + [potential_new_medoid]
                    new_cost = self._calculate_total_cost(data_subset, temp_medoids)
                    
                    if new_cost < best_cost_for_subset:
                        best_cost_for_subset = new_cost
                        potential_best_medoids = temp_medoids
                        made_swap = True
                    
                    # --- Progress Bar Update ---
                    if self.verbose:
                        swaps_checked += 1
                        self._print_progress_bar(swaps_checked, total_swaps_to_check, prefix=f'  - Iteration {i+1} Progress:', length=40)
                    # --- End Update ---

            current_medoids = potential_best_medoids
            iter_runtime = time.time() - iter_start_time

            if self.verbose:
                # Create the summary string
                summary_line = f"  - Iteration {i + 1}: Cost = {best_cost_for_subset:.2f}, Swap: {made_swap}, Runtime: {iter_runtime:.4f}s"
                
                # Pad the string with spaces on the right to ensure it covers the entire progress bar line
                # 80 characters is a safe width for most terminals.
                padded_summary = summary_line.ljust(80)

                # Write the padded summary to the same line as the progress bar, then add a newline
                sys.stdout.write('\r' + padded_summary + '\n')
                sys.stdout.flush()
                
                if enable_iter_plotting and output_dir:
                    if plot_prefix.startswith('clara'):
                        parts = plot_prefix.split('_')
                        title = f'Iteration {i+1} (Sample {int(parts[2])})'
                    else:
                        title = f'PAM Iteration {i+1}'
                    
                    clusters_to_plot = self._assign_points_to_clusters(data_to_plot, current_medoids)
                    filepath = os.path.join(output_dir, f'{plot_prefix}_{i+1:03d}.png')
                    save_cluster_plot(clusters_to_plot, current_medoids, self.k, title, filepath)

            if not made_swap:
                if self.verbose: print("  - PAM converged for this subset.")
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
            initial_medoids = random.sample(data, self.k)

            if self.verbose:
                initial_cost = self._calculate_total_cost(data, initial_medoids)
                print(f"Initial cost with random medoids: {initial_cost:.2f}")
            
            self.medoids = self._run_pam_swap_phase(data, initial_medoids, enable_iter_plotting=self.verbose, 
                                                    output_dir=output_dir, plot_prefix='pam_iter')

        elif self.swap_phase_method == 'clara':

            if self.verbose:
                print(f"\n--- Running CLARA Algorithm ---")
                print(f"Number of samples: {self.num_samples}, Sample size: {self.sample_size}")
                print(f"Plotting scope for iterations: '{self.clara_plot_scope}'")
            
            best_medoids_so_far = None
            min_overall_cost = float('inf')

            for i in range(self.num_samples):
                sample_data = random.sample(data, self.sample_size)
                if self.verbose: print(f"\nProcessing Sample {i+1}/{self.num_samples}...")
                
                data_for_plotting = data if self.clara_plot_scope == 'full' else sample_data
                
                sample_medoids = self._run_pam_swap_phase(
                    sample_data,
                    enable_iter_plotting=self.verbose,
                    output_dir=output_dir,
                    plot_prefix=f'clara_sample_{i+1:03d}_iter',
                    plotting_data=data_for_plotting
                )
                
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
    
    medoid_to_color_idx = {m['id']: i for i, m in enumerate(medoids)}

    for medoid_id, points in clusters.items():
        if not points: continue
        color_idx = medoid_to_color_idx.get(medoid_id)
        if color_idx is None: continue

        cluster_coords = np.array([[p['x'], p['y']] for p in points])
        plt.scatter(cluster_coords[:, 0], cluster_coords[:, 1], c=[colors[color_idx]], label=f'Cluster {color_idx+1}', alpha=0.7, s=30)

    medoid_coords = np.array([[m['x'], m['y']] for m in medoids])
    plt.scatter(medoid_coords[:, 0], medoid_coords[:, 1], c='red', marker='X', s=400,
                edgecolors='white', linewidths=1.5, label='Medoids', zorder=10)

    plt.title(f'Final K-Medoid ({method_name.upper()}) Clustering Results', fontsize=16, fontweight='bold')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.gca().set_aspect('equal', adjustable='box')
    
    if output_dir:
        filepath = os.path.join(output_dir, 'final_result.png')
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f"Final plot saved to '{filepath}'")
    
    plt.show()


if __name__ == '__main__':
    file_path = 'dataset.json'
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
        SHOW_LOGS_AND_SAVE_PLOTS = True
        
        # --- Set to 'pam' to test the fix ---
        SWAP_METHOD = 'clara'

        # --- CLARA-specific parameters (these can be left here, they won't affect PAM mode) ---
        NUM_SAMPLES = 3
        SAMPLE_SIZE = 4000
        CLARA_PLOT_SCOPE = 'sample'
        
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
            clara_plot_scope=CLARA_PLOT_SCOPE,
            verbose=SHOW_LOGS_AND_SAVE_PLOTS
        )
        
        kmedoids_model.fit(nodes_data, output_dir=output_folder)

        final_medoids = kmedoids_model.medoids
        final_clusters = kmedoids_model.clusters

        print("\n--- Clustering Results ---")
        if final_medoids:
            print("\nFinal Medoids (Cluster Centers):")
            for medoid in final_medoids:
                print(f"  - Node ID: {medoid['id']}, Coordinates: (x={medoid['x']}, y={medoid['y']})")
            plot_final_clusters(final_clusters, final_medoids, K_CLUSTERS, SWAP_METHOD, output_dir=output_folder)
        else:
            print("Clustering did not result in a set of medoids.")