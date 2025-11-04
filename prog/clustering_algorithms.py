import numpy as np
import time
import networkx as nx
import sys
import os
import matplotlib.pyplot as plt

# --- Helper function to save plots without showing them (from test_kmedoids.py) ---
def save_cluster_plot(clusters, medoids, k, title, filepath):
    """
    Generates and saves a plot of the current cluster state to a file.
    """
    plt.figure(figsize=(12, 10))
    # Handle cases where k is less than the number of unique colors in a colormap
    num_colors = max(k, 2)
    colors = plt.cm.viridis(np.linspace(0, 1, num_colors))

    # Create a mapping from medoid ID to a color index
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
    # Consolidate legend to avoid duplicate labels
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.gca().set_aspect('equal', adjustable='box')
    
    plt.savefig(filepath, dpi=100, bbox_inches='tight')
    plt.close()


# --- Base Geometric Clustering Algorithms ---
class PAM:
    """
    PAM (Partitioning Around Medoids) using GEOMETRIC (Euclidean) distance.
    MODIFIED to include verbose logging, progress bars, and iteration plotting.
    
    NEW: Includes a secondary swap method from test_kmedoids.py, controllable
         via the `rand_swap` parameter.
    """
    def __init__(self, k=3, max_iterations=100, use_build=True, verbose=True, 
                 output_dir=None, plot_prefix='pam_iter', rand_swap=False, **kwargs):
        self.k, self.max_iterations, self.use_build = k, max_iterations, use_build
        self.medoids, self.labels_, self.iterations_ = None, None, 0
        self.final_cost_ = float('inf')

        # --- NEW: Attribute to switch between swap methods ---
        self.rand_swap = rand_swap

        # --- Attributes for enhanced logging ---
        self.verbose = verbose
        self.output_dir = output_dir
        self.plot_prefix = plot_prefix
        if self.output_dir:
            os.makedirs(self.output_dir, exist_ok=True)
            if self.verbose: 
                print(f"  (PAM) Iteration plots will be saved to '{self.output_dir}'")
        if self.verbose and self.rand_swap:
            print("  (PAM) Using test_kmedoids.py swapping method.")


    def _print_progress_bar(self, iteration, total, prefix='', suffix='', length=50, fill='█'):
        """
        Prints a terminal progress bar.
        """
        percent = ("{0:.1f}").format(100 * (iteration / float(total)))
        filled_length = int(length * iteration // total)
        bar = fill * filled_length + '-' * (length - filled_length)
        sys.stdout.write(f'\r{prefix} |{bar}| {percent}% {suffix}')
        sys.stdout.flush()

    def _create_cluster_dict(self, data, medoid_indices, assignments):
        """
        Helper to create the dictionary structure required for plotting.
        """
        medoid_data = [data[i] for i in medoid_indices]
        medoid_ids = [m['id'] for m in medoid_data]
        clusters = {mid: [] for mid in medoid_ids}
        
        for point_idx, assignment_idx in enumerate(assignments):
            assigned_medoid_id = medoid_ids[assignment_idx]
            clusters[assigned_medoid_id].append(data[point_idx])
            
        return clusters, medoid_data

    def build_phase(self, data):
        """
        MODIFIED: Selects initial medoids, correctly handling pre-defined centroids
        by prioritizing them but never selecting more than k.
        """
        n = len(data)
        medoid_indices = []
        if self.verbose: print("\n  (PAM) BUILD Phase on sample...")

        # --- START OF FIX ---
        # Step 1: Find all pre-defined centroids first.
        predefined_centroids = [i for i, node in enumerate(data) if node.get('is_centroid')]

        if predefined_centroids:
            num_predefined = len(predefined_centroids)
            if self.verbose:
                print(f"  (PAM) Found {num_predefined} pre-defined centroids in the data.")

            # Step 2: If there are more pre-defined centroids than k,
            # take the first k and warn the user.
            if num_predefined >= self.k:
                if self.verbose:
                    print(f"  (PAM) WARNING: {num_predefined} pre-defined centroids found, but k is {self.k}.")
                    print(f"           Using the first {self.k} pre-defined centroids as the initial medoids.")
                medoid_indices = predefined_centroids[:self.k]
                return np.array(medoid_indices)

            # Step 3: If there are fewer than k, use all of them as the starting seed.
            else:
                if self.verbose:
                    print(f"  (PAM) Using all {num_predefined} pre-defined centroids as the initial seed.")
                medoid_indices.extend(predefined_centroids)
        # --- END OF FIX ---


        # Find the first medoid if none were pre-defined
        if not medoid_indices:
            min_total_dist, first_medoid = float('inf'), 0
            for i in range(n):
                total_dist = np.sum([self.calculate_distance(data[i], data[j]) for j in range(n)])
                if total_dist < min_total_dist:
                    min_total_dist, first_medoid = total_dist, i
            medoid_indices.append(first_medoid)

        # Iteratively add the remaining medoids until we have k
        while len(medoid_indices) < self.k:
            max_gain, best_candidate = -float('inf'), -1
            
            candidate_nodes = [i for i in range(n) if i not in medoid_indices]
            if not candidate_nodes: # Stop if no more nodes are available
                break

            for i in candidate_nodes:
                gain = self.calculate_gain(data, medoid_indices, i)
                if gain > max_gain:
                    max_gain, best_candidate = gain, i
            
            if best_candidate != -1:
                medoid_indices.append(best_candidate)
            else:
                if self.verbose:
                    print("  (PAM) BUILD Phase: No candidate offered positive gain. Selecting a random node to continue.")
                medoid_indices.append(np.random.choice(candidate_nodes))

        return np.array(medoid_indices)


    def calculate_gain(self, data, current_medoids, candidate):
        total_gain = 0
        for j in range(len(data)):
            if j not in current_medoids and j != candidate:
                current_min_dist = min(self.calculate_distance(data[j], data[medoid_idx]) for medoid_idx in current_medoids)
                dist_to_candidate = self.calculate_distance(data[j], data[candidate])
                if dist_to_candidate < current_min_dist:
                    total_gain += (current_min_dist - dist_to_candidate)
        return total_gain

    def calculate_distance(self, point1, point2):
        if isinstance(point1, dict):
             p1 = np.array([point1['x'], point1['y']])
             p2 = np.array([point2['x'], point2['y']])
             return np.sqrt(np.sum((p1 - p2) ** 2))
        return np.sqrt(np.sum((point1 - point2) ** 2))

    def calculate_total_cost(self, data, medoid_indices, assignments):
        total_cost = 0
        medoid_points = [data[i] for i in medoid_indices]
        for i, point in enumerate(data):
            medoid_point = medoid_points[assignments[i]]
            total_cost += self.calculate_distance(point, medoid_point)
        return total_cost

    def assign_clusters(self, data, medoid_indices):
        assignments = []
        medoid_points = [data[i] for i in medoid_indices]
        for point in data:
            distances = [self.calculate_distance(point, medoid) for medoid in medoid_points]
            assignments.append(np.argmin(distances))
        return np.array(assignments)

    def calculate_swap_cost(self, data, medoid_indices, assignments, i, m):
        new_medoids = medoid_indices.copy()
        new_medoids[m] = i
        new_assignments = self.assign_clusters(data, new_medoids)
        old_cost = self.calculate_total_cost(data, medoid_indices, assignments)
        new_cost = self.calculate_total_cost(data, new_medoids, new_assignments)
        return new_cost - old_cost

    def fit(self, data):
        """
        MODIFIED: This method now includes verbose logging, progress bars,
        and saves plots of each iteration if configured. It can also switch
        between two different swapping implementations.
        """
        n = len(data)
        if self.k > n:
            raise ValueError(f"k ({self.k}) cannot be greater than the number of data points ({n}).")

        if n == self.k:
            if self.verbose:
                print("  (PAM) INFO: k equals the number of data points. Every point is a cluster.")
            self.medoids = np.arange(n)
            self.labels_ = np.arange(n)
            self.iterations_ = 0
            self.final_cost_ = 0
            return self

        medoid_indices = self.build_phase(data) if self.use_build else np.random.choice(n, self.k, replace=False)
        
        if self.verbose and self.output_dir:
            initial_assignments = self.assign_clusters(data, medoid_indices)
            clusters, medoid_data = self._create_cluster_dict(data, medoid_indices, initial_assignments)
            title = f'Initial State ({self.plot_prefix})'
            filepath = os.path.join(self.output_dir, f'{self.plot_prefix}_000_initial.png')
            save_cluster_plot(clusters, medoid_data, self.k, title, filepath)
        
        self.iterations_ = 0
        if self.verbose: print("  (PAM) SWAP Phase on sample...")
        
        # ========================================================================
        # START OF SWAPPING LOGIC
        # ========================================================================
        
        for i in range(self.max_iterations):
            self.iterations_ += 1
            iter_start_time = time.time()
            made_swap = False

            # --- BRANCH TO SELECT SWAPPING METHOD ---
            if self.rand_swap:
                # =================================================================
                # METHOD 1: test_kmedoids.py brute-force style
                # Finds the single best swap in the entire configuration space
                # and performs it at the end of the iteration.
                # =================================================================
                
                # Calculate the cost at the start of the iteration
                initial_assignments = self.assign_clusters(data, medoid_indices)
                best_cost_so_far = self.calculate_total_cost(data, medoid_indices, initial_assignments)
                
                potential_best_medoids = medoid_indices
                
                non_medoids = [p_idx for p_idx in range(n) if p_idx not in medoid_indices]
                total_swaps_to_check = len(medoid_indices) * len(non_medoids)
                swaps_checked = 0
                if self.verbose:
                    self._print_progress_bar(0, total_swaps_to_check, prefix=f'  - Iteration {self.iterations_} Progress:', length=40)

                for medoid_idx_in_list, current_medoid_p_idx in enumerate(medoid_indices):
                    for non_medoid_p_idx in non_medoids:
                        # Create a temporary set of medoids for evaluation
                        temp_medoids = list(medoid_indices)
                        temp_medoids[medoid_idx_in_list] = non_medoid_p_idx
                        
                        # Calculate the total cost for this new configuration
                        temp_assignments = self.assign_clusters(data, temp_medoids)
                        new_cost = self.calculate_total_cost(data, temp_medoids, temp_assignments)
                        
                        if new_cost < best_cost_so_far:
                            best_cost_so_far = new_cost
                            potential_best_medoids = temp_medoids
                            made_swap = True
                        
                        if self.verbose:
                            swaps_checked += 1
                            self._print_progress_bar(swaps_checked, total_swaps_to_check, prefix=f'  - Iteration {self.iterations_} Progress:', length=40)
                
                medoid_indices = np.array(potential_best_medoids)

            else:
                # =================================================================
                # METHOD 2: Original greedy style
                # For each medoid, finds the best swap and performs it immediately
                # before moving to the next medoid.
                # =================================================================
                assignments = self.assign_clusters(data, medoid_indices)
                non_medoids = [p for p in range(n) if p not in medoid_indices]
                total_swaps_to_check = len(medoid_indices) * len(non_medoids)
                swaps_checked = 0
                if self.verbose:
                    self._print_progress_bar(0, total_swaps_to_check, prefix=f'  - Iteration {self.iterations_} Progress:', length=40)
                
                for m in range(len(medoid_indices)):
                    if isinstance(data[medoid_indices[m]], dict) and data[medoid_indices[m]].get('is_centroid'):
                        swaps_checked += len(non_medoids) # Skip checks for predefined centroids
                        continue
                    
                    best_swap_cost, best_swap_candidate = 0, None
                    for p_idx in non_medoids:
                        swap_cost = self.calculate_swap_cost(data, medoid_indices, assignments, p_idx, m)
                        if swap_cost < best_swap_cost:
                            best_swap_cost, best_swap_candidate = swap_cost, p_idx
                        
                        if self.verbose:
                            swaps_checked += 1
                            self._print_progress_bar(swaps_checked, total_swaps_to_check, prefix=f'  - Iteration {self.iterations_} Progress:', length=40)

                    if best_swap_candidate is not None:
                        medoid_indices[m] = best_swap_candidate
                        made_swap = True
            
            # --- Iteration Summary and Plotting (common to both methods) ---
            iter_runtime = time.time() - iter_start_time
            current_cost = self.calculate_total_cost(data, medoid_indices, self.assign_clusters(data, medoid_indices))

            if self.verbose:
                summary_line = f"Iteration {self.iterations_}: Cost = {current_cost:.2f}, Swap: {made_swap}, Runtime: {iter_runtime:.4f}s"
                padded_summary = summary_line.ljust(80)
                sys.stdout.write('\r' + padded_summary + '\n')
                sys.stdout.flush()

                if self.output_dir:
                    current_assignments = self.assign_clusters(data, medoid_indices)
                    clusters, medoid_data = self._create_cluster_dict(data, medoid_indices, current_assignments)
                    title = f'Iteration {self.iterations_} ({self.plot_prefix})'
                    filepath = os.path.join(self.output_dir, f'{self.plot_prefix}_{self.iterations_:03d}.png')
                    save_cluster_plot(clusters, medoid_data, self.k, title, filepath)
            
            if not made_swap:
                if self.verbose: print("  (PAM) Converged.")
                break
        
        # ========================================================================
        # END OF SWAPPING LOGIC
        # ========================================================================

        self.medoids = medoid_indices
        self.labels_ = self.assign_clusters(data, medoid_indices)
        self.final_cost_ = self.calculate_total_cost(data, self.medoids, self.labels_)
        return self
    
class CLARA(PAM):
    """
    CLARA algorithm, which applies PAM on multiple samples of the data.
    """
    def __init__(self, k=3, num_samples=5, sample_size=3200, use_build=True, verbose=True, output_dir=None, **kwargs):
        if sample_size <= k: raise ValueError("Sample size must be greater than k.")
        self.num_samples, self.sample_size, self.use_build = num_samples, sample_size, use_build
        self.medoids_, self.labels_, self.best_cost_ = None, None, float('inf')
        # Pass relevant arguments to the parent PAM constructor
        super().__init__(k=k, use_build=use_build, verbose=verbose, output_dir=output_dir, **kwargs)

    def fit(self, data):
        n = len(data)
        best_medoid_indices_in_full_data = None
        init_method = "BUILD Phase" if self.use_build else "Random Initialization"
        if self.verbose:
            print("=" * 50)
            print(f"Running CLARA Algorithm with {init_method}")
            print(f"Total data points: {n}")
            print(f"Number of samples: {self.num_samples}, Sample size: {self.sample_size}")
            print("=" * 50)

        for i in range(self.num_samples):
            if self.verbose: print(f"\n--- Sample {i + 1} of {self.num_samples} ---")
            
            sample_indices = np.random.choice(n, self.sample_size, replace=False)
            sample_data = [data[idx] for idx in sample_indices]
            
            plot_prefix = f'clara_sample_{i+1:03d}_iter'
            
            # Apply PAM to the sample, passing all relevant settings
            pam_on_sample = PAM(k=self.k, use_build=self.use_build, verbose=self.verbose,
                                output_dir=self.output_dir, plot_prefix=plot_prefix,
                                rand_swap=self.rand_swap)
            pam_on_sample.fit(sample_data)
            medoid_indices_in_full_data = [sample_indices[m_idx] for m_idx in pam_on_sample.medoids]

            if self.verbose: print("  Assigning all data points to sample medoids...")
            assignments_on_full_data = self.assign_clusters(data, medoid_indices_in_full_data)
            if self.verbose: print("  Calculating total cost for the full dataset...")
            current_cost = self.calculate_total_cost(data, medoid_indices_in_full_data, assignments_on_full_data)
            if self.verbose: print(f"  Cost for this sample's medoids: {current_cost:.2f}")

            if current_cost < self.best_cost_:
                self.best_cost_ = current_cost
                best_medoid_indices_in_full_data = medoid_indices_in_full_data
                if self.verbose: print(f"  *** Found new best set of medoids! New best cost: {self.best_cost_:.2f} ***")

        if self.verbose:
            print("\n" + "=" * 50)
            print(f"CLARA ({init_method}) has finished.")
            print(f"Final best cost: {self.best_cost_:.2f}")
        
        self.medoids_ = best_medoid_indices_in_full_data
        if self.verbose: print("\nAssigning final labels to all data points...")
        self.labels_ = self.assign_clusters(data, self.medoids_)
        return self

# --- Network-Aware Clustering Algorithms ---
class PAM_Network(PAM):
    def __init__(self, road_graph=None, node_positions=None, house_to_road_map=None, **kwargs):
        super().__init__(**kwargs)
        if road_graph is None or node_positions is None or house_to_road_map is None:
            raise ValueError("PAM_Network requires a road_graph, node_positions, and house_to_road_map.")
        self.G, self.pos, self.house_map = road_graph, node_positions, house_to_road_map
        if self.verbose:
            print("  (PAM_Network) Pre-calculating all-pairs shortest path...")
        self.shortest_paths = dict(nx.all_pairs_dijkstra_path_length(self.G, weight='weight'))
        if self.verbose: print("  (PAM_Network) Pre-calculation complete.")

    def calculate_distance(self, point1_house, point2_house):
        house1_id, house2_id = point1_house['id'], point2_house['id']
        road_entry1, road_entry2 = self.house_map[house1_id], self.house_map[house2_id]
        pos_house1, pos_house2 = (point1_house['x'], point1_house['y']), (point2_house['x'], point2_house['y'])
        dist_house1_to_road = super(PAM_Network, self).calculate_distance(np.array(pos_house1), np.array(self.pos[road_entry1]))
        dist_house2_to_road = super(PAM_Network, self).calculate_distance(np.array(pos_house2), np.array(self.pos[road_entry2]))
        on_road_dist = 0 if road_entry1 == road_entry2 else self.shortest_paths[road_entry1].get(road_entry2, float('inf'))
        return dist_house1_to_road + on_road_dist + dist_house2_to_road

class CLARA_Network(CLARA):
    """
    An enhanced version of CLARA that uses PAM_Network to perform clustering
    based on network distance.
    """
    def __init__(self, road_graph=None, node_positions=None, house_to_road_map=None, **kwargs):
        super().__init__(**kwargs)
        self.G, self.pos, self.house_map = road_graph, node_positions, house_to_road_map
        self.pam_engine = PAM_Network(k=self.k, verbose=False, road_graph=self.G,
                                      node_positions=self.pos, house_to_road_map=self.house_map,
                                      rand_swap=self.rand_swap)

    def fit(self, data):
        n = len(data)
        data_dict = {node['id']: node for node in data}
        best_medoid_ids = None
        init_method = "BUILD Phase" if self.use_build else "Random Initialization"
        if self.verbose:
            print("=" * 50)
            print(f"Running CLARA_Network Algorithm with {init_method}")
            print(f"Total house nodes: {n}")
            print(f"Number of samples: {self.num_samples}, Sample size: {self.sample_size}")
            print("=" * 50)
            
        for i in range(self.num_samples):
            if self.verbose: print(f"\n--- Sample {i + 1} of {self.num_samples} ---")
            sample_indices = np.random.choice(n, self.sample_size, replace=False)
            sample_data = [data[i] for i in sample_indices]

            plot_prefix = f'clara_net_sample_{i+1:03d}_iter'

            pam_on_sample = PAM_Network(
                k=self.k, use_build=self.use_build, road_graph=self.G, 
                node_positions=self.pos, house_to_road_map=self.house_map,
                verbose=self.verbose, output_dir=self.output_dir, plot_prefix=plot_prefix,
                rand_swap=self.rand_swap
            )
            pam_on_sample.fit(sample_data)
            medoid_ids_from_sample = [sample_data[m_idx]['id'] for m_idx in pam_on_sample.medoids]
            
            if self.verbose: print("  Assigning all data points to sample medoids (using network distance)...")
            assignments = [np.argmin([self.pam_engine.calculate_distance(house, data_dict[med_id]) for med_id in medoid_ids_from_sample]) for house in data]

            if self.verbose: print("  Calculating total cost for the full dataset...")
            current_cost = sum(self.pam_engine.calculate_distance(house, data_dict[medoid_ids_from_sample[assignments[j]]]) for j, house in enumerate(data))
            
            if np.isinf(current_cost):
                if self.verbose: print(f"  Warning: Cost for this sample is infinite. Discarding sample.")
                continue
            if self.verbose: print(f"  Cost for this sample's medoids: {current_cost:.2f}")

            if current_cost < self.best_cost_:
                self.best_cost_ = current_cost
                best_medoid_ids = medoid_ids_from_sample
                if self.verbose: print(f"  *** Found new best set of medoids! New best cost: {self.best_cost_:.2f} ***")
        
        if self.verbose:
            print("\n" + "=" * 50)
            if best_medoid_ids is None:
                print("FATAL ERROR: Could not find a valid set of medoids.")
                self.labels_ = np.array([])
                return self
            print(f"CLARA_Network ({init_method}) has finished.")
            print(f"Final best cost: {self.best_cost_:.2f}")
        
        self.medoids_ = best_medoid_ids
        if self.verbose: print("\nAssigning final labels to all data points...")
        self.labels_ = np.array([np.argmin([self.pam_engine.calculate_distance(house, data_dict[med_id]) for med_id in self.medoids_]) for house in data])
        return self

# --- Risk-Aware Clustering Algorithms (REVISED SECTION) ---

class PAM_RiskAware(PAM):
    """
    Extends PAM to incorporate a risk score for each data point.
    """
    def __init__(self, risk_scores=None, fire_risk_weight=1.0, **kwargs):
        super().__init__(**kwargs)
        self.risk_scores = risk_scores if risk_scores is not None else {}
        self.fire_risk_weight = fire_risk_weight
        if self.verbose:
            print(f"  (PAM_RiskAware) Initialized with fire_risk_weight: {self.fire_risk_weight}")

    def get_risk_factor(self, point_data):
        point_id = point_data.get('id', None)
        risk_score = self.risk_scores.get(point_id, 0)
        return 1 + (self.fire_risk_weight * risk_score)

    def calculate_total_cost(self, data, medoid_indices, assignments):
        total_cost = 0
        medoid_points = [data[i] for i in medoid_indices]
        for i, point in enumerate(data):
            risk_factor = self.get_risk_factor(point)
            medoid_point = medoid_points[assignments[i]]
            distance = self.calculate_distance(point, medoid_point)
            total_cost += risk_factor * distance
        return total_cost

    def calculate_gain(self, data, current_medoids, candidate):
        total_gain = 0
        for j in range(len(data)):
            if j not in current_medoids and j != candidate:
                risk_factor = self.get_risk_factor(data[j])
                current_min_dist = min(self.calculate_distance(data[j], data[medoid_idx]) for medoid_idx in current_medoids)
                dist_to_candidate = self.calculate_distance(data[j], data[candidate])
                improvement = current_min_dist - dist_to_candidate
                if improvement > 0:
                    total_gain += risk_factor * improvement
        return total_gain


class PAM_Network_RiskAware(PAM_RiskAware, PAM_Network):
    """
    Combines Network-based distance with Risk-Aware cost weighting.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class CLARA_Network_RiskAware(CLARA_Network):
    """
    The ultimate risk-aware, network-aware version of CLARA.
    """
    def __init__(self, risk_scores=None, fire_risk_weight=1.0, **kwargs):
        super().__init__(**kwargs)
        self.risk_scores = risk_scores
        self.fire_risk_weight = fire_risk_weight
        # Override the pam_engine with the correctly initialized risk-aware version
        self.pam_engine = PAM_Network_RiskAware(
            k=self.k, verbose=False, road_graph=self.G,
            node_positions=self.pos, house_to_road_map=self.house_map,
            risk_scores=self.risk_scores, fire_risk_weight=self.fire_risk_weight,
            rand_swap=self.rand_swap
        )

    def fit(self, data):
        """
        OVERRIDE: This method is a copy of CLARA_Network.fit but is modified to:
        1. Instantiate the correct `PAM_Network_RiskAware` class for each sample.
        2. Calculate the total cost using the risk-aware weighting.
        """
        n = len(data)
        data_dict = {node['id']: node for node in data}
        best_medoid_ids = None
        init_method = "BUILD Phase" if self.use_build else "Random Initialization"
        if self.verbose:
            print("=" * 50)
            print(f"Running CLARA_Network_RiskAware Algorithm with {init_method}")
            print(f"Total house nodes: {n}")
            print(f"Number of samples: {self.num_samples}, Sample size: {self.sample_size}")
            print("=" * 50)
            
        for i in range(self.num_samples):
            if self.verbose: print(f"\n--- Sample {i + 1} of {self.num_samples} ---")
            sample_indices = np.random.choice(n, self.sample_size, replace=False)
            sample_data = [data[i] for i in sample_indices]
            
            plot_prefix = f'clara_net_risk_sample_{i+1:03d}_iter'
            
            # KEY CHANGE 1: Instantiate the correct risk-aware PAM engine for the sample
            pam_on_sample = PAM_Network_RiskAware(
                k=self.k, use_build=self.use_build, road_graph=self.G, 
                node_positions=self.pos, house_to_road_map=self.house_map,
                risk_scores=self.risk_scores, fire_risk_weight=self.fire_risk_weight,
                verbose=self.verbose, output_dir=self.output_dir, plot_prefix=plot_prefix,
                rand_swap=self.rand_swap
            )
            
            pam_on_sample.fit(sample_data)
            medoid_ids_from_sample = [sample_data[m_idx]['id'] for m_idx in pam_on_sample.medoids]
            
            if self.verbose: print("  Assigning all data points to sample medoids (using network distance)...")
            assignments = [np.argmin([self.pam_engine.calculate_distance(house, data_dict[med_id]) for med_id in medoid_ids_from_sample]) for house in data]

            if self.verbose: print("  Calculating total WEIGHTED cost for the full dataset...")
            current_cost = 0
            for j, house in enumerate(data):
                assigned_medoid_id = medoid_ids_from_sample[assignments[j]]
                # KEY CHANGE 2: Calculate cost using the risk factor
                risk_factor = self.pam_engine.get_risk_factor(house)
                distance = self.pam_engine.calculate_distance(house, data_dict[assigned_medoid_id])
                current_cost += risk_factor * distance

            if np.isinf(current_cost):
                if self.verbose: print(f"  Warning: Cost for this sample is infinite. Discarding sample.")
                continue
            if self.verbose: print(f"  Weighted Cost for this sample's medoids: {current_cost:.2f}")
            if current_cost < self.best_cost_:
                self.best_cost_ = current_cost
                best_medoid_ids = medoid_ids_from_sample
                if self.verbose: print(f"  *** Found new best set of medoids! New best weighted cost: {self.best_cost_:.2f} ***")
        
        if self.verbose:
            print("\n" + "=" * 50)
            if best_medoid_ids is None:
                print("FATAL ERROR: Could not find a valid set of medoids.")
                self.labels_ = np.array([])
                return self
            self.medoids_ = best_medoid_ids
            print(f"CLARA_Network ({init_method}) has finished.")
            print(f"Final best weighted cost: {self.best_cost_:.2f}")
        
        if self.verbose: print("\nAssigning final labels to all data points...")
        final_assignments = [np.argmin([self.pam_engine.calculate_distance(house, data_dict[med_id]) for med_id in self.medoids_]) for house in data]
        self.labels_ = np.array(final_assignments)
        return self