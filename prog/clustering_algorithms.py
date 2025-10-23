import numpy as np
import time
import networkx as nx

# --- Base Geometric Clustering Algorithms ---

class PAM:
    """
    PAM (Partitioning Around Medoids) using GEOMETRIC (Euclidean) distance.
    """
    def __init__(self, k=3, max_iterations=100, use_build=True, verbose=True, **kwargs):
        # Accepts **kwargs to be compatible with super() in child classes
        self.k, self.max_iterations, self.use_build, self.verbose = k, max_iterations, use_build, verbose
        self.medoids, self.labels_, self.iterations_ = None, None, 0

    def build_phase(self, data):
        # ... (no changes in this method)
        n = len(data)
        medoid_indices = []
        if self.verbose: print("\n  (PAM) BUILD Phase on sample...")
        for i, node in enumerate(data):
            if isinstance(node, dict) and node.get('is_centroid'):
                medoid_indices.append(i)
        if not medoid_indices:
            min_total_dist, first_medoid = float('inf'), 0
            for i in range(n):
                total_dist = np.sum([self.calculate_distance(data[i], data[j]) for j in range(n)])
                if total_dist < min_total_dist:
                    min_total_dist, first_medoid = total_dist, i
            medoid_indices.append(first_medoid)
        while len(medoid_indices) < self.k:
            max_gain, best_candidate = -float('inf'), -1
            for i in range(n):
                if i not in medoid_indices:
                    gain = self.calculate_gain(data, medoid_indices, i)
                    if gain > max_gain:
                        max_gain, best_candidate = gain, i
            if best_candidate != -1:
                medoid_indices.append(best_candidate)
            else:
                break
        return np.array(medoid_indices)

    def calculate_gain(self, data, current_medoids, candidate):
        # ... (no changes in this method)
        total_gain = 0
        for j in range(len(data)):
            if j not in current_medoids and j != candidate:
                current_min_dist = min(self.calculate_distance(data[j], data[medoid_idx]) for medoid_idx in current_medoids)
                dist_to_candidate = self.calculate_distance(data[j], data[candidate])
                if dist_to_candidate < current_min_dist:
                    total_gain += (current_min_dist - dist_to_candidate)
        return total_gain

    def calculate_distance(self, point1, point2):
        # ... (no changes in this method)
        if isinstance(point1, dict):
             p1 = np.array([point1['x'], point1['y']])
             p2 = np.array([point2['x'], point2['y']])
             return np.sqrt(np.sum((p1 - p2) ** 2))
        return np.sqrt(np.sum((point1 - point2) ** 2))

    def calculate_total_cost(self, data, medoid_indices, assignments):
        # ... (no changes in this method)
        total_cost = 0
        medoid_points = [data[i] for i in medoid_indices]
        for i, point in enumerate(data):
            medoid_point = medoid_points[assignments[i]]
            total_cost += self.calculate_distance(point, medoid_point)
        return total_cost

    def assign_clusters(self, data, medoid_indices):
        # ... (no changes in this method)
        assignments = []
        medoid_points = [data[i] for i in medoid_indices]
        for point in data:
            distances = [self.calculate_distance(point, medoid) for medoid in medoid_points]
            assignments.append(np.argmin(distances))
        return np.array(assignments)

    def calculate_swap_cost(self, data, medoid_indices, assignments, i, m):
        # ... (no changes in this method)
        new_medoids = medoid_indices.copy()
        new_medoids[m] = i
        new_assignments = self.assign_clusters(data, new_medoids)
        old_cost = self.calculate_total_cost(data, medoid_indices, assignments)
        new_cost = self.calculate_total_cost(data, new_medoids, new_assignments)
        return new_cost - old_cost

    def fit(self, data):
        # ... (no changes in this method)
        n = len(data)
        medoid_indices = self.build_phase(data) if self.use_build else np.random.choice(n, self.k, replace=False)
        if len(medoid_indices) < self.k:
            remaining_indices = [i for i in range(n) if i not in medoid_indices]
            needed = self.k - len(medoid_indices)
            if len(remaining_indices) >= needed:
                medoid_indices = np.concatenate([medoid_indices, np.random.choice(remaining_indices, needed, replace=False)])
        changed, self.iterations_ = True, 0
        if self.verbose: print("  (PAM) SWAP Phase on sample...")
        while changed and self.iterations_ < self.max_iterations:
            changed, self.iterations_ = False, self.iterations_ + 1
            assignments = self.assign_clusters(data, medoid_indices)
            for m in range(self.k):
                if isinstance(data[medoid_indices[m]], dict) and data[medoid_indices[m]].get('is_centroid'):
                    continue
                best_swap_cost, best_swap_candidate = 0, None
                for i in range(n):
                    if i not in medoid_indices:
                        swap_cost = self.calculate_swap_cost(data, medoid_indices, assignments, i, m)
                        if swap_cost < best_swap_cost:
                            best_swap_cost, best_swap_candidate = swap_cost, i
                if best_swap_candidate is not None:
                    medoid_indices[m], changed = best_swap_candidate, True
        self.medoids = medoid_indices
        self.labels_ = self.assign_clusters(data, medoid_indices)
        return self

class CLARA(PAM):
    """
    CLARA algorithm, which applies PAM on multiple samples of the data to handle
    large datasets.
    """
    def __init__(self, k=3, num_samples=5, sample_size=3200, use_build=True):
        if sample_size <= k: raise ValueError("Sample size must be greater than k.")
        self.k, self.num_samples, self.sample_size, self.use_build = k, num_samples, sample_size, use_build
        self.medoids_, self.labels_, self.best_cost_ = None, None, float('inf')
        super().__init__(k=k, use_build=use_build)

    def fit(self, data):
        n = data.shape[0]
        best_medoid_indices_in_full_data = None
        init_method = "BUILD Phase" if self.use_build else "Random Initialization"
        print("=" * 50)
        print(f"Running CLARA Algorithm with {init_method}")
        print(f"Total data points: {n}")
        print(f"Number of samples: {self.num_samples}, Sample size: {self.sample_size}")
        print("=" * 50)

        for i in range(self.num_samples):
            print(f"\n--- Sample {i + 1} of {self.num_samples} ---")
            # 1. Draw a sample from the dataset
            sample_indices = np.random.choice(n, self.sample_size, replace=False)
            sample_data = data[sample_indices]
            
            # 2. Apply PAM to the sample to find medoids
            pam_on_sample = PAM(k=self.k, verbose=False, use_build=self.use_build)
            pam_on_sample.fit(sample_data)
            medoid_indices_in_full_data = sample_indices[pam_on_sample.medoids]

            # 3. Calculate the cost on the entire dataset
            print("  Assigning all data points to sample medoids...")
            assignments_on_full_data = self.assign_clusters(data, medoid_indices_in_full_data)
            print("  Calculating total cost for the full dataset...")
            current_cost = self.calculate_total_cost(data, medoid_indices_in_full_data, assignments_on_full_data)
            print(f"  Cost for this sample's medoids: {current_cost:.2f}")

            # 4. Keep track of the best set of medoids
            if current_cost < self.best_cost_:
                self.best_cost_ = current_cost
                best_medoid_indices_in_full_data = medoid_indices_in_full_data
                print(f"  *** Found new best set of medoids! New best cost: {self.best_cost_:.2f} ***")

        print("\n" + "=" * 50)
        print(f"CLARA ({init_method}) has finished.")
        print(f"Final best cost: {self.best_cost_:.2f}")
        self.medoids_ = best_medoid_indices_in_full_data
        print("\nAssigning final labels to all data points...")
        self.labels_ = self.assign_clusters(data, self.medoids_)
        return self

# --- Network-Aware Clustering Algorithms ---
class PAM_Network(PAM):
    def __init__(self, k=3, max_iterations=100, use_build=True, verbose=True, 
                 road_graph=None, node_positions=None, house_to_road_map=None, **kwargs):
        # MODIFIED: Pass **kwargs up the chain
        super().__init__(k=k, max_iterations=max_iterations, use_build=use_build, verbose=verbose, **kwargs)
        if road_graph is None or node_positions is None or house_to_road_map is None:
            raise ValueError("PAM_Network requires a road_graph, node_positions, and house_to_road_map.")
        self.G, self.pos, self.house_map = road_graph, node_positions, house_to_road_map
        if self.verbose:
            print("  (PAM_Network) Pre-calculating all-pairs shortest path...")
        self.shortest_paths = dict(nx.all_pairs_dijkstra_path_length(self.G, weight='weight'))
        if self.verbose: print("  (PAM_Network) Pre-calculation complete.")

    def calculate_distance(self, point1_house, point2_house):
        # ... (no changes in this method)
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
    def __init__(self, k=3, num_samples=5, sample_size=3200, use_build=True,
                 road_graph=None, node_positions=None, house_to_road_map=None):
        super().__init__(k, num_samples, sample_size, use_build)
        self.G, self.pos, self.house_map = road_graph, node_positions, house_to_road_map
        self.pam_engine = PAM_Network(k=self.k, verbose=False, road_graph=self.G,
                                      node_positions=self.pos, house_to_road_map=self.house_map)

    def fit(self, data):
        n = len(data)
        data_dict = {node['id']: node for node in data}
        best_medoid_ids = None
        init_method = "BUILD Phase" if self.use_build else "Random Initialization"
        print("=" * 50)
        print(f"Running CLARA_Network Algorithm with {init_method}")
        print(f"Total house nodes: {n}")
        print(f"Number of samples: {self.num_samples}, Sample size: {self.sample_size}")
        print("=" * 50)
        for i in range(self.num_samples):
            print(f"\n--- Sample {i + 1} of {self.num_samples} ---")
            # 1. Draw a sample from the dataset
            sample_indices = np.random.choice(n, self.sample_size, replace=False)
            sample_data = [data[i] for i in sample_indices]

            # 2. Apply PAM_Network to the sample
            pam_on_sample = PAM_Network(k=self.k, verbose=False, use_build=self.use_build,
                                        road_graph=self.G, node_positions=self.pos, house_to_road_map=self.house_map)
            pam_on_sample.fit(sample_data)
            medoid_ids_from_sample = [sample_data[m_idx]['id'] for m_idx in pam_on_sample.medoids]
            
            # 3. Assign all data points to the sample medoids and calculate cost
            print("  Assigning all data points to sample medoids (using network distance)...")
            assignments, last_update_time = [], time.time()
            for j, house in enumerate(data):
                distances = [self.pam_engine.calculate_distance(house, data_dict[med_id]) for med_id in medoid_ids_from_sample]
                assignments.append(np.argmin(distances))
                if time.time() - last_update_time > 5.0:
                    progress = (j + 1) / n * 100
                    print(f"    ...assignment progress: {progress:.1f}% ({j+1}/{n})", end='\r')
                    last_update_time = time.time()
            print(" " * 80, end='\r')

            print("  Calculating total cost for the full dataset...")
            current_cost, last_update_time = 0, time.time()
            for j, house in enumerate(data):
                assigned_medoid_id = medoid_ids_from_sample[assignments[j]]
                current_cost += self.pam_engine.calculate_distance(house, data_dict[assigned_medoid_id])
                if time.time() - last_update_time > 5.0:
                    progress = (j + 1) / n * 100
                    print(f"    ...cost calculation progress: {progress:.1f}% ({j+1}/{n})", end='\r')
                    last_update_time = time.time()
            print(" " * 80, end='\r')
            if np.isinf(current_cost):
                print(f"  Warning: Cost for this sample is infinite. Discarding sample.")
                continue
            print(f"  Cost for this sample's medoids: {current_cost:.2f}")

            # 4. Keep track of the best set of medoids
            if current_cost < self.best_cost_:
                self.best_cost_ = current_cost
                best_medoid_ids = medoid_ids_from_sample
                print(f"  *** Found new best set of medoids! New best cost: {self.best_cost_:.2f} ***")
        
        print("\n" + "=" * 50)
        if best_medoid_ids is None:
            print("FATAL ERROR: Could not find a valid set of medoids.")
            self.labels_ = np.array([])
            return self
        self.medoids_ = best_medoid_ids
        print(f"CLARA_Network ({init_method}) has finished.")
        print(f"Final best cost: {self.best_cost_:.2f}")
        
        # 5. Assign final labels based on the best medoids found
        print("\nAssigning final labels to all data points...")
        final_assignments, last_update_time = [], time.time()
        for j, house in enumerate(data):
            distances = [self.pam_engine.calculate_distance(house, data_dict[med_id]) for med_id in self.medoids_]
            final_assignments.append(np.argmin(distances))
            if time.time() - last_update_time > 5.0:
                progress = (j + 1) / n * 100
                print(f"    ...final assignment progress: {progress:.1f}% ({j+1}/{n})", end='\r')
                last_update_time = time.time()
        print(" " * 80, end='\r')
        self.labels_ = np.array(final_assignments)
        return self
    # --- Risk-Aware Clustering Algorithms (REVISED SECTION) ---

class PAM_RiskAware(PAM):
    """
    Extends PAM to incorporate a risk score for each data point.
    """
    def __init__(self, k=3, max_iterations=100, use_build=True, verbose=True,
                 risk_scores=None, fire_risk_weight=1.0, **kwargs):
        # MODIFIED: Pass **kwargs up the chain
        super().__init__(k=k, max_iterations=max_iterations, use_build=use_build, verbose=verbose, **kwargs)
        self.risk_scores = risk_scores if risk_scores is not None else {}
        self.fire_risk_weight = fire_risk_weight
        if self.verbose:
            print(f"  (PAM_RiskAware) Initialized with fire_risk_weight: {self.fire_risk_weight}")

    def get_risk_factor(self, point_data):
        # ... (no changes in this method)
        point_id = point_data.get('id', None)
        risk_score = self.risk_scores.get(point_id, 0)
        return 1 + (self.fire_risk_weight * risk_score)

    def calculate_total_cost(self, data, medoid_indices, assignments):
        # ... (no changes in this method)
        total_cost = 0
        medoid_points = [data[i] for i in medoid_indices]
        for i, point in enumerate(data):
            risk_factor = self.get_risk_factor(point)
            medoid_point = medoid_points[assignments[i]]
            distance = self.calculate_distance(point, medoid_point)
            total_cost += risk_factor * distance
        return total_cost

    def calculate_gain(self, data, current_medoids, candidate):
        # ... (no changes in this method)
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
    CORRECTED CLASS: Combines Network-based distance with Risk-Aware cost weighting.
    This class now properly inherits from both parents.
    """
    def __init__(self, **kwargs):
        # This simplified init uses super() to correctly call all parent __init__ methods
        # in the right order with all necessary arguments.
        super().__init__(**kwargs)


class CLARA_Network_RiskAware(CLARA_Network):
    """
    The ultimate risk-aware, network-aware version of CLARA.
    """
    def __init__(self, k=3, num_samples=5, sample_size=400, use_build=True,
                 road_graph=None, node_positions=None, house_to_road_map=None,
                 risk_scores=None, fire_risk_weight=1.0):
        super().__init__(k, num_samples, sample_size, use_build, road_graph, node_positions, house_to_road_map)
        self.risk_scores = risk_scores
        self.fire_risk_weight = fire_risk_weight
        # Override the pam_engine with the correctly initialized risk-aware version
        self.pam_engine = PAM_Network_RiskAware(
            k=self.k, verbose=False, road_graph=self.G,
            node_positions=self.pos, house_to_road_map=self.house_map,
            risk_scores=self.risk_scores, fire_risk_weight=self.fire_risk_weight
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
        print("=" * 50)
        print(f"Running CLARA_Network_RiskAware Algorithm with {init_method}")
        print(f"Total house nodes: {n}")
        print(f"Number of samples: {self.num_samples}, Sample size: {self.sample_size}")
        print("=" * 50)
        for i in range(self.num_samples):
            print(f"\n--- Sample {i + 1} of {self.num_samples} ---")
            sample_indices = np.random.choice(n, self.sample_size, replace=False)
            sample_data = [data[i] for i in sample_indices]
            
            # KEY CHANGE 1: Instantiate the correct risk-aware PAM engine for the sample
            pam_on_sample = PAM_Network_RiskAware(
                k=self.k, verbose=False, use_build=self.use_build,
                road_graph=self.G, node_positions=self.pos, house_to_road_map=self.house_map,
                risk_scores=self.risk_scores, fire_risk_weight=self.fire_risk_weight)
            
            pam_on_sample.fit(sample_data)
            medoid_ids_from_sample = [sample_data[m_idx]['id'] for m_idx in pam_on_sample.medoids]
            
            print("  Assigning all data points to sample medoids (using network distance)...")
            assignments, _ = [], time.time()
            for house in data:
                distances = [self.pam_engine.calculate_distance(house, data_dict[med_id]) for med_id in medoid_ids_from_sample]
                assignments.append(np.argmin(distances))

            print("  Calculating total WEIGHTED cost for the full dataset...")
            current_cost = 0
            for j, house in enumerate(data):
                assigned_medoid_id = medoid_ids_from_sample[assignments[j]]
                # KEY CHANGE 2: Calculate cost using the risk factor
                risk_factor = self.pam_engine.get_risk_factor(house)
                distance = self.pam_engine.calculate_distance(house, data_dict[assigned_medoid_id])
                current_cost += risk_factor * distance

            if np.isinf(current_cost):
                print(f"  Warning: Cost for this sample is infinite. Discarding sample.")
                continue
            print(f"  Weighted Cost for this sample's medoids: {current_cost:.2f}")
            if current_cost < self.best_cost_:
                self.best_cost_ = current_cost
                best_medoid_ids = medoid_ids_from_sample
                print(f"  *** Found new best set of medoids! New best weighted cost: {self.best_cost_:.2f} ***")
        
        # ... Final part of fit is the same as the original ...
        print("\n" + "=" * 50)
        if best_medoid_ids is None:
            print("FATAL ERROR: Could not find a valid set of medoids.")
            self.labels_ = np.array([])
            return self
        self.medoids_ = best_medoid_ids
        print(f"CLARA_Network ({init_method}) has finished.")
        print(f"Final best weighted cost: {self.best_cost_:.2f}")
        print("\nAssigning final labels to all data points...")
        final_assignments, _ = [], time.time()
        for j, house in enumerate(data):
            distances = [self.pam_engine.calculate_distance(house, data_dict[med_id]) for med_id in self.medoids_]
            final_assignments.append(np.argmin(distances))
        self.labels_ = np.array(final_assignments)
        return self