import networkx as nx
import geopandas as gpd
from shapely.geometry import Point, LineString
import numpy as np
import pennylane as qml


class ChiayiMicrogridMapper:
    """Geospatial projection of the IEEE 33-bus system onto Chiayi, Taiwan."""

    def __init__(self, base_lat=23.4800, base_lon=120.4400):
        self.base_lat = base_lat
        self.base_lon = base_lon
        self.graph = nx.Graph()
        self.bus_coords = {}

    def generate_topology(self):
        """Generates the COMPLETE IEEE 33-bus topology with simulated Chiayi coordinates."""
        # Standard IEEE 33-bus complete radial connections (32 branches)
        edges = [
            (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9), (9, 10),
            (10, 11), (11, 12), (12, 13), (13, 14), (14, 15), (15, 16), (16, 17), (17, 18),
            (2, 19), (19, 20), (20, 21), (21, 22),
            (3, 23), (23, 24), (24, 25),
            (6, 26), (26, 27), (27, 28), (28, 29), (29, 30), (30, 31), (31, 32), (32, 33)
        ]
        self.graph.add_edges_from(edges)

        # Spatial assignment: Node 1 at city center, others radiating outward
        np.random.seed(42)
        for node in self.graph.nodes():
            # Coastal nodes (e.g., >18) get coordinates pushed further West
            lon_offset = np.random.uniform(-0.15, 0) if node > 18 else np.random.uniform(-0.05, 0.05)
            lat_offset = np.random.uniform(-0.1, 0.1)

            self.bus_coords[node] = Point(self.base_lon + lon_offset, self.base_lat + lat_offset)
            self.graph.nodes[node]['geometry'] = self.bus_coords[node]
            # Initialize vulnerability base based on coastal proximity
            self.graph.nodes[node]['coastal_exposure'] = abs(lon_offset) * 10

        return gpd.GeoDataFrame(
            [{'bus': k, 'geometry': v} for k, v in self.bus_coords.items()],
            crs="EPSG:4326"
        )

    def extract_spatial_features(self, typhoon_trajectory: LineString):
        """Calculates distance from each bus to the predicted typhoon path."""
        features = {}
        for node, point in self.bus_coords.items():
            distance = point.distance(typhoon_trajectory)
            features[node] = {'distance_to_eye': distance}
        return features

    def simulate_typhoon_failures(self, failed_edges):
        """Removes predicted failed lines from the grid topology."""
        post_disaster_grid = self.graph.copy()
        post_disaster_grid.remove_edges_from(failed_edges)
        return post_disaster_grid


class QuantumWalkIslandingMapper:
    """Identifies microgrid islands post-typhoon using Continuous-Time Quantum Walks."""

    def __init__(self, post_disaster_grid: nx.Graph):
        self.grid = post_disaster_grid
        self.n_nodes = len(post_disaster_grid.nodes)

        # Qubits needed to encode the nodes (e.g., 33 nodes -> 6 qubits)
        self.n_qubits = int(np.ceil(np.log2(max(self.grid.nodes()) + 1)))
        if self.n_qubits == 0: self.n_qubits = 1
        self.dev = qml.device('default.qubit', wires=self.n_qubits)

        self.node_to_idx = {node: int(node) for node in self.grid.nodes()}
        self.idx_to_node = {int(node): node for node in self.grid.nodes()}

    def _get_graph_hamiltonian(self):
        """Converts the graph Adjacency matrix into a Quantum Hamiltonian."""
        A = nx.adjacency_matrix(self.grid).todense()
        dim = 2 ** self.n_qubits
        H_matrix = np.zeros((dim, dim))

        # Map node connections to the Hamiltonian
        node_list = list(self.grid.nodes())
        for i in range(len(node_list)):
            for j in range(len(node_list)):
                idx_i = self.node_to_idx[node_list[i]]
                idx_j = self.node_to_idx[node_list[j]]
                H_matrix[idx_i, idx_j] = A[i, j]

        return qml.Hermitian(H_matrix, wires=range(self.n_qubits))

    def simulate_ctqw(self, start_node, time_t=5.0):
        """Evolves the quantum walk starting from a specific node."""
        H = self._get_graph_hamiltonian()
        start_idx = self.node_to_idx[start_node]

        @qml.qnode(self.dev)
        def quantum_walk_circuit():
            qml.BasisState(np.array([int(x) for x in format(start_idx, f'0{self.n_qubits}b')]),
                           wires=range(self.n_qubits))
            qml.ApproxTimeEvolution(H, time_t, n_steps=10)
            return qml.probs(wires=range(self.n_qubits))

        return quantum_walk_circuit()

    def identify_islands(self, threshold=1e-5):
        """Reconstructs the islanded microgrids by analyzing wave function spread."""
        unvisited = set(self.grid.nodes())
        islands = []

        while unvisited:
            start_node = unvisited.pop()
            probs = self.simulate_ctqw(start_node, time_t=10.0)

            island_nodes = set()
            for idx, prob in enumerate(probs):
                if prob > threshold and idx in self.idx_to_node and self.idx_to_node[idx] in self.grid.nodes():
                    island_nodes.add(self.idx_to_node[idx])

            island_nodes.add(start_node)
            islands.append(island_nodes)
            unvisited = unvisited - island_nodes

        return self._format_zones(islands)

    def _format_zones(self, islands):
        microgrid_zones = {}
        for idx, island_nodes in enumerate(islands):
            is_main_grid = 1 in island_nodes
            zone_id = "Main_Grid" if is_main_grid else f"Quantum_Island_{idx}"
            microgrid_zones[zone_id] = {
                'nodes': list(island_nodes),
                'size': len(island_nodes),
                'sub_graph': self.grid.subgraph(island_nodes).copy()
            }
        return microgrid_zones