import networkx as nx
import geopandas as gpd
from shapely.geometry import Point, LineString
import numpy as np
import pennylane as qml
import torch


class ChiayiMicrogridMapper:
    """Geospatial projection of the IEEE 33-bus system spread across Chiayi County."""

    def __init__(self):
        # Anchor the main substation (Bus 1) inland near Chiayi City
        self.base_lat = 23.4800
        self.base_lon = 120.4400
        # Coastline reference for vulnerability scoring
        self.coastline_lon = 120.1500

        self.graph = nx.Graph()
        self.bus_coords = {}

    def generate_topology(self):
        """Generates the IEEE 33-bus topology with a realistic geographic spread."""

        # Table IV: Branches Data (From, To, R(Ω), X(Ω), Type)
        branch_data = [
            (1, 2, 0.0922, 0.0470, 'Fixed'), (2, 3, 0.4930, 0.2511, 'Fixed'),
            (3, 4, 0.3660, 0.1864, 'Fixed'), (4, 5, 0.3811, 0.1941, 'Fixed'),
            (5, 6, 0.8190, 0.7070, 'Fixed'), (6, 7, 0.1872, 0.6188, 'Fixed'),
            (7, 8, 0.7114, 0.2351, 'Fixed'), (8, 9, 1.0300, 0.7400, 'Fixed'),
            (9, 10, 1.0440, 0.7400, 'Fixed'), (10, 11, 0.1966, 0.0650, 'Fixed'),
            (11, 12, 0.3744, 0.1238, 'Fixed'), (12, 13, 1.4680, 1.1550, 'Fixed'),
            (13, 14, 0.5416, 0.7129, 'Fixed'), (14, 15, 0.5910, 0.5260, 'Fixed'),
            (15, 16, 0.7463, 0.5450, 'Fixed'), (16, 17, 1.2890, 1.7210, 'Fixed'),
            (17, 18, 0.7320, 0.5740, 'Fixed'), (2, 19, 0.1640, 0.1565, 'Fixed'),
            (19, 20, 1.5042, 1.3554, 'Fixed'), (20, 21, 0.4095, 0.4784, 'Fixed'),
            (21, 22, 0.7089, 0.9373, 'Fixed'), (3, 23, 0.4512, 0.3083, 'Fixed'),
            (23, 24, 0.8980, 0.7091, 'Fixed'), (24, 25, 0.8960, 0.7011, 'Fixed'),
            (6, 26, 0.2030, 0.1034, 'Fixed'), (26, 27, 0.2842, 0.1447, 'Fixed'),
            (27, 28, 1.0590, 0.9337, 'Fixed'), (28, 29, 0.8042, 0.7006, 'Fixed'),
            (29, 30, 0.5075, 0.2585, 'Fixed'), (30, 31, 0.9744, 0.9630, 'Fixed'),
            (31, 32, 0.3105, 0.3619, 'Fixed'), (32, 33, 0.3410, 0.5302, 'Fixed'),
            (21, 8, 2.000, 2.000, 'Switchable'), (12, 22, 2.000, 2.000, 'Switchable'),
            (25, 29, 0.500, 0.500, 'Switchable')
        ]

        for u, v, r, x, b_type in branch_data:
            self.graph.add_edge(u, v, R=r, X=x, type=b_type)

        anchors = {
            1: (self.base_lon, self.base_lat),
            18: (120.15, 23.45),
            22: (120.18, 23.38),
            25: (120.20, 23.55),
            33: (120.25, 23.40)
        }

        branches = {
            'main': list(range(1, 19)),
            'arm1': [2] + list(range(19, 23)),
            'arm2': [3] + list(range(23, 26)),
            'arm3': [6] + list(range(26, 34))
        }

        for arm_name, nodes in branches.items():
            start_node, end_node = nodes[0], nodes[-1]
            start_lon, start_lat = anchors.get(start_node, self.bus_coords.get(start_node))
            end_lon, end_lat = anchors[end_node]

            num_segments = len(nodes) - 1

            for i, node in enumerate(nodes):
                if node not in self.bus_coords:
                    fraction = i / num_segments
                    noise_lon = np.random.uniform(-0.001, 0.001)
                    noise_lat = np.random.uniform(-0.001, 0.001)

                    interp_lon = start_lon + (end_lon - start_lon) * fraction + noise_lon
                    interp_lat = start_lat + (end_lat - start_lat) * fraction + noise_lat

                    self.bus_coords[node] = (interp_lon, interp_lat)

        for node, (lon, lat) in anchors.items():
            self.bus_coords[node] = (lon, lat)

        for node in self.graph.nodes():
            lon, lat = self.bus_coords[node]
            self.graph.nodes[node]['geometry'] = Point(lon, lat)
            exposure = max(0, 1 - ((lon - self.coastline_lon) / (self.base_lon - self.coastline_lon)))
            self.graph.nodes[node]['coastal_exposure'] = exposure

        return gpd.GeoDataFrame(
            [{'bus': k, 'geometry': self.graph.nodes[k]['geometry']} for k in self.graph.nodes()],
            crs="EPSG:4326"
        )

    def extract_spatial_features(self, typhoon_trajectory: LineString):
        """Calculates distance from each bus to the predicted typhoon path in degrees."""
        features = {}
        for node, coords in self.bus_coords.items():
            point = Point(coords[0], coords[1])
            distance = point.distance(typhoon_trajectory)
            features[node] = {'distance_to_eye': distance}
        return features

    def export_edge_index(self):
        """
        Exports the graph topology into a PyTorch Geometric edge_index tensor.
        Required for Graph Neural Network (GATConv) message passing.
        """
        node_mapping = {node: i for i, node in enumerate(self.graph.nodes())}
        edges = [(node_mapping[u], node_mapping[v]) for u, v in self.graph.edges()]

        bidirectional_edges = edges + [(v, u) for u, v in edges]

        edge_index = torch.tensor(bidirectional_edges, dtype=torch.long).t().contiguous()
        return edge_index, node_mapping

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

        node_list = list(self.grid.nodes())
        for i in range(len(node_list)):
            for j in range(len(node_list)):
                idx_i = self.node_to_idx[node_list[i]]
                idx_j = self.node_to_idx[node_list[j]]
                H_matrix[idx_i, idx_j] = A[i, j]

        return qml.pauli_decompose(H_matrix, wire_order=list(range(self.n_qubits)))

    def simulate_ctqw(self, start_node, time_t=5.0):
        """Evolves the quantum walk starting from a specific node."""
        H = self._get_graph_hamiltonian()
        start_idx = self.node_to_idx[start_node]

        @qml.qnode(self.dev)
        def quantum_walk_circuit():
            qml.BasisState(np.array([int(x) for x in format(start_idx, f'0{self.n_qubits}b')]),
                           wires=range(self.n_qubits))
            qml.ApproxTimeEvolution(H, time_t, n=10)
            return qml.probs(wires=range(self.n_qubits))

        return quantum_walk_circuit()

    def identify_islands(self, threshold=1e-5):
        """Reconstructs the islanded microgrids by analyzing wave function spread."""

        # --- THE FIX: CATASTROPHIC COLLAPSE BYPASS ---
        # If 100% of the lines have failed, the Adjacency Matrix is purely zeros.
        # Bypass the Quantum Walk simulator to prevent the pauli_decompose array stack crash.
        if len(self.grid.edges) == 0:
            islands = [{n} for n in self.grid.nodes()]
            return self._format_zones(islands)

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