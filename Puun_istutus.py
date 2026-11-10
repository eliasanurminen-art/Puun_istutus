from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional
import random
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from networkx.linalg.graphmatrix import adjacency_matrix
import scipy.sparse as sp


@dataclass
class TreeGraphData:
    n: int
    edges: List[Tuple[int, int]]

    # Attribuuttimatriisi: [pituus, paksuus, poikkipinta-ala, karheus]
    edge_features: np.ndarray   # shape (m, 4)

    # Yhdistetty rakenne
    edge_data: List[Tuple[Tuple[int, int], np.ndarray]]

    adjacency: np.ndarray
    incidence: np.ndarray
    degrees: np.ndarray
    edge_index_u: np.ndarray
    edge_index_v: np.ndarray


def _try_generate_tree_with_degree_bounds(n, deg_min, deg_max, rng):
    degrees = [0] * n
    edges = []
    in_tree = {0}

    for new_node in range(1, n):
        candidates = [v for v in in_tree if degrees[v] < deg_max]
        if not candidates:
            return None

        parent = rng.choice(candidates)
        edges.append((parent, new_node))
        degrees[parent] += 1
        degrees[new_node] += 1
        in_tree.add(new_node)

    if any(d < deg_min or d > deg_max for d in degrees):
        return None

    edges = [(u, v) if u < v else (v, u) for u, v in edges]
    edges.sort()
    return edges


def generate_random_tree(
    n_min=20,
    n_max=100,
    deg_min=1,
    deg_max=4,
    seed: Optional[int] = None,
    max_tries=5000,
) -> TreeGraphData:

    rng = random.Random(seed)
    n = rng.randint(n_min, n_max)#solmut

    edges = None
    for _ in range(max_tries):
        edges = _try_generate_tree_with_degree_bounds(n, deg_min, deg_max, rng)
        if edges is not None:
            break

    if edges is None:
        raise RuntimeError("Tree generation failed.")

    m = len(edges)#linkit

    # --- Attribuutit ---
    lengths = np.array(
        [rng.uniform(100.0, 5000.0) for _ in range(m)],
        dtype=np.float64
    )

    thicknesses = np.linspace(
        rng.uniform(0.8, 1.0),
        rng.uniform(0.5, 0.7),
        m
    )
    # Poikkipinta-ala (ympyrä)
    areas = np.pi * (thicknesses ** 2) / 4.0

    roughnesses = np.array(
        [rng.uniform(1e-5, 5e-4) for _ in range(m)],
        dtype=np.float64
    )

    # Yhdistetty attribuuttimatriisi
    edge_features = np.column_stack(
        (lengths, thicknesses, areas, roughnesses)
    )
    edge_indices = np.array(edges, dtype=np.float64)  # shape (m, 2)
    edge_features = np.column_stack((edge_features, edge_indices))

    # Yhdistetty edge_data
    edge_data = [
        (edges[i], edge_features[i])
        for i in range(m)
    ]


    # --- Adjacency ---
    adjacency = np.zeros((n, n), dtype=np.int8)
    for u, v in edges:
        adjacency[u, v] = 1
        adjacency[v, u] = 1

    edges = np.array(edges, dtype=np.int32)
    print(edges)

    # linkki-indeksit (rivit)
    row = np.repeat(np.arange(m), 2)

    # solmuindeksit (sarakkeet)
    col = edges.flatten()

    # arvot: alku = -1, loppu = +1
    data = np.tile([-1, 1], m)

    adjacency = np.zeros((n, n), dtype=np.int8)
    for u, v in edges:
        adjacency[u, v] = 1
        adjacency[v, u] = 1
    degrees = adjacency.sum(axis=1)
    edges_list = [(int(u), int(v)) for u, v in edges]

    # --- Incidence ---
    incidence = np.zeros((n, m), dtype=np.int8)
    edge_u = np.empty(m, dtype=np.int32)
    edge_v = np.empty(m, dtype=np.int32)

    for i, (u, v) in enumerate(edges):
        incidence[u, i] = 1
        incidence[v, i] = 1
        edge_u[i] = u
        edge_v[i] = v

    # ==========================================================
    # LISÄTÄÄN 5 SATUNNAISTA SYKLIÄ (MESHAUS)
    # ==========================================================

    extra_edges_needed = 1
    existing_edges_set = set((min(u, v), max(u, v)) for u, v in edges_list)

    new_edges = []
    new_features = []

    while len(new_edges) < extra_edges_needed:
        u = rng.randint(0, n - 1)
        v = rng.randint(0, n - 1)

        if u == v:
            continue

        edge = (min(u, v), max(u, v))

        if edge in existing_edges_set:
            continue

        existing_edges_set.add(edge)
        new_edges.append(edge)

    # --- Generoidaan attribuutit samalla tavalla kuin alkuperäiset ---

    m_new = len(new_edges)

    lengths_new = np.array(
        [rng.uniform(100.0, 5000.0) for _ in range(m_new)],
        dtype=np.float64
    )

    thicknesses_new = np.linspace(
        rng.uniform(0.8, 1.0),
        rng.uniform(0.5, 0.7),
        m_new
    )

    areas_new = np.pi * (thicknesses_new ** 2) / 4.0

    roughnesses_new = np.array(
        [rng.uniform(1e-5, 5e-4) for _ in range(m_new)],
        dtype=np.float64
    )

    edge_indices_new = np.array(new_edges, dtype=np.float64)

    edge_features_new = np.column_stack(
        (lengths_new, thicknesses_new, areas_new, roughnesses_new)
    )
    edge_features_new = np.column_stack((edge_features_new, edge_indices_new))

    # --- Yhdistetään vanhoihin ---

    edges_list.extend(new_edges)

    edge_features = np.vstack((edge_features, edge_features_new))

    edge_data.extend(
        [(new_edges[i], edge_features_new[i]) for i in range(m_new)]
    )

    # ==========================================================
    # PÄIVITETÄÄN MATRIISIT JA INDEKSOINTI
    # ==========================================================

    edges = np.array(edges_list, dtype=np.int32)
    m_total = len(edges)

    # --- Adjacency ---
    adjacency = np.zeros((n, n), dtype=np.int8)
    for u, v in edges:
        adjacency[u, v] = 1
        adjacency[v, u] = 1

    degrees = adjacency.sum(axis=1)

    # --- Incidence ---
    incidence = np.zeros((n, m_total), dtype=np.int8)
    edge_u = np.empty(m_total, dtype=np.int32)
    edge_v = np.empty(m_total, dtype=np.int32)

    for i, (u, v) in enumerate(edges):
        incidence[u, i] = 1
        incidence[v, i] = 1
        edge_u[i] = u
        edge_v[i] = v

    adjacency = sp.csr_matrix((data, (row, col)), shape=(m_total, n))
    print(adjacency.shape)
    np.savez(
        "network_data.npz",
        edges=edges,
        edge_features=edge_features,
        adjacency=adjacency.toarray())

    adjacency = np.zeros((n, n), dtype=np.int8)
    for u, v in edges:
        adjacency[u, v] = 1
        adjacency[v, u] = 1

    return TreeGraphData(
        n=n,
        edges=edges_list,
        edge_features=edge_features,
        edge_data=edge_data,
        adjacency=adjacency,
        incidence=incidence,
        degrees=degrees,
        edge_index_u=edge_u,
        edge_index_v=edge_v,
    )
def plot_tree_graph(data: TreeGraphData, root: int = 0):
    """
    Piirtää puu/verkko-graafin NetworkX:n avulla.
    Viimeiset 5 lisättyä linkkiä piirretään vihreinä.
    """

    G = nx.Graph()
    G.add_nodes_from(range(data.n))
    G.add_edges_from(data.edges)

    plt.figure(figsize=(10, 8))

    pos = nx.spring_layout(G, seed=42)

    node_colors = ["red" if node == root else "lightblue"
                   for node in G.nodes()]

    # ---------------------------------------
    # Erotellaan puun reunat ja lisätyt reunat
    # ---------------------------------------

    extra_edges_count = 5
    meshed_edges = data.edges[-extra_edges_count:]
    tree_edges = data.edges[:-extra_edges_count]

    # Piirretään ensin puun reunat
    nx.draw_networkx_edges(
        G,
        pos,
        edgelist=tree_edges,
        edge_color="gray",
        width=1.5
    )

    # Piirretään lisätyt sykli-reunat vihreällä
    nx.draw_networkx_edges(
        G,
        pos,
        edgelist=meshed_edges,
        edge_color="green",
        width=3
    )

    # Solmut päälle
    nx.draw_networkx_nodes(
        G,
        pos,
        node_color=node_colors,
        node_size=500
    )

    nx.draw_networkx_labels(
        G,
        pos,
        font_size=10
    )

    plt.title("Meshed verkko (5 sykliä vihreällä)")
    plt.tight_layout()
    plt.show()


def main():
    data = generate_random_tree(seed=42)
    plot_tree_graph(data)

    print("Solmuja:", data.n)
    print("Linkkejä:", len(data.edges))

    print("\nEnsimmäinen linkki:")
    print("Indeksi:", data.edge_data[0][0])
    print("Attribuutit [pituus, paksuus, ala, karheus]:")
    print(data.edge_data[0][1])

    print("\nKoko edge_features-matriisi muoto:", data.edge_features.shape)


if __name__ == "__main__":
    main()
