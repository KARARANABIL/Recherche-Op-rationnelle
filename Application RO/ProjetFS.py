





import tkinter as tk
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import random
from collections import deque
import string
import pandas as pd
import numpy as np
from tabulate import tabulate

# Constantes pour le style
EMSI_GREEN = "#006838"
DARK_GRAY = "#333333"
WINDOW_BG = "#FFFFFF"

# Bouton moderne
class ModernButton(tk.Button):
    def __init__(self, master=None, **kwargs):
        super().__init__(master, **kwargs)
        self.configure(
            relief=tk.FLAT,
            bg=EMSI_GREEN,
            fg="white",
            font=("Helvetica", 11),
            cursor="hand2",
            pady=8
        )
        self.bind("<Enter>", self.on_enter)
        self.bind("<Leave>", self.on_leave)

    def on_enter(self, e):
        self['background'] = DARK_GRAY

    def on_leave(self, e):
        self['background'] = EMSI_GREEN

# Fonction pour créer une fenêtre moderne
def create_modern_window(title, geometry):
    window = tk.Toplevel()
    window.title(title)
    window.geometry(geometry)
    window.configure(bg=WINDOW_BG)
    return window

# Fonction pour afficher un graphe dans une nouvelle fenêtre
def show_graph(graph, title, path=None, mst_edges=None, bellman_ford_paths=None, colors=None, flow=None):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)

    pos = nx.spring_layout(graph)
    node_colors = [colors.get(node, 'lightblue') for node in graph.nodes()] if colors else 'lightblue'
    nx.draw(graph, pos, with_labels=True, node_color=node_colors, node_size=700, ax=ax, arrows=True)

    if path:
        path_edges = list(zip(path, path[1:]))
        nx.draw_networkx_edges(graph, pos, edgelist=path_edges, edge_color='red', width=2, arrows=True)

    if mst_edges:
        nx.draw_networkx_edges(graph, pos, edgelist=mst_edges, edge_color='green', width=2, arrows=True)

    if bellman_ford_paths:
        for path in bellman_ford_paths.values():
            path_edges = list(zip(path, path[1:]))
            nx.draw_networkx_edges(graph, pos, edgelist=path_edges, edge_color='blue', width=2, arrows=True)

    if flow:
        labels = {(u, v): f"{flow[u][v]}/{graph[u][v]['capacity']}" for u, v in graph.edges()}
    else:
        labels = nx.get_edge_attributes(graph, 'weight')

    nx.draw_networkx_edge_labels(graph, pos, edge_labels=labels, ax=ax)

    ax.set_title(title)

    # Intégrer le graphe dans une fenêtre Tkinter
    plot_window = tk.Toplevel()
    plot_window.title(title)
    canvas = FigureCanvasTkAgg(fig, master=plot_window)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

# Welsh-Powell
def welsh_powell(graph):
    sorted_nodes = sorted(graph.degree, key=lambda x: x[1], reverse=True)
    sorted_nodes = [node for node, degree in sorted_nodes]
    colors = {}
    current_color = 0
    for node in sorted_nodes:
        if node not in colors:
            colors[node] = current_color
            for other_node in sorted_nodes:
                if other_node not in colors and not any(graph.has_edge(other_node, n) for n in colors if colors[n] == current_color):
                    colors[other_node] = current_color
            current_color += 1
    return colors

# Dijkstra
def dijkstra(graph, start, end):
    try:
        path = nx.dijkstra_path(graph, start, end, weight='weight')
        return path
    except nx.NetworkXNoPath:
        return None

# Kruskal
def kruskal(graph):
    edges = sorted(graph.edges(data=True), key=lambda x: x[2]['weight'])
    mst = nx.Graph()
    mst.add_nodes_from(graph.nodes())
    union_find = {node: node for node in graph.nodes()}

    def find(node):
        if union_find[node] != node:
            union_find[node] = find(union_find[node])
        return union_find[node]

    def union(node1, node2):
        root1, root2 = find(node1), find(node2)
        union_find[root1] = root2

    for edge in edges:
        node1, node2, data = edge
        if find(node1) != find(node2):
            mst.add_edge(node1, node2, weight=data['weight'])
            union(node1, node2)

    return mst

# Bellman-Ford
def bellman_ford(graph, source):
    try:
        shortest_paths = nx.single_source_bellman_ford_path(graph, source)
        shortest_distances = nx.single_source_bellman_ford_path_length(graph, source)
        return shortest_paths, shortest_distances
    except nx.NetworkXUnbounded:
        return None, None

# Potentiel-Métra
def generer_taches(nb_taches):
    taches = []
    for i in range(nb_taches):
        duree = random.randint(1, 10)
        jour_debut = random.randint(1, 30)
        taches.append({
            'Tache': f'Tâche {i+1}',
            'Durée': duree,
            'Jour Début': jour_debut
        })
    return taches

def appliquer_methode_potentiel(taches):
    taches_sorted = sorted(taches, key=lambda x: x['Jour Début'])
    for tache in taches_sorted:
        tache['Jour Fin'] = tache['Jour Début'] + tache['Durée'] - 1
    return taches_sorted

# Ford-Fulkerson
def ford_fulkerson(capacity, source, sink):
    n = len(capacity)
    flow = [[0] * n for _ in range(n)]
    max_flow = 0

    def bfs():
        parent = [-1] * n
        parent[source] = -2
        queue = deque([(source, float('inf'))])
        while queue:
            u, min_cap = queue.popleft()
            for v in range(n):
                if parent[v] == -1 and capacity[u][v] - flow[u][v] > 0:
                    parent[v] = u
                    new_flow = min(min_cap, capacity[u][v] - flow[u][v])
                    if v == sink:
                        return new_flow, parent
                    queue.append((v, new_flow))
        return 0, parent

    while True:
        path_flow, parent = bfs()
        if path_flow == 0:
            break
        max_flow += path_flow
        v = sink
        while v != source:
            u = parent[v]
            flow[u][v] += path_flow
            flow[v][u] -= path_flow
            v = u
    return max_flow, flow

# Stepping Stone
def stepping_stone(couts, allocation):
    rows, cols = allocation.shape
    couts = couts.astype(float)
    while True:
        empty_cells = [(i, j) for i in range(rows) for j in range(cols) if allocation[i, j] == 0]
        best_improvement = 0
        best_allocation = allocation.copy()

        for cell in empty_cells:
            cycle, gain = find_cycle_and_gain(couts, allocation, cell)
            if cycle and gain < best_improvement:
                best_improvement = gain
                best_allocation = adjust_allocation(allocation, cycle)

        if best_improvement >= 0:
            break
        allocation = best_allocation

    return allocation

def find_cycle_and_gain(couts, allocation, start_cell):
    rows, cols = allocation.shape
    visited = set()
    cycle = []

    def dfs(cell, path):
        if cell in visited:
            if cell == start_cell and len(path) >= 4:
                return path
            return None

        visited.add(cell)
        row, col = cell

        for next_cell in [(row, c) for c in range(cols)] + [(r, col) for r in range(rows)]:
            if next_cell != cell and allocation[next_cell] > 0 or next_cell == start_cell:
                new_path = dfs(next_cell, path + [cell])
                if new_path:
                    return new_path

        visited.remove(cell)
        return None

    cycle = dfs(start_cell, [])

    if not cycle:
        return None, 0

    gain = calculate_cycle_gain(couts, allocation, cycle)
    return cycle, gain

def calculate_cycle_gain(couts, allocation, cycle):
    gain = 0
    for k, (i, j) in enumerate(cycle):
        sign = 1 if k % 2 == 0 else -1
        gain += sign * couts[i, j]
    return gain

def adjust_allocation(allocation, cycle):
    min_alloc = min(allocation[i, j] for k, (i, j) in enumerate(cycle) if k % 2 == 1)

    for k, (i, j) in enumerate(cycle):
        sign = 1 if k % 2 == 0 else -1
        allocation[i, j] += sign * min_alloc

    return allocation

# Moindre Coût
def moindre_cout(couts, offre, demande):
    allocation = np.zeros_like(couts)
    large_value = 1e9  # Utiliser une valeur très grande au lieu de np.inf
    while np.sum(offre) > 0 and np.sum(demande) > 0:
        # Trouver la cellule avec le coût minimum
        i, j = np.unravel_index(np.argmin(couts), couts.shape)
        quantite = min(offre[i], demande[j])
        allocation[i, j] = quantite
        offre[i] -= quantite
        demande[j] -= quantite
        couts[i, j] = large_value  # Éliminer cette cellule pour les itérations suivantes
    return allocation

# Nord-Ouest
def nord_ouest(couts, offre, demande):
    allocation = np.zeros_like(couts)
    i, j = 0, 0
    while i < len(offre) and j < len(demande):
        quantite = min(offre[i], demande[j])
        allocation[i, j] = quantite
        offre[i] -= quantite
        demande[j] -= quantite
        if offre[i] == 0:
            i += 1
        else:
            j += 1
    return allocation

# Fonctions pour exécuter les algorithmes via l'interface graphique
def execute_welsh_powell_algorithm():
    window = create_modern_window("Welsh-Powell", "400x300")
    main_frame = tk.Frame(window, bg=WINDOW_BG, padx=20, pady=20)
    main_frame.pack(fill=tk.BOTH, expand=True)

    tk.Label(main_frame, text="Nombre de sommets :", bg=WINDOW_BG).pack(pady=5)
    vertices_entry = tk.Entry(main_frame)
    vertices_entry.pack(pady=5)

    tk.Label(main_frame, text="Probabilité (0-100) :", bg=WINDOW_BG).pack(pady=5)
    probability_entry = tk.Entry(main_frame)
    probability_entry.pack(pady=5)

    result_label = tk.Label(main_frame, text="", bg=WINDOW_BG)
    result_label.pack(pady=10)

    def run_algorithm():
        try:
            num_vertices = int(vertices_entry.get())
            probability = float(probability_entry.get()) / 100  # Convertir la probabilité en pourcentage
            graph = nx.erdos_renyi_graph(num_vertices, probability)
            for u, v in graph.edges():
                graph[u][v]['weight'] = random.randint(1, 10)
            colors = welsh_powell(graph)
            chromatic_number = len(set(colors.values()))
            result_label.config(text=f"Nombre chromatique : {chromatic_number}")
            show_graph(graph, "Welsh-Powell Graph", colors=colors)
        except Exception as e:
            result_label.config(text=f"Erreur : {str(e)}")

    ModernButton(main_frame, text="Exécuter", command=run_algorithm).pack(pady=15)

def execute_dijkstra_algorithm():
    window = create_modern_window("Dijkstra", "400x400")
    main_frame = tk.Frame(window, bg=WINDOW_BG, padx=20, pady=20)
    main_frame.pack(fill=tk.BOTH, expand=True)

    tk.Label(main_frame, text="Nombre de sommets :", bg=WINDOW_BG).pack(pady=5)
    vertices_entry = tk.Entry(main_frame)
    vertices_entry.pack(pady=5)

    tk.Label(main_frame, text="Probabilité (0-100) :", bg=WINDOW_BG).pack(pady=5)
    probability_entry = tk.Entry(main_frame)
    probability_entry.pack(pady=5)

    tk.Label(main_frame, text="Sommet de départ :", bg=WINDOW_BG).pack(pady=5)
    start_entry = tk.Entry(main_frame)
    start_entry.pack(pady=5)

    tk.Label(main_frame, text="Sommet d'arrivée :", bg=WINDOW_BG).pack(pady=5)
    end_entry = tk.Entry(main_frame)
    end_entry.pack(pady=5)

    result_label = tk.Label(main_frame, text="", bg=WINDOW_BG)
    result_label.pack(pady=10)

    def run_algorithm():
        try:
            num_vertices = int(vertices_entry.get())
            probability = float(probability_entry.get()) / 100  # Convertir la probabilité en pourcentage
            start = int(start_entry.get())
            end = int(end_entry.get())
            graph = nx.erdos_renyi_graph(num_vertices, probability)
            for u, v in graph.edges():
                graph[u][v]['weight'] = random.randint(1, 10)
            path = dijkstra(graph, start, end)
            if path:
                result_label.config(text=f"Chemin le plus court : {' -> '.join(map(str, path))}")
                show_graph(graph, "Dijkstra's Shortest Path", path)
            else:
                result_label.config(text="Pas de chemin trouvé.")
        except Exception as e:
            result_label.config(text=f"Erreur : {str(e)}")

    ModernButton(main_frame, text="Exécuter", command=run_algorithm).pack(pady=15)

def execute_kruskal_algorithm():
    window = create_modern_window("Kruskal", "400x300")
    main_frame = tk.Frame(window, bg=WINDOW_BG, padx=20, pady=20)
    main_frame.pack(fill=tk.BOTH, expand=True)

    tk.Label(main_frame, text="Nombre de sommets :", bg=WINDOW_BG).pack(pady=5)
    vertices_entry = tk.Entry(main_frame)
    vertices_entry.pack(pady=5)

    tk.Label(main_frame, text="Probabilité (0-100) :", bg=WINDOW_BG).pack(pady=5)
    probability_entry = tk.Entry(main_frame)
    probability_entry.pack(pady=5)

    result_label = tk.Label(main_frame, text="", bg=WINDOW_BG)
    result_label.pack(pady=10)

    def run_algorithm():
        try:
            num_vertices = int(vertices_entry.get())
            probability = float(probability_entry.get()) / 100  # Convertir la probabilité en pourcentage
            graph = nx.erdos_renyi_graph(num_vertices, probability)
            for u, v in graph.edges():
                graph[u][v]['weight'] = random.randint(1, 10)
            mst = kruskal(graph)
            total_weight = sum(mst[u][v]['weight'] for u, v in mst.edges())
            result_label.config(text=f"Poids total de l'arbre couvrant minimal : {total_weight}")
            show_graph(graph, "Kruskal MST", mst_edges=mst.edges())
        except Exception as e:
            result_label.config(text=f"Erreur : {str(e)}")

    ModernButton(main_frame, text="Exécuter", command=run_algorithm).pack(pady=15)

def execute_bellman_ford_algorithm():
    window = create_modern_window("Bellman-Ford", "400x400")
    main_frame = tk.Frame(window, bg=WINDOW_BG, padx=20, pady=20)
    main_frame.pack(fill=tk.BOTH, expand=True)

    tk.Label(main_frame, text="Nombre de sommets :", bg=WINDOW_BG).pack(pady=5)
    vertices_entry = tk.Entry(main_frame)
    vertices_entry.pack(pady=5)

    tk.Label(main_frame, text="Probabilité d'arête (0-100) :", bg=WINDOW_BG).pack(pady=5)
    probability_entry = tk.Entry(main_frame)
    probability_entry.pack(pady=5)

    tk.Label(main_frame, text="Sommet source :", bg=WINDOW_BG).pack(pady=5)
    source_entry = tk.Entry(main_frame)
    source_entry.pack(pady=5)

    result_label = tk.Label(main_frame, text="", bg=WINDOW_BG)
    result_label.pack(pady=10)

    def run_algorithm():
        try:
            num_vertices = int(vertices_entry.get())
            probability = float(probability_entry.get()) / 100  # Convertir la probabilité en pourcentage
            source = int(source_entry.get())
            graph = nx.erdos_renyi_graph(num_vertices, probability, directed=True)
            for u, v in graph.edges():
                graph[u][v]['weight'] = random.randint(1, 10)  # Ensure positive weights
            shortest_paths, shortest_distances = bellman_ford(graph, source)
            if shortest_paths is None:
                result_label.config(text="Le graphe contient un cycle de poids négatif.")
            else:
                result_text = f"Résultats de Bellman-Ford depuis le sommet {source}:\n"
                for target, path in shortest_paths.items():
                    distance = shortest_distances[target]
                    result_text += f"Vers {target}: {' -> '.join(map(str, path))} (distance: {distance})\n"
                result_label.config(text=result_text)
                show_graph(graph, "Bellman-Ford Graph", bellman_ford_paths=shortest_paths)
        except Exception as e:
            result_label.config(text=f"Erreur : {str(e)}")

    ModernButton(main_frame, text="Exécuter", command=run_algorithm).pack(pady=15)

def execute_potentiel_metra_algorithm():
    window = create_modern_window("Potentiel-Métra", "400x400")
    main_frame = tk.Frame(window, bg=WINDOW_BG, padx=20, pady=20)
    main_frame.pack(fill=tk.BOTH, expand=True)

    tk.Label(main_frame, text="Nombre de tâches :", bg=WINDOW_BG).pack(pady=5)
    tasks_entry = tk.Entry(main_frame)
    tasks_entry.pack(pady=5)

    result_label = tk.Label(main_frame, text="", bg=WINDOW_BG)
    result_label.pack(pady=10)

    def run_algorithm():
        try:
            nb_taches = int(tasks_entry.get())
            taches = generer_taches(nb_taches)
            taches_calculees = appliquer_methode_potentiel(taches)
            df = pd.DataFrame(taches_calculees)
            result_label.config(text=df.to_string(index=False))
        except Exception as e:
            result_label.config(text=f"Erreur : {str(e)}")

    ModernButton(main_frame, text="Exécuter", command=run_algorithm).pack(pady=15)

def execute_ford_fulkerson_algorithm():
    window = create_modern_window("Ford-Fulkerson", "400x400")
    main_frame = tk.Frame(window, bg=WINDOW_BG, padx=20, pady=20)
    main_frame.pack(fill=tk.BOTH, expand=True)

    tk.Label(main_frame, text="Nombre de sommets :", bg=WINDOW_BG).pack(pady=5)
    vertices_entry = tk.Entry(main_frame)
    vertices_entry.pack(pady=5)

    tk.Label(main_frame, text="Capacité maximale :", bg=WINDOW_BG).pack(pady=5)
    max_capacity_entry = tk.Entry(main_frame)
    max_capacity_entry.pack(pady=5)

    result_label = tk.Label(main_frame, text="", bg=WINDOW_BG)
    result_label.pack(pady=10)

    def run_algorithm():
        try:
            num_vertices = int(vertices_entry.get())
            max_capacity = int(max_capacity_entry.get())
            graph = nx.DiGraph()
            for i in range(num_vertices):
                for j in range(num_vertices):
                    if i != j:
                        capacity = random.randint(1, max_capacity)
                        graph.add_edge(i, j, capacity=capacity)
            capacity_matrix = nx.to_numpy_array(graph, weight='capacity', nonedge=0)
            source = 0
            sink = num_vertices - 1
            max_flow, flow = ford_fulkerson(capacity_matrix, source, sink)
            result_label.config(text=f"Flot maximal : {max_flow}")
            show_graph(graph, "Ford-Fulkerson Flow Network", flow=flow)
        except Exception as e:
            result_label.config(text=f"Erreur : {str(e)}")

    ModernButton(main_frame, text="Exécuter", command=run_algorithm).pack(pady=15)

def execute_stepping_stone_algorithm():
    window = create_modern_window("Stepping Stone", "400x400")
    main_frame = tk.Frame(window, bg=WINDOW_BG, padx=20, pady=20)
    main_frame.pack(fill=tk.BOTH, expand=True)

    tk.Label(main_frame, text="Nombre d'usines :", bg=WINDOW_BG).pack(pady=5)
    nb_usines_entry = tk.Entry(main_frame)
    nb_usines_entry.pack(pady=5)

    tk.Label(main_frame, text="Nombre de magasins :", bg=WINDOW_BG).pack(pady=5)
    nb_magasins_entry = tk.Entry(main_frame)
    nb_magasins_entry.pack(pady=5)

    result_label = tk.Label(main_frame, text="", bg=WINDOW_BG)
    result_label.pack(pady=10)

    def run_algorithm():
        try:
            nb_usines = int(nb_usines_entry.get())
            nb_magasins = int(nb_magasins_entry.get())
            couts = np.random.randint(1, 20, size=(nb_usines, nb_magasins))
            capacites = np.random.randint(10, 50, size=nb_usines)
            demandes = np.random.randint(10, 50, size=nb_magasins)
            allocation = stepping_stone(couts, np.zeros((nb_usines, nb_magasins)))
            result_label.config(text=f"Allocation optimisée :\n{allocation}")
        except Exception as e:
            result_label.config(text=f"Erreur : {str(e)}")

    ModernButton(main_frame, text="Exécuter", command=run_algorithm).pack(pady=15)

def execute_moindre_cout_algorithm():
    window = create_modern_window("Moindre Coût", "400x400")
    main_frame = tk.Frame(window, bg=WINDOW_BG, padx=20, pady=20)
    main_frame.pack(fill=tk.BOTH, expand=True)

    tk.Label(main_frame, text="Nombre d'usines :", bg=WINDOW_BG).pack(pady=5)
    nb_usines_entry = tk.Entry(main_frame)
    nb_usines_entry.pack(pady=5)

    tk.Label(main_frame, text="Nombre de magasins :", bg=WINDOW_BG).pack(pady=5)
    nb_magasins_entry = tk.Entry(main_frame)
    nb_magasins_entry.pack(pady=5)

    result_label = tk.Label(main_frame, text="", bg=WINDOW_BG)
    result_label.pack(pady=10)

    def run_algorithm():
        try:
            nb_usines = int(nb_usines_entry.get())
            nb_magasins = int(nb_magasins_entry.get())
            couts = np.random.randint(1, 20, size=(nb_usines, nb_magasins))
            offre = np.random.randint(10, 50, size=nb_usines)
            demande = np.random.randint(10, 50, size=nb_magasins)
            allocation = moindre_cout(couts.copy(), offre.copy(), demande.copy())

            # Afficher les coûts et l'allocation
            result_text = "Coûts entre les usines et les magasins :\n"
            result_text += tabulate(couts, tablefmt="grid") + "\n\n"
            result_text += "Allocation Moindre Coût :\n"
            result_text += tabulate(allocation, tablefmt="grid")
            result_label.config(text=result_text)
        except Exception as e:
            result_label.config(text=f"Erreur : {str(e)}")

    ModernButton(main_frame, text="Exécuter", command=run_algorithm).pack(pady=15)

def execute_nord_ouest_algorithm():
    window = create_modern_window("Nord-Ouest", "400x400")
    main_frame = tk.Frame(window, bg=WINDOW_BG, padx=20, pady=20)
    main_frame.pack(fill=tk.BOTH, expand=True)

    tk.Label(main_frame, text="Nombre d'usines :", bg=WINDOW_BG).pack(pady=5)
    nb_usines_entry = tk.Entry(main_frame)
    nb_usines_entry.pack(pady=5)

    tk.Label(main_frame, text="Nombre de magasins :", bg=WINDOW_BG).pack(pady=5)
    nb_magasins_entry = tk.Entry(main_frame)
    nb_magasins_entry.pack(pady=5)

    result_label = tk.Label(main_frame, text="", bg=WINDOW_BG)
    result_label.pack(pady=10)

    def run_algorithm():
        try:
            nb_usines = int(nb_usines_entry.get())
            nb_magasins = int(nb_magasins_entry.get())
            couts = np.random.randint(1, 20, size=(nb_usines, nb_magasins))
            offre = np.random.randint(10, 50, size=nb_usines)
            demande = np.random.randint(10, 50, size=nb_magasins)
            allocation = nord_ouest(couts.copy(), offre.copy(), demande.copy())

            # Afficher les coûts et l'allocation
            result_text = "Coûts entre les usines et les magasins :\n"
            result_text += tabulate(couts, tablefmt="grid") + "\n\n"
            result_text += "Allocation Nord-Ouest :\n"
            result_text += tabulate(allocation, tablefmt="grid")
            result_label.config(text=result_text)
        except Exception as e:
            result_label.config(text=f"Erreur : {str(e)}")

    ModernButton(main_frame, text="Exécuter", command=run_algorithm).pack(pady=15)

# Interface principale
def show_main_interface():
    root = tk.Tk()
    root.title("Algorithmes de Graphes")
    root.geometry("600x400")
    root.configure(bg=WINDOW_BG)

    main_frame = tk.Frame(root, bg=WINDOW_BG, padx=20, pady=20)
    main_frame.pack(fill=tk.BOTH, expand=True)

    title_label = tk.Label(
        main_frame,
        text="Algorithmes de Théorie des Graphes",
        font=("Helvetica", 18, "bold"),
        fg=EMSI_GREEN,
        bg=WINDOW_BG
    )
    title_label.pack(pady=20)

    subtitle_label = tk.Label(
        main_frame,
        text="École Marocaine des Sciences de l'Ingénieur\n Encadré par: Dr El MKHALET Mouna\n Présenté par: GOUNINE Youssef - KARARA Nabil",
        font=("Helvetica", 12),
        fg=DARK_GRAY,
        bg=WINDOW_BG
    )
    
    subtitle_label.pack(pady=10)

    buttons_frame = tk.Frame(main_frame, bg=WINDOW_BG)
    buttons_frame.pack(pady=30)

    algorithms = [
        ("Welsh-Powell", execute_welsh_powell_algorithm),
        ("Dijkstra", execute_dijkstra_algorithm),
        ("Kruskal", execute_kruskal_algorithm),
        ("Bellman-Ford", execute_bellman_ford_algorithm),
        ("Potentiel-Métra", execute_potentiel_metra_algorithm),
        ("Ford-Fulkerson", execute_ford_fulkerson_algorithm),
        ("Stepping Stone", execute_stepping_stone_algorithm),
        ("Moindre Coût", execute_moindre_cout_algorithm),
        ("Nord-Ouest", execute_nord_ouest_algorithm),
    ]

    for i, (text, command) in enumerate(algorithms):
        btn = ModernButton(buttons_frame, text=text, command=command, width=20)
        btn.grid(row=i // 3, column=i % 3, padx=10, pady=10)

    root.mainloop()

# Exécuter l'application
if __name__ == "__main__":
    show_main_interface()
