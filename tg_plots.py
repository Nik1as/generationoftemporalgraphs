from collections import Counter
from itertools import repeat
from typing import Callable, Iterable
from temporal_graph import TemporalGraph

import os
import math
import csv
import json
import enum
import statistics
import graph_io
import utils
import powerlaw
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import networkit as nk
import pytglib as tgl

sns.set_theme()

DATA_PATH = "data"
GRAPH_STATS_PATH = os.path.join(DATA_PATH, "graphs_stats.csv")
TEMPORAL_GRAPHS_PATH = os.path.join(DATA_PATH, "temporal_graphs")

with open(os.path.join(DATA_PATH, "directed.json")) as f:
    DIRECTED = json.load(f)


class TemporalGraphCategories(str, enum.Enum):
    CITATION_GRAPHS = "citation"
    COMMUNICATION_GRAPHS = "communication"
    HUMAN_CONTACT_GRAPHS = "human_contact"
    SOCIAL_MEDIA_GRAPHS = "social_media"

    def __str__(self) -> str:
        return self.value

    @classmethod
    def values(cls):
        return list(map(lambda c: c.value, cls))


def edge_cardinality_distribution(path: str, ax=None):
    ax = ax or plt.gca()

    tg = TemporalGraph(path, is_directed(path))
    cardinalities, exponent = tg.edge_cardinalities(return_exponent=True)

    sns.scatterplot(ax=ax, data=Counter(cardinalities))
    ax.set_title(f"{utils.file_name(path).replace('_', ' ')} - {round(exponent, 2)}")
    ax.set_yscale("log")
    if max(cardinalities) > 5:
        ax.set_xscale("log")
    else:
        ax.set_xticks(range(1, 4))


def timestamp_distribution(path: str, artifacts: float=None, scatter=True, ax=None):
    ax = ax or plt.gca()

    tg = graph_io.read_temporal_graph(path, is_directed(path))
    timestamps = [int(e[2]) for e in tg.iterEdgesWeights()]
    timestamp_counts = Counter(timestamps)
    if artifacts:
        remove = int(artifacts * len(timestamp_counts))
        timestamp_counts = dict(sorted(timestamp_counts.items())[remove:-remove])
    if scatter:
        sns.scatterplot(ax=ax, data=timestamp_counts)
    else:
        timestamps.clear()
        for t in timestamp_counts.keys():
            timestamps.extend(repeat(t, timestamp_counts[t]))

        sns.histplot(ax=ax, x=timestamps, bins=100)
    
    ax.set_title(utils.file_name(path).replace("_", " "))
    ax.set_yscale("log")


def temporal_clustering(path: str, ax=None):
    ax = ax or plt.gca()

    tgs = tgl.load_ordered_edge_list(path, is_directed(path))
    tg = tgl.to_incident_lists(tgs)
    
    clustering = tgl.temporal_clustering_coefficient(tg, tg.getTimeInterval())

    sns.histplot(ax=ax, x=clustering, bins=5)
    ax.set_title(f"{utils.file_name(path).replace('_', ' ')} - {round(statistics.fmean(clustering), 2)}")
    ax.set_yscale("log")
    ax.set_ylabel("")
    ax.set_xlim(0, 1)


def topological_overlap(path: str, ax=None):
    ax = ax or plt.gca()

    tgs = tgl.load_ordered_edge_list(path, is_directed(path))
    tg = tgl.to_incident_lists(tgs)

    top_overlap = [
        tgl.topological_overlap(tg, i, tg.getTimeInterval())
        for i in range(tgs.getNumberOfNodes())
    ]
    top_overlap = list(filter(lambda x: not math.isnan(x), top_overlap))

    sns.histplot(ax=ax, x=top_overlap, bins=10)
    ax.set_title(f"{utils.file_name(path).replace('_', ' ')} - {round(statistics.fmean(top_overlap), 2)}")
    ax.set_yscale("log")
    ax.set_ylabel("")


def degree_vs_cardinality(path: str, degree_func: Callable[[int, int], int]=lambda x, y: x * y, ax=None):
    ax = ax or plt.gca()

    tg = graph_io.read_temporal_graph(path, is_directed(path))
    cardinalities = Counter(tg.iterEdges())
    tg.removeMultiEdges()
    tg.removeSelfLoops()

    deg = nk.centrality.DegreeCentrality(tg)
    deg.run()
    deg = dict(deg.ranking())

    x = []
    y = []
    for (u, v), k in cardinalities.items():
        x.append(degree_func(deg[u], deg[v]))
        y.append(k)

    sns.scatterplot(ax=ax, x=x, y=y)
    ax.set_title(utils.file_name(path).replace("_", " "))
    ax.set_xscale("log")
    ax.set_yscale("log")


def degree_vs_topological_overlap(path: str, ax=None):
    ax = ax or plt.gca()

    tg = tgl.load_ordered_edge_list(path, is_directed(path))
    tg = tgl.to_incident_lists(tg)
    degree = []
    top_overlap = []
    for node in range(tg.getNumberOfNodes()):
        edges = {(e.u, e.v) for e in tg.getNode(node).outEdges}
        degree.append(len(edges))
        top_overlap.append(tgl.topological_overlap(tg, node, tg.getTimeInterval()))

    sns.histplot(ax=ax, x=degree, y=top_overlap, bins=30, log_scale=(True, False))
    ax.set_title(utils.file_name(path).replace("_", " "))


def degree_vs_temporal_clustering(path: str, ax=None):
    ax = ax or plt.gca()

    tg = tgl.load_ordered_edge_list(path, is_directed(path))
    tg = tgl.to_incident_lists(tg)
    degree = []
    clustering = []
    for i in range(tg.getNumberOfNodes()):
        edges = {(e.u, e.v) for e in tg.getNode(i).outEdges}
        degree.append(len(edges))
        clustering.append(
            tgl.temporal_clustering_coefficient(tg, i, tg.getTimeInterval())
        )

    sns.scatterplot(ax=ax, x=degree, y=clustering)
    ax.set_title(utils.file_name(path).replace("_", " "))
    ax.set_xscale("log")


# ===============
# static graphs
# ===============


def degree_distribution(path: str, ax=None):
    ax = ax or plt.gca()

    graph = graph_io.read_temporal_graph(path, is_directed(path), True)

    dd = nk.centrality.DegreeCentrality(graph).run().scores()
    dd = list(filter(lambda x: x > 0, dd))

    fit = powerlaw.Fit(dd, discrete=True, xmin=1)

    degree_counts = Counter(dd)

    sns.scatterplot(ax=ax, data=degree_counts)
    ax.set_title(f"{utils.file_name(path).replace('_', ' ')} - {round(fit.alpha.item(), 2)}")
    ax.set_yscale("log")
    ax.set_xscale("log")


def assortativity_degree(path: str, ax=None):
    ax = ax or plt.gca()

    graph = graph_io.read_temporal_graph(path, is_directed(path), True)
    degrees = dict(
        filter(
            lambda x: x[1] > 0, nk.centrality.DegreeCentrality(graph).run().ranking()
        )
    )

    x = []
    y = []
    for node in degrees.keys():
        x.append(degrees[node])
        y.append(statistics.fmean([degrees[x] for x in graph.iterNeighbors(node)]))

    sns.histplot(ax=ax, x=x, y=y, bins=30, log_scale=(True, True))
    ax.set_title(utils.file_name(path).replace("_", " "))


def clustering(path: str, ax=None):
    ax = ax or plt.gca()

    graph = graph_io.read_temporal_graph(path, is_directed(path), True)
    graph.removeSelfLoops()
    lcc = nk.centrality.LocalClusteringCoefficient(graph)
    lcc.run()
    sns.histplot(ax=ax, x=lcc.scores(), bins=10)
    ax.set_title(f"{utils.file_name(path).replace('_', ' ')} - {round(statistics.fmean(lcc.scores()), 2)}")
    ax.set_yscale("log")
    ax.set_ylabel("")


def plot_graphs(category: str, func: callable, 
                ncols: int=4, plot_width: int=5, plot_height: int=4, 
                x_label: str="", y_label: str="",
                file_extension="pdf",
                **kwargs):
    
    files = list(iter_temporal_graphs([category]))
    n = len(files)
    nrows = math.ceil(n / ncols)
    fig, axs = plt.subplots(
        nrows, ncols, figsize=(ncols * plot_width, nrows * plot_height)
    )
    fig.tight_layout(pad=3)

    for i, file in enumerate(files):
        row = i // ncols
        col = i % ncols
        if nrows > 1:
            func(file, ax=axs[row, col], **kwargs)
        else:
            func(file, ax=axs[col], **kwargs)

    for i in range(n, nrows * ncols):
        row = i // ncols
        col = i % ncols
        axs[row, col].set_visible(False)

    fig.supxlabel(x_label)
    fig.supylabel(y_label)
    fig.savefig(f"data/graphics/{category}/{func.__name__}.{file_extension}", bbox_inches="tight")
    fig.show()


# ===============
# stats functions
# ===============


def stats(paths: Iterable[str], directed: Callable[[str], bool], node_limit: int=500_000):
    with open(GRAPH_STATS_PATH, "w") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "Datensatz",
                "Kategorie",
                "Knoten",
                "Kanten",
                "statische_Kanten",
                "Zeitstempel",
                "Gewichte",
                "Kanten_Kardinalitaeten_Exponent",
                "Grad_Exponent",
                "temp_Durchmesser",
                "Erreichbarkeit",
                "temp_Korrelation",
                "temp_Clustering",
                "Grad",
                "Clustering",
                "Grad_assortativity"
            ]
        )

        for network_type, path in paths:
            is_directed = directed(path)

            tg = TemporalGraph(path, is_directed)
            stats = tg.get_statistics()

            diameter = -1
            reachability = -1
            if tg.number_of_nodes() < node_limit:
                reachability = tg.avg_reachability()
                diameter = tg.temporal_diameter()

            temp_correlation = tg.temporal_correlation_coefficient()
            temp_clustering = tg.temporal_clustering_coefficient()

            _, cardinalities_exponent = tg.edge_cardinalities(return_exponent=True)

            tg_nk = graph_io.read_temporal_graph(path, is_directed, static=True)

            degree_seq = nk.centrality.DegreeCentrality(tg_nk).run().scores()
            degree_fit = powerlaw.Fit(degree_seq, discrete=True, xmin=1)

            assortativity = nk.correlation.Assortativity(tg_nk, degree_seq).run().getCoefficient()

            tg_nk.removeSelfLoops()

            # Local clustering coefficient is currently not implemented for directed graphs in networkit
            if is_directed: 
                tg_nk = nk.graphtools.toUndirected(tg_nk)

            clustering = nk.centrality.LocalClusteringCoefficient(tg_nk).run().scores()

            writer.writerow(
                [
                    utils.file_name(path),
                    network_type,
                    stats.numberOfNodes,
                    stats.numberOfEdges,
                    stats.numberOfStaticEdges,
                    stats.numberOfTimeStamps,
                    stats.numberOfTransitionTimes,
                    cardinalities_exponent,
                    degree_fit.alpha,
                    diameter,
                    reachability,
                    temp_correlation,
                    temp_clustering,
                    statistics.fmean(degree_seq),
                    statistics.fmean(clustering),
                    assortativity
                ]
            )

            print(path, tg.number_of_nodes())


def avg_reachability(outside_legend=False):
    data = pd.read_csv(GRAPH_STATS_PATH).query("Erreichbarkeit > 0")

    sns.scatterplot(data, x="Knoten", y="Erreichbarkeit", hue="Kategorie")
    plt.plot(*zip(*[(x, x) for x in range(1, max(data["Knoten"]))]), label="n")
    plt.loglog()
    plt.xlabel("Knoten")
    plt.ylabel("durchschnittliche Erreichbarkeit")
    show_legend(outside_legend)
    plt.savefig("data/graphics/reachability.pdf", bbox_inches="tight")
    plt.show()


def avg_reachability_ratio(outside_legend=False):
    data = pd.read_csv(GRAPH_STATS_PATH).query("Erreichbarkeit > 0")
    data["ratio"] = data["Erreichbarkeit"] / data["Knoten"]

    sns.scatterplot(data, x="Knoten", y="ratio", hue="Kategorie")
    plt.xscale("log")
    plt.ylim(0, 1)
    plt.xlabel("Knoten")
    plt.ylabel("Anteil erreichbarer Knoten")
    show_legend(outside_legend)
    plt.savefig("data/graphics/reachability_ratio.pdf", bbox_inches="tight")
    plt.show()


def temporal_diameter(outside_legend=False):
    data = pd.read_csv(GRAPH_STATS_PATH).query("temp_Durchmesser > 0")

    sns.scatterplot(data, x="Knoten", y="temp_Durchmesser", hue="Kategorie")
    plt.plot(*zip(*[(x, math.log2(x)) for x in range(1, max(data["Knoten"]))]), label="log2(n)")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Knoten")
    plt.ylabel("temporaler Durchmesser")
    show_legend(outside_legend)
    plt.savefig("data/graphics/diameter.pdf", bbox_inches="tight")
    plt.show()


def temporal_correlation_coefficient(aspect=2):
    data = pd.read_csv(GRAPH_STATS_PATH)

    sns.catplot(data, x="Kategorie", y="temp_Korrelation", kind="violin", height=5, aspect=aspect)
    plt.ylim(0, 1)
    plt.xlabel("Graph Kategorie")
    plt.ylabel("temporaler Korrelations-Koeffizient")
    plt.savefig("data/graphics/temporal_correlation_coefficient.pdf", bbox_inches="tight")
    plt.show()


def temporal_clustering(aspect=2):
    data = pd.read_csv(GRAPH_STATS_PATH)

    sns.catplot(data, x="Kategorie", y="temp_Clustering", kind="violin", height=5, aspect=aspect)
    plt.ylim(0, 1)
    plt.xlabel("Graph Kategorie")
    plt.ylabel("temporales Clustering")
    plt.savefig("data/graphics/temp_clustering.pdf", bbox_inches="tight")
    plt.show()


def assortativity(aspect=2):
    data = pd.read_csv(GRAPH_STATS_PATH)

    sns.catplot(data, x="Kategorie", y="Grad_assortativity", kind="violin", height=5, aspect=aspect)
    plt.ylim(-1, 1)
    plt.xlabel("Graph Kategorie")
    plt.ylabel("Grad Assortativity")
    plt.savefig("data/graphics/assortativity.pdf", bbox_inches="tight")
    plt.show()


def avg_degree(outside_legend=False):
    data = pd.read_csv(GRAPH_STATS_PATH)

    sns.scatterplot(data, x="Knoten", y="Grad", hue="Kategorie")
    plt.loglog()
    plt.xlabel("Knoten")
    plt.ylabel("Durchschnittsgrad")
    show_legend(outside_legend)
    plt.savefig("data/graphics/avg_degree.pdf", bbox_inches="tight")
    plt.show()


def static_clustering(aspect=2):
    data = pd.read_csv(GRAPH_STATS_PATH)

    sns.catplot(data, x="Kategorie", y="Clustering", kind="violin", height=5, aspect=aspect)
    plt.ylim(0, 1)
    plt.xlabel("Graph Kategorie")
    plt.ylabel("lokales Clustering")
    plt.savefig("data/graphics/static_clustering.pdf", bbox_inches="tight")
    plt.show()


def degree_exponent(aspect=2):
    data = pd.read_csv(GRAPH_STATS_PATH)

    sns.catplot(data, x="Kategorie", y="Grad_Exponent", kind="violin", height=5, aspect=aspect)
    plt.ylim(1, 3)
    plt.xlabel("Graph Kategorie")
    plt.ylabel("Grad Exponent")
    plt.savefig("data/graphics/degree_exponent.pdf", bbox_inches="tight")
    plt.show()


def edge_cardinality_exponent(aspect=2):
    data = pd.read_csv(GRAPH_STATS_PATH)

    sns.catplot(data, x="Kategorie", y="Kanten_Kardinalitaeten_Exponent", kind="violin", height=5, aspect=aspect)
    plt.ylim(1, 3)
    plt.xlabel("Graph Kategorie")
    plt.ylabel("Kardinalitäten Exponent")
    plt.savefig("data/graphics/cardinality_exponent.pdf", bbox_inches="tight")
    plt.show()


def connected_components(aspect=2):
    data = pd.read_csv(os.path.join(DATA_PATH, "connected_components.csv"))
    data["ratio"] = data["Knoten_Zusammenhangskomponente"] / data["Knoten"]

    sns.catplot(data, x="Kategorie", y="ratio", kind="violin", height=5, aspect=aspect)
    plt.ylim(0, 1)
    plt.xlabel("Graph Kategorie")
    plt.ylabel("größte Zusammenhangskomponente")
    plt.savefig("data/graphics/connected_components.pdf", bbox_inches="tight")
    plt.show()


def iter_temporal_graphs(network_types: list, return_type: bool=False) -> Iterable[str]:
    for folder in network_types:
        path = os.path.join(TEMPORAL_GRAPHS_PATH, folder)
        for file in sorted(os.listdir(path), key=str.casefold):
            file_path = os.path.join(path, file)
            if os.path.isfile(file_path):
                if return_type:
                    yield folder, file_path
                else:
                    yield file_path


# ===============
# util functions
# ===============

def show_legend(outside_legend):
    if outside_legend:
        plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    else:
        plt.legend()


def is_directed(path: str) -> bool:
    return DIRECTED.get(utils.file_name(path), False)
