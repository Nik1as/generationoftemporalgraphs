import networkit as nk
import networkx as nx
import graph_io
import utils
import sampling
import sampling
import statistics
from collections import Counter


def temporal_graph_generator(G: nk.graph.Graph, 
                             phi: sampling.Distribution, 
                             weights: list, 
                             timestamps: list=None, 
                             wswr: sampling.WeightedSamplingWithoutReplacement=sampling.Hybrid):
    timestamps = timestamps or range(len(weights)) # len(weights) = T
    wswr = wswr(timestamps, weights)

    edges = []
    for u, v in G.iterEdges():
        k = phi.generate()
        for t in wswr.sample(k):
            edges.append((u, v, t))
    return edges


def erdos_renyi_tg_from_graph(path: str, directed: bool=False):
    graph = graph_io.read_temporal_graph(path, directed=directed, static=False)

    timestamps, weights = timestamps_and_weights(graph)
    sampler = sampling.AliasAddOne(cardinality_distribution(graph, len(timestamps)))

    graph.removeMultiEdges()
    graph.removeSelfLoops()

    p = get_prob(graph.numberOfNodes(), graph.numberOfEdges(), directed)
    
    erdos_renyi_graph = nk.generators.ErdosRenyiGenerator(graph.numberOfNodes(), p, directed, selfLoops=False).generate()

    return temporal_graph_generator(erdos_renyi_graph, sampler, weights, timestamps)


def chung_lu_tg_from_graph(path: str, directed: bool=False):
    graph = graph_io.read_temporal_graph(path, directed=directed, static=False)

    timestamps, weights = timestamps_and_weights(graph)
    sampler = sampling.AliasAddOne(cardinality_distribution(graph, len(timestamps)))

    graph.removeMultiEdges()
    graph.removeSelfLoops()

    if directed:
        graph = nk.graphtools.toUndirected(graph)

    degree_seq = nk.centrality.DegreeCentrality(graph).run().scores()
    chung_lu_graph = nk.generators.ChungLuGenerator(degree_seq).generate()

    return temporal_graph_generator(chung_lu_graph, sampler, weights, timestamps)


def havel_hakimi_tg_from_graph(path: str, directed: bool=False):
    graph = graph_io.read_temporal_graph(path, directed=directed, static=False)

    timestamps, weights = timestamps_and_weights(graph)
    sampler = sampling.AliasAddOne(cardinality_distribution(graph, len(timestamps)))

    graph.removeMultiEdges()

    if directed:
        in_degree = []
        out_degree = []
        for node in graph.iterNodes():
            in_degree.append(graph.degreeIn(node))
            out_degree.append(graph.degreeOut(node))

        havel_hakimi_graph = nx.directed_havel_hakimi_graph(in_degree, out_degree)
        havel_hakimi_graph = nk.nxadapter.nx2nk(havel_hakimi_graph)
        return temporal_graph_generator(havel_hakimi_graph, sampler, weights, timestamps)
    else:
        sequence = nk.centrality.DegreeCentrality(graph).run().scores()
        havel_hakimi_graph = nk.generators.HavelHakimiGenerator(sequence).generate()
        return temporal_graph_generator(havel_hakimi_graph, sampler, weights, timestamps)


def hyperbolic_tg_from_graph(path: str, directed: bool=False, power_law_exponent: float=2.5):
    graph = graph_io.read_temporal_graph(path, directed=directed, static=False)

    timestamps, weights = timestamps_and_weights(graph)
    sampler = sampling.AliasAddOne(cardinality_distribution(graph, len(timestamps)))

    graph.removeMultiEdges()
    graph.removeSelfLoops()

    if directed:
        graph = nk.graphtools.toUndirected(graph)

    degree_seq = nk.centrality.DegreeCentrality(graph).run().scores()

    hyperbolic_graph = nk.generators.HyperbolicGenerator(graph.numberOfNodes(), statistics.fmean(degree_seq), power_law_exponent).generate()

    return temporal_graph_generator(hyperbolic_graph, sampler, weights, timestamps)


def timestamps_and_weights(graph: nk.graph.Graph):
    timestamps = [int(e[2]) for e in graph.iterEdgesWeights()]
    timestamp_counts = Counter(timestamps)
    return zip(*timestamp_counts.items())


def cardinality_distribution(graph: nk.graph.Graph, timestamps: int):
    cardinalities = list(Counter(graph.iterEdges()).values())
    cardinalities_counts = Counter(cardinalities)
    cardinality_distribution = []
    for i in range(1, timestamps + 1):
        cardinality_distribution.append(cardinalities_counts.get(i, 0) / len(cardinalities))
    return cardinality_distribution


def get_prob(n: int, m: int, directed: bool, selfLoops=False) -> float:
    if directed:
        if selfLoops:
            return m / (n * n)
        else:
            return m / (n * (n - 1))
    else:
        return (2 * m) / (n * (n - 1))
