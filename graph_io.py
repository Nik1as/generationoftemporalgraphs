import csv
import networkit as nk


def read_temporal_graph(path: str, directed: bool=False, static: bool=False):
    with open(path) as csvfile:
        graph = nk.Graph(weighted=True, directed=directed)
        reader = csv.reader(csvfile, delimiter=" ")

        for row in reader:
            u, v, t, *tt = row
            graph.addEdge(int(u), int(v), float(t), addMissing=True)
        if static:
            graph.removeMultiEdges()
        return graph


def read_edge_list(path: str) -> list:
    edges = []
    with open(path, "r") as file:
        reader = csv.reader(file, delimiter=" ")
        for row in reader:
            edges.append(list(map(int, row)))
    return edges


def write_edge_list(path: str, edges: list):
    with open(path, "w") as file:
        writer = csv.writer(file, delimiter=" ")
        writer.writerows(edges)
