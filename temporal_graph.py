import math
import statistics
import collections
import powerlaw
import utils
import pytglib as tgl


class TemporalGraph:

    def __init__(self, path, directed):
        self.directed = directed
        self.edge_list = tgl.load_ordered_edge_list(path, directed)
        self.incident_list = tgl.to_incident_lists(self.edge_list)

    def number_of_nodes(self) -> int:
        return self.edge_list.getNumberOfNodes()
    
    def number_of_edges(self) -> int:
        return self.edge_list.getNumberOfEdges()

    def is_directed(self) -> bool:
        return self.directed

    def getNodeMap(self) -> dict:
        return self.edge_list.getNodeMap()
    
    def getReverseNodeMap(self) -> dict:
        return self.edge_list.getReverseNodeMap()

    def get_statistics(self) -> tgl.TemporalGraphStatistics:
        return tgl.get_statistics(self.edge_list)

    def timestamps(self) -> list:
        return [e.t for e in self.edge_list.getEdges()]
    
    def number_of_timestamps(self) -> int:
        return len(set(self.timestamps()))

    def edge_counts(self) -> collections.Counter:
        edges = [(e.u, e.v) for e in self.edge_list.getEdges()]
        return collections.Counter(edges)

    def edge_cardinalities(self, return_exponent=False) -> list:
        cardinalities = list(self.edge_counts().values())

        if return_exponent:
            with utils.HiddenPrints():
                return cardinalities, powerlaw.Fit(cardinalities, discrete=True, xmin=1).alpha
        else:
            return cardinalities

    def temporal_diameter(self) -> float:
        return tgl.temporal_diameter(self.edge_list, self.edge_list.getTimeInterval(), tgl.Distance_Type.Minimum_Hops)
    
    def topological_overlap(self) -> list:
        top_overlap = [tgl.topological_overlap(self.incident_list, i, self.incident_list.getTimeInterval()) for i in range(self.number_of_nodes())]
        return list(filter(lambda x: not math.isnan(x), top_overlap))
    
    def temporal_correlation_coefficient(self) -> float:
        overlap = self.topological_overlap()
        return statistics.fmean(overlap) if overlap else 0

    def temporal_clustering(self) -> list:
        return tgl.temporal_clustering_coefficient(self.incident_list, self.incident_list.getTimeInterval())
    
    def temporal_clustering_coefficient(self) -> float:
        clustering = self.temporal_clustering()
        return statistics.fmean(clustering) if clustering else 0

    def reachability(self) -> list:
        reachable = []
        for i in range(self.number_of_nodes()):
            reachable.append(tgl.number_of_reachable_nodes(self.edge_list, i, self.edge_list.getTimeInterval()))
        return reachable
    
    def avg_reachability(self) -> float:
        return statistics.fmean(self.reachability())

    def neighbors(self, u: int) -> set:
        return {e.v for e in self.incident_list.getNode(u).outEdges}
    
    def degree(self, u: int) -> int:
        return len(self.neighbors(u))

    def degree_sequence(self, return_exponent=False) -> list:
        degrees = [self.degree(i) for i in range(self.number_of_nodes())]
        
        if return_exponent:
            with utils.HiddenPrints():
                return degrees, powerlaw.Fit(degrees, discrete=True, xmin=1).alpha
        else:
            return degrees

    def degree_by_node(self) -> dict:
        return {i: self.degree(i) for i in range(self.number_of_nodes())}

    def max_degree(self) -> int:
        return max(self.degree_sequence())

    def avg_degree(self) -> float:
        return statistics.fmean(self.degree_sequence())
