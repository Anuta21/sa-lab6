import numpy as np
import networkx as nx
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

class CognitiveMap:
    def __init__(self, adj_matrix, node_names):
        assert adj_matrix.shape[0] == adj_matrix.shape[1]
        self.graph = nx.from_numpy_array(adj_matrix, create_using=nx.DiGraph)
        self.node_names = node_names
        nx.set_node_attributes(self.graph, node_names, 'name')
        self.cycles = [cycle for cycle in nx.simple_cycles(self.graph)]
    
    def getAdjMatrx(self):
        return nx.adjacency_matrix(self.graph).todense()

    @staticmethod
    def getSpectralRadius(matrix):
        return np.max(np.abs(np.linalg.eigvals(matrix)))

    def isPerturbationStable(self):
        return self.getSpectralRadius(self.getAdjMatrx()) <= 1

    def isNumericallyStable(self):
        return self.getSpectralRadius(self.getAdjMatrx()) < 1

    def getCycles(self):
        return self.cycles

    def _isEven(self, cycle, adj_matrix=None):
        if adj_matrix is None:
            adj_matrix = self.getAdjMatrx()
        cycle_temp = cycle.copy()
        cycle_temp.append(cycle[0])
        sign = 1
        for i in range(len(cycle)):
            sign *= np.sign(adj_matrix[cycle_temp[i], cycle_temp[i+1]])
        
        return sign == 1

    def getEvenCycles(self):
        adj_matrix = self.getAdjMatrx()
        return [cycle for cycle in self.cycles if self._isEven(cycle, adj_matrix)]

    def getEigenvalues(self):
        return np.linalg.eigvals(self.getAdjMatrx())
            
    def impulse_model(self, init_q, steps):
        N = len(self.node_names)
        x_0 = np.zeros((N, 1))
        x_list = [x_0, x_0]
        q = np.array(init_q).reshape((N, 1))
        A = self.getAdjMatrx()
        for _ in range(steps):
            x_next = x_list[-1] + A @ (x_list[-1] - x_list[-2]) + q
            x_list.append(x_next)

        res = np.array(x_list)[1:, :, 0]

        return res
