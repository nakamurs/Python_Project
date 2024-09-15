class IndexMinPQ:
    def __init__(self, data):
        self.nodes = []
        self.weights = []
        self.index = {}
        for node, weight in data:
            self.nodes.append(node)
            self.weights.append(weight)
            self.index[node] = len(self.nodes) - 1

        self.length = len(data)
        self.build_heap()

    def find_left_index(self, index):
        return 2 * index + 1

    def find_right_index(self, index):
        return 2 * index + 2

    def find_parent_index(self, index):
        return (index - 1) // 2

    def swap(self, i, j):
        self.nodes[i], self.nodes[j] = self.nodes[j], self.nodes[i]
        self.weights[i], self.weights[j] = self.weights[j], self.weights[i]
        self.index[self.nodes[i]], self.index[self.nodes[j]] = i, j

    def heapify(self, index):
        smallest_known_index = index
        left_index = self.find_left_index(index)
        right_index = self.find_right_index(index)

        if left_index < self.length and self.weights[left_index] < self.weights[index]:
            smallest_known_index = left_index
        if right_index < self.length and self.weights[right_index] < self.weights[smallest_known_index]:
            smallest_known_index = right_index
        if smallest_known_index != index:
            self.swap(index, smallest_known_index)
            self.heapify(smallest_known_index)

    def build_heap(self):
        for i in range(self.length // 2 - 1, -1, -1):
            self.heapify(i)

    def swim(self, index):
        parent_index = self.find_parent_index(index)

        while index > 0 and self.weights[index] < self.weights[parent_index]:
            self.swap(index, parent_index)
            index = parent_index
            parent_index = self.find_parent_index(index)

    def insert_value(self, node, weight):
        self.nodes.append(node)
        self.weights.append(weight)
        self.length += 1
        self.index[node] = self.length - 1  # Correct index assignment
        self.swim(self.length - 1)

    def update(self, node, new_weight):
        if node not in self.index:
            return None

        node_index = self.index[node]
        old_weight = self.weights[node_index]
        self.weights[node_index] = new_weight

        if new_weight < old_weight:
            self.swim(node_index)
        else:
            self.heapify(node_index)

    def extract_min(self):
        if self.length == 0:
            return 
        
        out = (self.nodes.pop(0), self.weights.pop(0))

        for k, v in self.index.items():
            if v == 0:
                del self.index[k]
                break  
        for k in self.index:
            self.index[k] -= 1

        self.length -= 1
        return out
    
import math
def A_Star(graph, source, destination, heuristic):
    predecessors = {}
    costs = {}
    
    W_edges = []
    for v in graph.graph.keys():
        costs[v] = math.inf
        predecessors[v] = None
        W_edges.append((v, math.inf))

    open_set = IndexMinPQ(W_edges)
    costs[source] = 0
    open_set.update(source, heuristic[source]) 
    
    while not open_set.length == 0:
        current_node = open_set.extract_min()[0]
        
        if current_node == destination:
            break

        for edge in graph.graph[current_node]:
            neighbor = edge.other(current_node)
            weight = edge.get_weight()
            new_cost = costs[current_node] + weight 
            if new_cost < costs[neighbor]:
                costs[neighbor] = new_cost
                predecessors[neighbor] = current_node
                open_set.update(neighbor, new_cost + heuristic[neighbor])  
    
    path = []
    current_node = destination
    while current_node is not None:
        path.append(current_node)
        current_node = predecessors[current_node]
    path.reverse()
    
    return predecessors, path