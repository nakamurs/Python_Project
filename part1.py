import unittest
import random
import math
import timeit

import matplotlib.pyplot as plt

class minHeap:
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
    
class WeightEdge:
    def __init__(self, u, v, weight) -> None:
        self.u = u
        self.v = v
        self.weight = weight
    
    def get_weight(self):
        return self.weight
    
    def start(self):
        return self.u
    
    def end(self):
        return self.v

    def other(self, node):
        if self.start() == node:
            return self.end()
        return self.start()
    
class Graph:
    def __init__(self, w_edges):
        self.graph = {}
        self.numE = 0         # Represents number of edges
        for edge in w_edges:
            v = edge.start()
            u = edge.end()
            if v not in self.graph.keys():
                self.graph[v]=[]
            self.graph[v].append(edge)

            if u not in self.graph.keys():
                self.graph[u]=[]
            self.graph[u].append(edge)
            # We are adding the same WeightEdge object to both the vertices
            self.numE +=2

        self.numV = len(self.graph.keys())  # Represents number of vertices

    def get_edges(self, node):
        edges = []
        if node not in self.graph:
            raise IndexError("vertex not exist")
        for e in self.graph[node]:
            edges.append([e.start(), e.end(), e.get_weight()])

        return edges
    
    def add_edge(self,node1, node2, w):
        edge = WeightEdge(node1, node2, w)

        if node1 not in self.graph.keys():
            self.graph[node1]=[]
            self.numV += 1
        self.graph[node1].append(edge)
        
        if node2 not in self.graph.keys():
            self.graph[node2]=[]
            self.numV += 1
        self.graph[node2].append(edge)

    def remove_edge(self, node):
        self.graph[node] = []
        for edges in self.graph.values():
            edges[:] = [edge for edge in edges if edge.start() != node and edge.end() != node]
                
    def get_graphList(self):
        all_edges = {}
        for key in self.graph.keys():
            edge = self.graph[key]
            for e in edge:
                if key not in all_edges:
                    all_edges[key]=[]
                all_edges[key].append([e.start(), e.end(), e.get_weight()])
        return all_edges
    
    def dijkstra(self, s, k=None):
        if k == None: 
            k = self.numV + 2

        dist = {}
        path = {}
        relaxLimit = {}
        for n in self.graph.keys():
            dist[n] = math.inf
            path[n] = []
            relaxLimit[n] = k

        W_edges = []
        for v in self.graph.keys():
            W_edges.append((v, math.inf))
        
        pq = minHeap(W_edges)

        pq.update(s, 0)
        dist[s] = 0
        relaxLimit[s] -=1

        while pq.length > 0:
            node, d = pq.extract_min()
            for e in self.graph[node]:
                adjNode = e.other(node)
                weight = e.get_weight()

                if (d + weight < dist[adjNode] and relaxLimit[adjNode] > 0):
                    dist[adjNode] = d + weight
                    pq.update(adjNode, dist[adjNode])
                    relaxLimit[adjNode] -= 1
                    path[adjNode] = path[node] + [node]

        return dist, path
    
    def bellmanFord(self, s, k=None):
        if k == None: 
            k = self.numV

        dist = {}
        path = {}
        relaxLimit = {}
        for n in self.graph.keys():
            dist[n] = math.inf
            path[n] = []
            relaxLimit[n] = k

        dist[s] = 0
        relaxLimit[s] -=1

        for i in range(self.numV - 1):
            for node in self.graph.keys():
                for e in self.graph[node]:
                    adjNode = e.other(node)
                    weight = e.get_weight()

                    if (dist[node] + weight < dist[adjNode] and relaxLimit[adjNode] > 0):
                        dist[adjNode] = dist[node] + weight
                        relaxLimit[adjNode] -= 1
                        path[adjNode] = path[node] + [node]

        # Check for negative cycles 
        for node in self.graph.keys():
            for e in self.graph[node]:
                adjNode = e.other(node)
                weight = e.get_weight()

                if (dist[node] + weight < dist[adjNode] and relaxLimit[adjNode] > 0):
                    print("Negative cycle detected")
                    return dist, path

        return dist, path
    
def draw_2_graphs(y1, y2, title):
    x_values = [i for i in range(30, 300, 30)]
    plt.figure()
    plt.plot(x_values, y1[1:], label="dijkstra", marker='o')
    plt.plot(x_values, y2[1:], label="bellmanFord", marker='s')
    plt.xlabel('Size')
    plt.ylabel('Run Times')
    plt.title(title)
    plt.legend()
    plt.show()

def randEdge(high, runs):
    w_edges = []
    generated_edges = []

    # To avoid getting disconnected graphs.
    for i in range(high):
        s = i
        e = random.randint(0, high - 1)
        
        while e == s:
            e = random.randint(0, high - 1)
        
        while (s, e) in generated_edges:
            e = random.randint(0, high - 1)
            while e == s:
                e = random.randint(0, high - 1)

        generated_edges.append((s, e))
        weight = random.randint(1, 10 * runs)
        w_edges.append(WeightEdge(s, e, weight))

    for i in range(runs - high):
        s = random.randint(0, high)
        e = random.randint(0, high)

        # In case we generate the same start and end node.
        while s == e: 
            e = random.randint(0, high-1)

        while (s, e) in generated_edges:
            s = random.randint(0, high - 1)
            e = random.randint(0, high - 1)
            while s == e:
                e = random.randint(0, high - 1)

        generated_edges.append((s,e))
        weight = random.randint(0, 10*runs)
        w_edges.append(WeightEdge(s,e,weight))

    return w_edges, generated_edges

runs = 300

dijkstraTime = []
bellmanFordTime = []

for i in range(30, runs + 1, 30): 
    var = randEdge(i,i)     
    edges = var[0]
    random_vertex = var[1][0][0]
    
    g1 = Graph(edges)

    start = timeit.default_timer()
    g1.bellmanFord(random_vertex)
    stop = timeit.default_timer()
    bellmanFordTime.append(stop-start)

    var = randEdge(i,i)     
    edges = var[0]
    random_vertex = var[1][0][0]
    
    g2 = Graph(edges)


    start = timeit.default_timer()
    g2.dijkstra(random_vertex)
    stop = timeit.default_timer()
    dijkstraTime.append(stop-start)


print("Average time for dijkstra :", sum(dijkstraTime) / len(dijkstraTime))
print("Average time for bellmanFordTime :", sum(bellmanFordTime) / len(bellmanFordTime))


draw_2_graphs(dijkstraTime, bellmanFordTime, "dijkstra v/s bellmanFord")

# K value 
def draw_2_graphs2(y1, y2, title):
    x_values = [i for i in range(1, 20)]
    plt.figure()
    plt.plot(x_values, y1[1:], label="dijkstra", marker='o')
    plt.plot(x_values, y2[1:], label="bellmanFord", marker='s')
    plt.xlabel('k values')
    plt.ylabel('Run Times')
    plt.title(title)
    plt.legend()
    plt.show()


runs = 20

dijkstraTime = []
bellmanFordTime = []

var = randEdge(40,40)     
edges = var[0]
random_vertex = var[1][0][0]


for i in range(1, runs + 1):    

    g1 = Graph(edges)
    g2 = Graph(edges)

    start = timeit.default_timer()
    g1.dijkstra(random_vertex, i)
    stop = timeit.default_timer()
    dijkstraTime.append(stop-start)

    start = timeit.default_timer()
    g2.bellmanFord(random_vertex, i)
    stop = timeit.default_timer()
    bellmanFordTime.append(stop-start)
    

print("Average time for dijkstra :", sum(dijkstraTime) / len(dijkstraTime))
print("Average time for bellmanFordTime :", sum(bellmanFordTime) / len(bellmanFordTime))


draw_2_graphs2(dijkstraTime, bellmanFordTime, "dijkstra v/s bellmanFord")

def draw_2_graphs3(y1, y2, title):
    x_values = [i for i in range(50, runs, 25)]
    plt.figure()
    plt.plot(x_values, y1[1:], label="dijkstra", marker='o')
    plt.plot(x_values, y2[1:], label="bellmanFord", marker='s')
    plt.xlabel('Graph Density')
    plt.ylabel('Run times')
    plt.title(title)
    plt.legend()
    plt.show()


runs = 300

dijkstraTime = []
bellmanFordTime = []

for i in range(50, runs + 1, 25): 
    var = randEdge(50,i)     
    edges = var[0]
    random_vertex = var[1][0][0]
    
    g1 = Graph(edges)

    start = timeit.default_timer()
    g1.bellmanFord(random_vertex)
    stop = timeit.default_timer()
    bellmanFordTime.append(stop-start)

    var = randEdge(50,i)     
    edges = var[0]
    random_vertex = var[1][0][0]
    
    g2 = Graph(edges)


    start = timeit.default_timer()
    g2.dijkstra(random_vertex)
    stop = timeit.default_timer()
    dijkstraTime.append(stop-start)


print("Average time for dijkstra :", sum(dijkstraTime) / len(dijkstraTime))
print("Average time for bellmanFordTime :", sum(bellmanFordTime) / len(bellmanFordTime))


draw_2_graphs3(dijkstraTime, bellmanFordTime, "dijkstra v/s bellmanFord")