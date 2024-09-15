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
    
def checkNegative_weight(g):
    for node in g.graph:
        for edge in g.get_edges(node):
            if edge[2] < 0:
                return True
    return False

def shortestPathDIJKSTRA(g):
    dist = {}
    prev = {}
    for node in g.graph.keys():
        dist_node = g.dijkstra(node)[0]
        prev_node = g.dijkstra(node)[1]

        for k in dist_node.keys():
            if (node,k) in dist.keys() or (k,node) in dist.keys():
                continue
            dist[(node,k)] = dist_node[k]
            if len(prev_node[k]) > 1:
                prev[(node,k)] = ((prev_node[k])[1:])[-1]
            else:
                prev[(node,k)] = 0

    return dist,prev

def shortestPathBF(g):
    dist = {}
    prev = {}
    for node in g.graph.keys():
        dist_node = g.bellmanFord(node)[0]
        prev_node = g.bellmanFord(node)[1]

        for k in dist_node.keys():
            if (node,k) in dist.keys() or (k,node) in dist.keys():
                continue
            dist[(node,k)] = dist_node[k]
            if len(prev_node[k]) > 1:
                prev[(node,k)] = ((prev_node[k])[1:])[-1]
            else:
                prev[(node,k)] = 0

    return dist,prev

def AllPair_shortestPath(g):
    if checkNegative_weight(g):
        return shortestPathBF(g)
    return shortestPathDIJKSTRA(g)
