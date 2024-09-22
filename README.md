# Shortest Path Algorithms

This project involves implementing and analyzing various shortest path algorithms, including Dijkstra's, Bellman-Ford, and A*. Performance is evaluated on both synthetic and real-world data. The report can be found in the file named 2XC3_Final_Project_Report.pdf.

## Table of Contents

1. [Part 1: Single Source Shortest Path Algorithms](#part-1-single-source-shortest-path-algorithms)
   - [1.1 Dijkstra’s Algorithm](#11-dijkstras-algorithm)
   - [1.2 Bellman-Ford Algorithm](#12-bellman-ford-algorithm)
   - [1.3 Performance Analysis](#13-performance-analysis)
2. [Part 2: All-Pairs Shortest Path Algorithm](#part-2-all-pairs-shortest-path-algorithm)
3. [Part 3: A* Algorithm](#part-3-a-algorithm)
   - [3.1 A* Algorithm Implementation](#31-a-algorithm-implementation)
   - [3.2 A* Algorithm Analysis](#32-a-algorithm-analysis)
4. [Part 4: Compare Shortest Path Algorithms](#part-4-compare-shortest-path-algorithms)
5. [Part 5: Code Organization and UML](#part-5-code-organization-and-uml)

## Part 1: Single Source Shortest Path Algorithms

### 1.1 Dijkstra’s Algorithm

Implemented a variation of Dijkstra’s algorithm where each node can be relaxed at most `k` times. The function `dijkstra(graph, source, k)` returns a dictionary mapping nodes to their shortest distance and path.

### 1.2 Bellman-Ford Algorithm

Implemented a variation of Bellman-Ford’s algorithm with a relaxation limit of `k` times. The function `bellman_ford(graph, source, k)` returns a dictionary mapping nodes to their shortest distance and path.

### 1.3 Performance Analysis

Designed experiments to analyze the performance of the algorithms from Parts 1.1 and 1.2. Factors considered include graph size, graph density, and the value of `k`. Performance is evaluated in terms of accuracy, time, and space complexity.

## Part 2: All-Pairs Shortest Path Algorithm

Implemented an algorithm to find shortest paths between all pairs of nodes. The solution addresses both positive and negative edge weights. Discussed the complexity of the algorithms for dense graphs, comparing with Dijkstra’s and Bellman-Ford’s complexities.

## Part 3: A* Algorithm

### 3.1 A* Algorithm Implementation

Implemented the A* algorithm using a priority queue. The function `A_Star(graph, source, destination, heuristic)` returns a tuple with a predecessor dictionary and the shortest path from the source to the destination.

### 3.2 A* Algorithm Analysis

- **Issues Addressed by A*:** A* improves upon Dijkstra’s by using a heuristic to guide the search, making it more efficient for certain types of problems.
- **Empirical Testing:** Discussed how to empirically test and compare Dijkstra’s and A* algorithms.
- **Heuristic Comparison:** Compared A* with randomly generated heuristics and discussed the impact on performance.
- **Applications:** Discussed scenarios where A* is preferred over Dijkstra’s.

## Part 4: Compare Shortest Path Algorithms

Compared the performance of Dijkstra’s and A* algorithms using real-world data from the London Subway system. Analyzed performance on different types of station connections, including stations on the same line, adjacent lines, and lines requiring multiple transfers. Discussed how many lines the shortest path uses and the circumstances under which A* outperforms Dijkstra’s.

## Part 5: Code Organization and UML

Organized code according to the provided UML diagram. Discussed design principles and patterns used, and addressed how the UML could be adapted for different types of nodes and graphs. Explored alternative implementations of the `Graph` class and other graph representations.

