import sys
import numpy as np
import time
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QWidget, QPushButton
)
from PyQt5.QtCore import Qt
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

class AntColonyAS:
    def __init__(self, graph):
        self.graph = graph
        self.num_nodes = graph.shape[0]
        self.pheromones = np.full_like(graph, 0.2, dtype=float)
        self.num_ants = 100

    def run(self):
        best_cost = float('inf')
        best_path = []
        
        for _ in range(self.num_ants):
            path, cost = self.simulate_ant()
            if cost < best_cost:
                best_cost = cost
                best_path = path

            self.update_pheromones(path, cost)

        return best_path, best_cost

    def simulate_ant(self):
        start = np.random.randint(self.num_nodes)
        visited = {start}
        path = [start]
        cost = 0

        while len(visited) < self.num_nodes:
            current = path[-1]
            probs = self.calculate_probabilities(current, visited)
            probs = np.clip(probs, 0, 1)
            if np.sum(probs) > 0:
                probs /= np.sum(probs)
            else:
                probs = np.ones(self.num_nodes - len(visited)) / (self.num_nodes - len(visited))

            next_node = np.random.choice(range(self.num_nodes), p=probs)
            path.append(next_node)
            visited.add(next_node)
            cost += self.graph[current, next_node]

        cost += self.graph[path[-1], path[0]]  # Return to start
        path.append(path[0])
        return path, cost

    def calculate_probabilities(self, current, visited):
        alpha = 1.0
        beta = 2.0
        probabilities = np.zeros(self.num_nodes)

        for i in range(self.num_nodes):
            if i not in visited:
                tau = self.pheromones[current, i] ** alpha
                eta = (1 / self.graph[current, i]) ** beta if self.graph[current, i] > 0 else 0
                probabilities[i] = tau * eta

        total = np.sum(probabilities)
        return probabilities / total if total > 0 else np.zeros(self.num_nodes)

    def update_pheromones(self, path, cost):
        evaporation_rate = 0.05
        self.pheromones *= (1 - evaporation_rate)
        delta_pheromone = 0.2 / cost

        for i in range(len(path) - 1):
            self.pheromones[path[i], path[i + 1]] += delta_pheromone
            self.pheromones[path[i + 1], path[i]] += delta_pheromone

class AntColonyASE:
    def __init__(self, graph):
        self.graph = graph
        self.num_nodes = graph.shape[0]
        self.pheromones = np.full_like(graph, 1.0, dtype=float)
        self.num_ants = 100

    def run(self):
        best_cost = float('inf')
        best_path = []

        for _ in range(self.num_ants):
            path, cost = self.simulate_ant()
            if cost < best_cost:
                best_cost = cost
                best_path = path

        return best_path, best_cost

    def simulate_ant(self):
        start = np.random.randint(self.num_nodes)
        visited = {start}
        path = [start]
        cost = 0

        while len(visited) < self.num_nodes:
            current = path[-1]
            probs = np.ones(self.num_nodes)
            probs[list(visited)] = 0  # Exclude visited nodes
            probs /= np.sum(probs)
            next_node = np.random.choice(range(self.num_nodes), p=probs)

            path.append(next_node)
            visited.add(next_node)
            cost += self.graph[current, next_node]

        cost += self.graph[path[-1], path[0]]  # Return to start
        path.append(path[0])
        return path, cost

class AntColonyMMAS:
    def __init__(self, graph):
        self.graph = graph
        self.num_nodes = graph.shape[0]
        self.pheromones = np.full_like(graph, 0.5, dtype=float)
        self.num_ants = 100

    def run(self):
        best_cost = float('inf')
        best_path = []
        
        for _ in range(self.num_ants):
            path, cost = self.simulate_ant()
            if cost < best_cost:
                best_cost = cost
                best_path = path

        return best_path, best_cost

    def simulate_ant(self):
        start = np.random.randint(self.num_nodes)
        visited = {start}
        path = [start]
        cost = 0

        while len(visited) < self.num_nodes:
            current = path[-1]
            probs = np.ones(self.num_nodes)
            probs[list(visited)] = 0  # Exclude visited nodes
            probs /= np.sum(probs)
            next_node = np.random.choice(range(self.num_nodes), p=probs)

            path.append(next_node)
            visited.add(next_node)
            cost += self.graph[current, next_node]

        cost += self.graph[path[-1], path[0]]  # Return to start
        path.append(path[0])
        return path, cost

class AntColonyComparator(QMainWindow):
    def __init__(self, graph, algorithms):
        super().__init__()
        self.graph = graph
        self.algorithms = algorithms

        self.setWindowTitle("Ant Colony Algorithm Comparator")
        self.setGeometry(100, 100, 1400, 900)

        layout = QVBoxLayout()
        central_widget = QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

        self.compare_button = QPushButton("Run Experiments")
        self.compare_button.clicked.connect(self.run_experiments)
        layout.addWidget(self.compare_button)

        self.figure = plt.figure(figsize=(12, 12))
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)

    def run_experiments(self):
        num_experiments = 10
        num_nodes_range = range(10, 101, 10)
        detailed_results = {"times": {algo.__name__: [] for algo in self.algorithms},
                            "costs": {algo.__name__: [] for algo in self.algorithms}}

        for num_nodes in num_nodes_range:
            graph = np.random.rand(num_nodes, num_nodes) * 10
            graph = (graph + graph.T) / 2  # Symmetric graph
            np.fill_diagonal(graph, 0)  # No self-loops

            for Algorithm in self.algorithms:
                instance = Algorithm(graph)

                times = []
                costs = []
                for _ in range(num_experiments):
                    start_time = time.time()
                    _, best_cost = instance.run()
                    elapsed_time = time.time() - start_time

                    times.append(elapsed_time)
                    costs.append(best_cost)

                detailed_results["times"][Algorithm.__name__].append(np.mean(times))
                detailed_results["costs"][Algorithm.__name__].append(np.mean(costs))

        self.plot_results(detailed_results, num_nodes_range)

    def plot_results(self, detailed_results, num_nodes_range):
        self.figure.clear()

        # Plot execution times
        ax1 = self.figure.add_subplot(211)
        for algo_name, times in detailed_results["times"].items():
            ax1.plot(num_nodes_range, times, label=algo_name.replace("AntColony", ""), linewidth=3)
        ax1.set_title("Время выполнения", fontsize=14)
        ax1.set_xlabel("Количество городо", fontsize=12)
        ax1.set_ylabel("Время", fontsize=12)
        ax1.legend(fontsize=10)
        ax1.set_position([0.1, 0.55, 0.8, 0.35])

        # Plot path costs
        ax2 = self.figure.add_subplot(212)
        for algo_name, costs in detailed_results["costs"].items():
            ax2.plot(num_nodes_range, costs, label=algo_name.replace("AntColony", ""), linewidth=3)
        ax2.set_title("Длина пути", fontsize=14)
        ax2.set_xlabel("Количество городов", fontsize=12)
        ax2.set_ylabel("Длина", fontsize=12)
        ax2.legend(fontsize=10)
        ax2.set_position([0.1, 0.1, 0.8, 0.35])

        self.canvas.draw()

def main():
    graph = np.random.rand(10, 10) * 10
    graph = (graph + graph.T) / 2  # Symmetric graph
    np.fill_diagonal(graph, 0)  # No self-loops

    algorithms = [AntColonyAS, AntColonyASE, AntColonyMMAS]

    app = QApplication(sys.argv)
    comparator = AntColonyComparator(graph, algorithms)
    comparator.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
