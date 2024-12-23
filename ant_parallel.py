import sys
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QWidget, QHBoxLayout, QLabel, QTableWidget,
    QTableWidgetItem, QPushButton
)
from PyQt5.QtCore import QSize, Qt
from PyQt5.QtGui import QPainter, QPen, QColor
from sklearn.manifold import MDS

class AntColonyVisualization(QMainWindow):
    def __init__(self, graph, num_ants=100, tau_min=0.1, tau_max=1.0):
        super().__init__()
        self.graph = graph
        self.num_ants = num_ants
        self.tau_min = tau_min
        self.tau_max = tau_max
        self.pheromones = np.full_like(graph, tau_max, dtype=float)  # Все рёбра инициализируются τmax
        self.global_best_path = None
        self.global_best_distance = float('inf')
        self.current_ant = 0
        self.paths = []  # История путей муравьев
        self.probabilities = np.zeros_like(graph)  # Вероятности перехода
        self.current_path = []  # Путь текущего муравья

        # Настройки окна
        self.setWindowTitle("Ant Colony Visualization")
        self.setGeometry(100, 100, 1400, 800)

        # Основное окно
        central_widget = QWidget()
        layout = QHBoxLayout()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

        # Левая часть: граф
        self.canvas = GraphCanvas(self.graph, self.pheromones)
        layout.addWidget(self.canvas)

        # Правая часть: таблицы и графики
        self.table_layout = QVBoxLayout()
        layout.addLayout(self.table_layout)

        # Таблица весов
        self.weight_table = QTableWidget()
        self.update_weight_table()
        self.table_layout.addWidget(QLabel("Весы графа"))
        self.table_layout.addWidget(self.weight_table)

        # Таблица вероятностей
        self.probability_table = QTableWidget()
        self.update_probability_table()
        self.table_layout.addWidget(QLabel("Вероятности переходов"))
        self.table_layout.addWidget(self.probability_table)

        # Таблица феромонов
        self.pheromone_table = QTableWidget()
        self.update_pheromone_table()
        self.table_layout.addWidget(QLabel("Уровни феромонов"))
        self.table_layout.addWidget(self.pheromone_table)

        # Управление
        self.control_layout = QVBoxLayout()
        self.next_ant_button = QPushButton("Следующий муравей")
        self.next_ant_button.clicked.connect(self.simulate_next_ant)
        self.control_layout.addWidget(self.next_ant_button)
        self.table_layout.addLayout(self.control_layout)

    def update_weight_table(self):
        num_nodes = self.graph.shape[0]
        self.weight_table.setRowCount(num_nodes)
        self.weight_table.setColumnCount(num_nodes)
        for i in range(num_nodes):
            for j in range(num_nodes):
                item = QTableWidgetItem(f"{self.graph[i, j]:.2f}")
                item.setFlags(Qt.ItemIsEnabled)  # Сделать ячейки только для чтения
                self.weight_table.setItem(i, j, item)

    def simulate_next_ant(self):
        if self.current_ant < self.num_ants:
            path, distance = self.simulate_single_ant()
            self.paths.append((path, distance))
            self.current_ant += 1
            self.current_path = path

            # Обновляем глобально лучший путь
            if distance < self.global_best_distance:
                self.global_best_distance = distance
                self.global_best_path = path

            # Обновление феромонов только для лучших путей
            self.update_pheromones()

            # Обновляем интерфейс
            self.canvas.update_graph(self.pheromones, self.current_path, self.current_ant)
            self.update_probability_table()
            self.update_pheromone_table()
        else:
            # Все муравьи завершили
            self.next_ant_button.setText("Все муравьи завершили!")
            self.current_path = []  # Убираем красный путь
            self.canvas.update_graph(self.pheromones, self.current_path, self.current_ant)

    def simulate_single_ant(self):
        num_nodes = self.graph.shape[0]
        start = np.random.randint(num_nodes)
        visited = {start}
        path = [start]

        while len(visited) < num_nodes:
            current = path[-1]
            probs = self.calculate_probabilities(current, visited)
            
            # Сохраняем вероятности для текущей строки
            self.probabilities[current, :] = probs
            
            next_node = np.random.choice(range(num_nodes), p=probs)
            path.append(next_node)
            visited.add(next_node)

        # Замыкаем путь
        path.append(path[0])
        distance = sum(self.graph[path[i], path[i + 1]] for i in range(len(path) - 1))
        return path, distance

    def calculate_probabilities(self, current, visited):
        num_nodes = self.graph.shape[0]
        alpha = 1.0  # Вес феромонов
        beta = 2.0   # Вес привлекательности

        probabilities = np.zeros(num_nodes)

        for i in range(num_nodes):
            if i not in visited:
                tau = self.pheromones[current, i] ** alpha
                eta = (1 / self.graph[current, i]) ** beta if self.graph[current, i] > 0 else 0
                probabilities[i] = tau * eta

        total = np.sum(probabilities)
        return probabilities / total if total > 0 else np.zeros(num_nodes)

    def update_pheromones(self):
        self.pheromones *= (1 - 0.05)  # Испарение феромонов

        # Обновление только для глобально лучшего пути
        if self.global_best_path:
            total_distance = sum(
                self.graph[self.global_best_path[i], self.global_best_path[i + 1]]
                for i in range(len(self.global_best_path) - 1)
            )
            delta_pheromone = 0.2 / total_distance
            for i in range(len(self.global_best_path) - 1):
                a, b = self.global_best_path[i], self.global_best_path[i + 1]
                self.pheromones[a, b] += delta_pheromone
                self.pheromones[b, a] += delta_pheromone

        # Ограничиваем значения феромонов диапазоном [tau_min, tau_max]
        self.pheromones = np.clip(self.pheromones, self.tau_min, self.tau_max)

    def update_probability_table(self):
        num_nodes = self.graph.shape[0]
        self.probability_table.setRowCount(num_nodes)
        self.probability_table.setColumnCount(num_nodes)
        for i in range(num_nodes):
            for j in range(num_nodes):
                item = QTableWidgetItem(f"{self.probabilities[i, j]:.2f}")
                self.probability_table.setItem(i, j, item)

    def update_pheromone_table(self):
        num_nodes = self.graph.shape[0]
        self.pheromone_table.blockSignals(True)
        self.pheromone_table.setRowCount(num_nodes)
        self.pheromone_table.setColumnCount(num_nodes)
        for i in range(num_nodes):
            for j in range(num_nodes):
                item = QTableWidgetItem(f"{self.pheromones[i, j]:.2f}")
                self.pheromone_table.setItem(i, j, item)
        self.pheromone_table.blockSignals(False)

# Остальной код без изменений
class GraphCanvas(QWidget):
    def __init__(self, graph, pheromones):
        super().__init__()
        self.graph = graph
        self.pheromones = pheromones
        self.current_path = []
        self.node_positions = self.generate_node_positions()
        self.iteration = 0

    def minimumSizeHint(self):
        return QSize(600, 600)

    def sizeHint(self):
        return QSize(800, 800)

    def resizeEvent(self, event):
        self.node_positions = self.generate_node_positions()
        super().resizeEvent(event)

    def generate_node_positions(self):
        num_nodes = self.graph.shape[0]
        distances = np.max(self.graph) - self.graph
        np.fill_diagonal(distances, 0)
        mds = MDS(n_components=2, dissimilarity="precomputed", random_state=42)
        coords = mds.fit_transform(distances)
        width, height = self.width(), self.height()
        scaled_coords = {
            i: (
                int((coord[0] - coords[:, 0].min()) / (coords[:, 0].max() - coords[:, 0].min()) * (width * 0.8) + width * 0.1),
                int((coord[1] - coords[:, 1].min()) / (coords[:, 1].max() - coords[:, 1].min()) * (height * 0.8) + height * 0.1)
            )
            for i, coord in enumerate(coords)
        }
        return scaled_coords

    def update_graph(self, pheromones, path, iteration):
        self.pheromones = pheromones
        self.current_path = path
        self.iteration = iteration
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        max_pheromone = np.max(self.pheromones) if np.max(self.pheromones) > 0 else 1

        # Рисуем рёбра с градиентом
        for i in range(self.graph.shape[0]):
            for j in range(i + 1, self.graph.shape[0]):
                if self.graph[i, j] > 0:
                    pheromone_level = self.pheromones[i, j] / max_pheromone
                    color = QColor(0, 0, 255, int(pheromone_level * 255))
                    painter.setPen(QPen(color, max(1, int(pheromone_level * 10))))
                    start = self.node_positions[i]
                    end = self.node_positions[j]
                    painter.drawLine(*start, *end)

        # Рисуем путь муравья
        if self.current_path:
            painter.setPen(QPen(QColor(255, 0, 0), 3, Qt.DashLine))
            for i in range(len(self.current_path) - 1):
                start = self.node_positions[self.current_path[i]]
                end = self.node_positions[self.current_path[i + 1]]
                painter.drawLine(*start, *end)

        # Рисуем узлы
        for i, (x, y) in self.node_positions.items():
            painter.setPen(QPen(Qt.black, 1))
            painter.setBrush(QColor(100, 200, 255))
            painter.drawEllipse(x - 12, y - 12, 24, 24)
            painter.setPen(Qt.black)
            painter.drawText(x - 10, y + 30, f"{i}")

        # Подпись итерации
        painter.setPen(Qt.black)
        painter.drawText(10, 20, f"Итерация: {self.iteration}")
        painter.end()


def main():
    graph = np.random.rand(10, 10) * 10
    graph = (graph + graph.T) / 2  # Симметричный граф
    np.fill_diagonal(graph, 0)  # Нет петель

    app = QApplication(sys.argv)
    window = AntColonyVisualization(graph)
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
