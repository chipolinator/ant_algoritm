import sys
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QWidget, QHBoxLayout, QLabel, QPushButton
)
from PyQt5.QtCore import QSize, Qt
from PyQt5.QtGui import QPainter, QPen, QColor
from sklearn.manifold import MDS
class AntColonyVisualization(QMainWindow):
    def __init__(self, graph, num_ants=100):
        super().__init__()
        self.graph = graph
        self.num_ants = num_ants
        self.pheromones = np.full_like(graph, 0.2, dtype=float)
        self.transition_probabilities = np.zeros_like(graph, dtype=float)
        self.current_ant = 0
        self.current_path = []

        # Настройки окна
        self.setWindowTitle("Ant Colony Visualization")
        self.setFixedSize(1100, 800)  # Фиксированный размер окна

        # Основное окно
        central_widget = QWidget()
        main_layout = QHBoxLayout()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

        # Левая часть: граф
        self.canvas = GraphCanvas(self.graph, self.pheromones)
        self.canvas.setFixedSize(600, 650)  # Фиксированный размер графика
        main_layout.addWidget(self.canvas)

        # Правая часть: таблицы и кнопка
        right_layout = QVBoxLayout()
        right_layout.setContentsMargins(10, 10, 10, 10)  # Уменьшение отступов
        right_layout.setSpacing(10)  # Уменьшение расстояния между таблицами
        main_layout.addLayout(right_layout)

        # Заголовок и таблица феромонов
        pheromone_label = QLabel("Феромоны")
        pheromone_label.setAlignment(Qt.AlignCenter)
        pheromone_label.setStyleSheet("font-size: 16px; font-weight: bold;")  # Стилизация заголовка
        right_layout.addWidget(pheromone_label)

        self.pheromone_display = PheromoneDisplay(self.pheromones)
        self.pheromone_display.setFixedSize(600, 600)  # Фиксированный размер таблицы
        right_layout.addWidget(self.pheromone_display)

        # Заголовок и таблица вероятностей перехода
        transition_label = QLabel("Вероятности перехода")
        transition_label.setAlignment(Qt.AlignCenter)
        transition_label.setStyleSheet("font-size: 16px; font-weight: bold;")  # Стилизация заголовка
        right_layout.addWidget(transition_label)

        self.transition_display = PheromoneDisplay(self.transition_probabilities)
        self.transition_display.setFixedSize(600, 600)  # Фиксированный размер таблицы
        right_layout.addWidget(self.transition_display)

        # Кнопка управления
        self.next_ant_button = QPushButton("Следующий муравей")
        self.next_ant_button.setFixedSize(200, 50)  # Фиксированный размер кнопки
        self.next_ant_button.clicked.connect(self.simulate_next_ant)
        right_layout.addWidget(self.next_ant_button, alignment=Qt.AlignCenter)


    def simulate_next_ant(self):
        if self.current_ant < self.num_ants:
            path = self.simulate_single_ant()
            self.current_ant += 1
            self.current_path = path

            # Обновление феромонов
            total_distance = sum(
                self.graph[path[i], path[i + 1]] for i in range(len(path) - 1)
            )
            for i in range(len(path) - 1):
                delta_pheromone = 0.2 / (total_distance ** 0.5)
                self.pheromones[path[i], path[i + 1]] += delta_pheromone
                self.pheromones[path[i + 1], path[i]] += delta_pheromone

            # Испарение феромонов
            evaporation_rate = 0.05
            self.pheromones *= (1 - evaporation_rate)
            self.pheromones = np.clip(self.pheromones, 0, 1)

            # Обновляем вероятности перехода
            self.update_transition_probabilities()

            # Обновляем интерфейс
            self.canvas.update_graph(self.pheromones, self.current_path, self.current_ant)
            self.pheromone_display.update_pheromones(self.pheromones)
            self.transition_display.update_pheromones(self.transition_probabilities)
        else:
            self.next_ant_button.setText("Все муравьи завершили!")

    def simulate_single_ant(self):
        num_nodes = self.graph.shape[0]
        start = np.random.randint(num_nodes)
        visited = {start}
        path = [start]

        while len(visited) < num_nodes:
            current = path[-1]
            probs = self.calculate_probabilities(current, visited)
            next_node = np.random.choice(range(num_nodes), p=probs)
            path.append(next_node)
            visited.add(next_node)

        path.append(path[0])
        return path

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

    def update_transition_probabilities(self):
        num_nodes = self.graph.shape[0]
        for i in range(num_nodes):
            self.transition_probabilities[i, :] = self.calculate_probabilities(i, set())


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

        # Рисуем рёбра
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
            painter.drawEllipse(x - 7, y - 7, 14, 14)

        # Подпись итерации
        painter.setPen(Qt.black)

        # Настройка шрифта: увеличение размера и установка жирного шрифта
        font = painter.font()
        font.setPointSize(12)  # Увеличенный размер шрифта
        font.setBold(True)  # Жирный текст
        painter.setFont(font)

        # Отрисовка текста
        painter.drawText(10, 20, f"Итерация: {self.iteration}")
        painter.end()


class PheromoneDisplay(QWidget):
    def __init__(self, pheromones):
        super().__init__()
        self.pheromones = pheromones
        self.setMinimumSize(800, 800)  # Минимальный размер виджета

    def update_pheromones(self, pheromones):
        self.pheromones = pheromones
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        rows, cols = self.pheromones.shape

        # Расчет размеров ячеек
        rect_width = self.width() / (cols + 1) * 0.8  # Ширина ячейки
        rect_height = self.height() / (rows + 1) * 0.5  # Высота ячейки

        # Увеличиваем размер шрифта
        font = painter.font()
        font.setPointSize(12)  # Размер шрифта
        painter.setFont(font)

        # Цвет цифр нумерации
        painter.setPen(QPen(QColor(0, 128, 0), 2))  # Зеленый цвет для цифр

        # Рисуем нумерацию столбцов сверху
        for j in range(cols):
            painter.drawText(
                int((j + 1) * rect_width),  # Смещаем на одну ячейку вправо
                int(0.3 * rect_height),  # Располагаем текст чуть выше первой строки
                int(rect_width),
                int(rect_height),
                Qt.AlignCenter,
                str(j + 1)  # Номер столбца
            )

        # Рисуем нумерацию строк справа
        for i in range(rows):
            painter.drawText(
                int((cols + 0.8) * rect_width),  # Располагаем текст справа от последней колонки
                int((i + 1) * rect_height),  # Смещаем на одну ячейку вниз
                int(rect_width),
                int(rect_height),
                Qt.AlignCenter,
                str(i + 1)  # Номер строки
            )

        # Рисуем разделяющие линии
        painter.setPen(QPen(Qt.black, 2))  # Черные линии
        # Горизонтальная линия под нумерацией столбцов
        painter.drawLine(
            int(rect_width),  # Начало под первым столбцом
            int(rect_height*1.1),  # Под нумерацией столбцов
            int((self.width() / (cols + 1) * 8.8)),  # Конец линии после последнего столбца
            int(rect_height*1.1), # На том же уровне
        )
        # Вертикальная линия слева от содержимого матрицы
        painter.drawLine(
            int((self.width() / (cols + 1) * 8.8)),  # Левая граница под первым столбцом
            int(rect_height*1.1),  # Начало под нумерацией строк
            int((self.width() / (cols + 1) * 8.8)),  # Левая граница
            int((rows + 1) * rect_height)  # Конец внизу
        )

        # Рисуем содержимое матрицы
        painter.setPen(Qt.black)  # Черный цвет для текста матрицы
        for i in range(rows):  # Перебираем все строки
            for j in range(cols):  # Перебираем все столбцы
                # Отображаем только элементы верхнего треугольника, включая диагональ
                if i > j:
                    continue

                # Рисуем текст феромонов
                text = f"{self.pheromones[i, j]:.2f}"  # Округление до двух знаков
                painter.drawText(
                    int((j + 1) * rect_width),  # Смещаем содержимое вправо
                    int((i + 1) * rect_height),  # Смещаем содержимое вниз
                    int(rect_width),
                    int(rect_height),
                    Qt.AlignCenter,
                    text
                )

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
