import numpy as np

# 1. Подсчитать произведение ненулевых элементов на диагонали прямоугольной матрицы.


def product_of_diagonal_elements_vectorized(matrix: np.array):
    return float(np.prod(np.diagonal(matrix)[np.diagonal(matrix) != 0]))

# 2. Даны два вектора x и y. Проверить, задают ли они одно и то же мультимножество.


def are_equal_multisets_vectorized(x: np.array, y: np.array):
    return np.array_equal(np.sort(x), np.sort(y))

# 3. Найти максимальный элемент в векторе x среди элементов, перед которыми стоит нулевой.


def max_before_zero_vectorized(x: np.array):
    idx = np.flatnonzero(x[:-1] == 0)
    if idx.size == 0:
        return np.nan
    return np.max(x[idx + 1])

# 4. Операции с изображением.
# Дан трёхмерный массив, содержащий изображение, размера (height, width, numChannels), а также вектор длины numChannels.
# Сложить каналы изображения с указанными весами, и вернуть результат в виде матрицы размера (height, width).
# Считать реальное изображение можно при помощи функции scipy.misc.imread
# (если изображение не в формате png, установите пакет pillow: conda install pillow).
# Преобразуйте цветное изображение в оттенки серого, использовав коэффициенты np.array([0.299, 0.587, 0.114]).


def add_weighted_channels_vectorized(image: np.array):
    weights = np.array([0.299, 0.587, 0.114])
    return np.dot(image, weights)

# 5. Реализовать кодирование длин серий (Run-length encoding). Дан вектор x.
# Необходимо вернуть кортеж из двух векторов одинаковой длины.
# Первый содержит числа, а второй - сколько раз их нужно повторить.


def run_length_encoding_vectorized(x: np.array):
    mask = np.r_[True, x[1:] != x[:-1], True]
    indices = np.flatnonzero(mask)
    values = x[indices[:-1]]
    counts = np.diff(indices)
    return (values, counts)
