import numpy as np
import random

def gen_error(w, error_rate):
    error = np.zeros(len(w), dtype=int)

    if error_rate == 1:
        # Однократная ошибка — случайный индекс
        index = random.randint(0, len(w) - 1)
        error[index] = 1
    elif error_rate == 2:
        # Двухкратная ошибка в пределах 3 соседних разрядов
        index1 = random.randint(0, len(w) - 2)
        index2 = index1 + random.choice([1, 2])
        error[index1] = 1
        error[index2] = 1
    else:
        # Для всех остальных случаев (больше двух ошибок) ошибки ставятся случайно
        error_indices = random.sample(range(w.shape[0]), error_rate)
        for index in error_indices:
            error[index] = 1

    return (w + error) % 2

def create_and_correct_error(a, g, error_rate):
    print("Входное сообщение:      ", a)
    print("Порождающий полином:    ", g)

    v = np.polymul(a, g)
    v %= 2
    print("Отправленное сообщение: ", v)

    w = gen_error(v, error_rate)
    print("Сообщение с ошибкой:    ", w)

    s = polynomial_division(w, g)
    error_templates = None
    if error_rate == 1:
        error_templates = [[1]]
    else:
        error_templates = [[1, 1, 1], [1, 0, 1], [1, 1], [1]]

    idx = 0
    found = False
    for template in error_templates:
        if np.array_equal(s, template):
            found = True
    while not found:
        s = polynomial_division(polynomial_multiply(s, np.array([0, 1])), g)
        for template in error_templates:
            if np.array_equal(s, template):
                found = True
        idx += 1

    temp = np.zeros(len(w), dtype=int)
    if idx == 0:
        temp[idx] = 1
    else:
        temp[len(temp) - idx] = 1

    e = polynomial_multiply(s, temp)
    e = e[:len(w)]
    message = (w + e) % 2
    print("Исправленное сообщение: ", message)
    
    if np.array_equal(v, message):
        print("Ошибка исправлена корректно")
    else:
        print("Ошибка исправлена некорректно")

def polynomial_division(dividend, divisor):

    remainder = list(dividend)
    len_divisor = len(divisor)
    
    # Пока степень остатка >= степени делителя
    while len(remainder) >= len_divisor:
        # Находим сдвиг для делителя, чтобы выровнять его с остатком
        shift = len(remainder) - len_divisor
        for i in range(len_divisor):
            remainder[shift + i] ^= divisor[i]
        # Удаляем все последние нули из остатка, чтобы уменьшить его степень
        while len(remainder) > 0 and remainder[len(remainder) - 1] == 0:
            remainder = remainder[:len(remainder) - 1]

    return np.array(remainder)

def polynomial_multiply(A, B):
    len_A = len(A)
    len_B = len(B)
    result = np.zeros(len_A + len_B - 1, dtype=int) 
    
    for i in range(len_B):
        if B[i] == 1:
            result[i:i + len_A] ^= A.astype(int)

    return result

def task_1():
    a = np.array([1, 0, 0 ,1])
    g = np.array([1, 0, 1, 1])

    create_and_correct_error(a, g, 1)
    create_and_correct_error(a, g, 2)
    create_and_correct_error(a, g, 3)
def task_2():
    a = np.array([1, 0, 0, 1, 0, 0, 0, 1, 1])
    g = np.array([1, 0, 0, 1, 1, 1, 1])

    create_and_correct_error(a, g, 1)
    create_and_correct_error(a, g, 2)
    create_and_correct_error(a, g, 3)
    create_and_correct_error(a, g, 4)

task_1()
task_2()