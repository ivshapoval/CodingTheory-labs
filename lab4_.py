import copy
import random

# import numpy as np
import numpy as np

B = [[1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1],
     [1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1],
     [0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1],
     [1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1],
     [1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1],
     [1, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1],
     [0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1],
     [0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1],
     [0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1],
     [1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1],
     [0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1],
     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0], ]


def GoleyG():
    return np.concatenate((np.eye(12, dtype=np.int16), B), axis=1)


def GoleyH():
    return np.concatenate((np.eye(12, dtype=np.int16), B), axis=0)


def get_error(w, H, B):
    s = w @ H % 2
    u1 = None
    if sum(s) <= 3:
        u1 = np.array(s)
        u1 = np.hstack((u1, np.zeros(len(s), dtype=int)))
    else:
        for i in range(len(B)):
            temp = (s + B[i]) % 2
            if sum(temp) <= 2:
                ei = np.zeros(len(s), dtype=int)
                ei[i] = 1
                u1 = np.hstack((temp, ei))
    if u1 is not None:
        u1
    else:
        sB = s @ B % 2
        if sum(sB) <= 3:
            u1 = np.hstack((np.zeros(len(s), dtype=int), sB))
        else:
            for i in range(len(B)):
                temp = (sB + B[i]) % 2
                if sum(temp) <= 2:
                    ei = np.zeros(len(s), dtype=int)
                    ei[i] = 1
                    u1 = np.hstack((ei, temp))
    return u1


def gen_error(n, num_errors):
    err = np.array([], dtype=int)
    for i in range(0, n):
        err = np.append(err, 0)
    for i in range(0, num_errors):
        flag = True
        while flag:
            j = round(random.random() * n) - 1
            if err[j] != 1:
                err[j] = 1
                flag = False
    return err


def RM(r, m):
    # Базовый случай: r = 0 -> вектор из 1 длины 2^m
    if r == 0:
        return np.ones((1, 2 ** m), dtype=int)

    # Базовый случай: r = m -> G(m-1, m) и внизу вектор [0...01]
    if r == m:
        G_m_m_1_m = RM(m - 1, m)
        bottom_row = np.zeros((1, 2 ** m), dtype=int)
        bottom_row[0, -1] = 1
        return np.vstack([G_m_m_1_m, bottom_row])

    # Рекурсивный случай: [[G(r, m-1), G(r, m-1)],[0, G(r-1, m-1)]]
    G_r_m_m_1 = RM(r, m - 1)
    G_r_m_1_m_m_1 = RM(r - 1, m - 1)

    # Верхняя часть: G(r, m-1) дублируется
    top = np.hstack([G_r_m_m_1, G_r_m_m_1])

    # Нижняя часть: нули слева и G(r-1, m-1) справа
    bottom = np.hstack([np.zeros((G_r_m_1_m_m_1.shape[0], G_r_m_1_m_m_1.shape[1]), dtype=int), G_r_m_1_m_m_1])

    # Объединение верхней и нижней части
    return np.vstack([top, bottom])


def kronecker(A, B):
    # Получаем размеры матриц
    rows_A, cols_A = A.shape
    rows_B, cols_B = B.shape

    # Инициализируем результирующую матрицу
    result = np.zeros((rows_A * rows_B, cols_A * cols_B), dtype=A.dtype)

    # Вычисляем произведение Кронекера
    for i in range(rows_A):
        for j in range(cols_A):
            result[i * rows_B:(i + 1) * rows_B, j * cols_B:(j + 1) * cols_B] = A[i, j] * B

    return result


def kronecker_H(H, m, i):
    matrix = np.eye(2 ** (m - i), dtype=int)
    matrix = kronecker(matrix, H)
    matrix = kronecker(matrix, np.eye(2 ** (i - 1)))
    return matrix


def check_error_RM(u, w, m):
    for i in range(len(w)):
        if w[i] == 0:
            w[i] = -1
    w_array = []
    H = np.array([[1, 1], [1, -1]])
    w_array.append(np.dot(w, kronecker_H(H, m, 1)))
    for i in range(2, m + 1):
        w_array.append(np.dot(w_array[-1], kronecker_H(H, m, i)))
    maximum = w_array[0][0]
    index = -1
    for i in range(len(w_array)):
        for j in range(len(w_array[i])):
            if abs(w_array[i][j]) > abs(maximum):
                index = j
                maximum = w_array[i][j]
    counter = 0
    for i in range(len(w_array)):
        for j in range(len(w_array[i])):
            if abs(w_array[i][j]) == abs(maximum):
                counter += 1
            if (counter > 1):
                print("Невозможно исправить ошибку!\n")
                return
    message = list(map(int, list(('{' + f'0:0{m}b' + '}').format(index))))
    if maximum > 0:
        message.append(1)
    else:
        message.append(0)
    print("Исправленное сообщение:", np.array(message[::-1]))
    if (not np.array_equal(u, message)):
        print("Сообщение было декодировано с ошибкой!\n")


def first_task():
    print('Часть1--------------------------------------------------------------------')

    G = GoleyG()
    print("\nG = ")
    for k in range(0, len(G)):
        print(G[k])

    H = GoleyH()
    print("\nH = ")
    for k in range(0, len(H)):
        print(H[k])

    u = np.array([1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0])
    print("\nU = ")
    print(u)

    v = np.dot(u, G)
    print("\nКодовое слово = ")
    print(v)

    e1 = gen_error(len(v), 1)
    print("\ne1 = ")
    print(e1)

    word_err1 = np.add(v, e1)
    print("\nкодовое слово с одной ошибкой = ")
    print(word_err1)

    sindrom1 = np.dot(word_err1, H) % 2
    print("\nсиндром кодового слова с одной ошибкой = ")
    print(sindrom1)

    error = get_error(word_err1, H, B)
    if error is None:
        print("Ошибка обнаружена, исправить невозможно!")
    else:
        message = (word_err1 + error) % 2
        print("\nисправленное кодовое слово c тремя ошибками = ", message)

        print("\nпроверка = ")
        print(np.dot(message, H) % 2)

    e2 = gen_error(len(v), 2)
    print("\ne2 = ")
    print(e2)

    word_err2 = np.add(v, e2)
    print("\nкодовое слово с двумя ошибками = ")
    print(word_err2)

    sindrom2 = np.dot(word_err2, H) % 2
    print("\nсиндром кодового слова с двумя ошибками = ")
    print(sindrom2)

    error = get_error(word_err2, H, B)
    if error is None:
        print("Ошибка обнаружена, исправить невозможно!")
    else:
        message = (word_err2 + error) % 2
        print("\nисправленное кодовое слово c тремя ошибками = ", message)

        print("\nпроверка = ")
        print(np.dot(message, H) % 2)

    e3 = gen_error(len(v), 3)
    print("\ne3 = ")
    print(e3)

    word_err3 = np.add(v, e3)
    print("\nкодовое слово с тремя ошибками = ")
    print(word_err3)

    sindrom3 = np.dot(word_err3, H) % 2
    print("\nсиндром кодового слова с тремя ошибками = ")
    print(sindrom3)

    error = get_error(word_err3, H, B)
    if error is None:
        print("Ошибка обнаружена, исправить невозможно!")
    else:
        message = (word_err3 + error) % 2
        print("\nисправленное кодовое слово c тремя ошибками = ", message)

        print("\nпроверка = ")
        print(np.dot(message, H) % 2)

    e4 = gen_error(len(v), 4)
    print("\ne4 = ")
    print(e4)

    word_err4 = np.add(v, e4)
    print("\nкодовое слово с четырьмя ошибками = ")
    print(word_err4)

    sindrom4 = np.dot(word_err4, H) % 2
    print("\nсиндром кодового слова с четырьмя ошибками = ")
    print(sindrom4)

    error = get_error(word_err4, H, B)
    if error is None:
        print("Ошибка обнаружена, исправить невозможно!")
    else:
        message = (word_err4 + error) % 2
        print("\nисправленное кодовое слово c тремя ошибками = ", message)

        print("\nпроверка = ")
        print(np.dot(message, H) % 2)


def second_task():
    print('Часть2--------------------------------------------------------------------')
    r = 1
    m = 3
    print("Код Рида-Маллера:(", r, ",", m, ")")

    G = RM(r, m)
    print("\nG = ", G)

    u = np.array([1, 1, 0, 0])
    print("Исходное слово\n", u)
    w = np.dot(u, G) % 2
    print("Закодированное слово:\n", w)

    err = gen_error(2 ** m, 1)
    print("Однократная ошибка:\n", err)
    word_with_error = (w + err) % 2
    print("Слово с однократной ошибкой:\n", word_with_error)

    check_error_RM(u, word_with_error, m)
    # print("Исправленное слово:\n", )
    # print("Исходное слово\n", u)

    err = gen_error(2 ** m, 2)
    print("Двухкратнаяя ошибка:\n", err)
    word_with_error = (w + err) % 2
    print("Слово с двухкратной ошибкой:\n", word_with_error)

    check_error_RM(u, word_with_error, m)

    r = 1
    m = 4
    print("Код Рида-Маллера:(", r, ",", m, ")")
    G = RM(r, m)
    print("\nG = ", G)

    u = np.array([1, 1, 0, 0, 1])
    print("Исходное слово\n", u)
    w = np.dot(u, G) % 2
    print("Закодированное слово:\n", w)

    err = gen_error(2 ** m, 1)
    print("Однократная ошибка:\n", err)
    word_with_error = (w + err) % 2
    print("Слово с однократной ошибкой:\n", word_with_error)

    check_error_RM(u, word_with_error, m)

    err = gen_error(2 ** m, 2)
    print("Двухкратнаяя ошибка:\n", err)
    word_with_error = (w + err) % 2
    print("Слово с двухкратной ошибкой:\n", word_with_error)

    check_error_RM(u, word_with_error, m)

    err = gen_error(2 ** m, 3)
    print("Трёхкратная ошибка:\n", err)
    word_with_error = (w + err) % 2
    print("Слово с трёхкратной ошибкой:\n", word_with_error)

    check_error_RM(u, word_with_error, m)

    err = gen_error(2 ** m, 4)
    print("Четырёхкратная ошибка:\n", err)
    word_with_error = (w + err) % 2
    check_error_RM(u, word_with_error, m)


first_task()
second_task()