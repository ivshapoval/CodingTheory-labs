import numpy as np
import itertools


def ref(matrix: np.ndarray) -> np.ndarray:
    matrix = matrix.copy()
    m, n = matrix.shape
    row = 0

    for col in range(n):
        # Находим ненулевую строку для текущего столбца
        row_with_leading_one = np.argmax(matrix[row:m, col]) + row
        if matrix[row_with_leading_one, col] == 0:
            continue

        if row_with_leading_one != row:
            matrix[[row_with_leading_one, row]] = matrix[[row, row_with_leading_one]]

        for row_below in range(row + 1, m):
            if matrix[row_below, col] == 1:
                matrix[row_below] ^= matrix[row]

        row += 1
        if row == m:
            break

    return matrix[np.any(matrix, axis=1)]


def rref(matrix):
    rows, columns = matrix.shape
    for current_row in range(rows - 1, -1, -1):

        leading_entry_index = next((index for index, value in enumerate(matrix[current_row]) if value != 0), -1)

        if leading_entry_index != -1:
            for above_row in range(current_row - 1, -1, -1):
                if matrix[above_row, leading_entry_index] != 0:
                    matrix[above_row] = (matrix[above_row] + matrix[current_row]) % 2
    # Удаление нулевых строк
    while not np.any(matrix[-1]):
        matrix = matrix[:-1, :]
        rows -= 1

    return matrix


def get_lead_columns(matrix):
    leading_indices = []
    for row in matrix:
        leading_index = next((index for index, value in enumerate(row) if value == 1), None)
        if leading_index is not None:
            leading_indices.append(leading_index)

    return leading_indices


def delete_leading_columns(matrix, lead_indices):
    array_matrix = np.array(matrix)
    del_matrix = np.delete(array_matrix, lead_indices, axis=1)
    return del_matrix


def build_H(X, I, leading_columns):
    H_columns = X.shape[1]
    H_rows = X.shape[0] + I.shape[0]
    H_matrix = np.zeros((H_rows, H_columns), dtype=int)

    not_leading_columns = []
    for i in range (H_columns):
        if i not in leading_columns:
            not_leading_columns.append(i)

    x_counter = 0
    i_counter = 0
    for i in range(H_rows):
        if i in leading_columns:
            H_matrix[i, :] = X[x_counter, :]
            x_counter += 1
        else:
            H_matrix[i, :] = I[i_counter, :]
            i_counter += 1

    return H_matrix


def XOR():
    M = np.array([[0, 1, 0, 0],
                  [0, 0, 1, 1],
                  [1, 1, 0, 0]])

    n = M.shape[0]
    xor_results = []

    for r in range(2, n + 1):
        for indices in itertools.combinations(range(n), r):
            result = np.zeros(M.shape[1], dtype=int)
            for index in indices:
                result = np.bitwise_xor(result, M[index])
            xor_results.append(result)
    return xor_results


def binary_words_k(G, H):
    u = np.array([1, 0, 1, 1, 0])
    v = np.dot(u,G)
    v %= 2
    check = np.dot(v, H)
    check %= 2
    return u, v, check


def code_length(G):
    n = len(G[0])
    k = len(G)

    min_weight = float('inf')

    for r in range(1, k + 1):
        for comb in ((G[i] for i in range(r, len(G))) for _ in range(k)):
            combined = np.zeros(n)
            for row in comb:
                for j in range(n):
                    combined[j] = (combined[j] + row[j]) % 2

            weight = sum(combined)
            if 0 < weight < min_weight:
                min_weight = weight

    d = min_weight
    t = (d - 1)

    return n, k, d, t


def add_error(v, e1, H):
    v_e1 = (v + e1) % 2

    error = np.dot(v_e1, H)
    error %= 2
    return error


def find_e2(v, H):
    for i in range(len(v) - 1):
        for j in range(i + 1, len(v)):
            err2 = np.zeros(len(v), dtype=int)
            err2[i] = 1
            err2[j] = 1
            err2 = np.array(err2)

            # вектор ошибок
            ve2 = (v + err2) % 2

            error = np.dot(ve2, H)
            error %= 2

            a = 0
            for k in range(0, len(error)):
                if error[k] == 0:
                    a += 1
            if a == len(error):
                return err2, ve2, error


print("Шаг 1.")
G = np.array([
        [1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1],
        [0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0],
        [0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
    ])
print("Input:\nG =\n", G)

ref_matrix = ref(G)
rref_matrix = rref(ref_matrix)
print("Result:\nG* =\n", rref_matrix)

print("\nШаг 2.")
print("Input:\nG* =\n", rref_matrix)
lead_columns = get_lead_columns(rref_matrix)
print("Result:\nlead: =", lead_columns)

print("\nШаг 3.")
print("Input:\n", rref_matrix)
X = delete_leading_columns(rref_matrix, lead_columns)
print("X =\n", X)

print("\nШаг 4.")
print("X =\n", X)
I = np.identity(X.shape[1], dtype=int)
print("I =\n", I)
H = build_H(X, I, lead_columns)
print("H =\n", H)

print("\nШаг 1.4")
print("Input:\nG =\n", G)
print("\nH =\n", H)
xor_results = XOR()
print("\nXOR:", *xor_results)
u, v, check = binary_words_k(G, H)
print("\nu =", u)
print("v = u@G =", v)
print("v@H =", check)

n, k, d, t = code_length(G)
print("\nn = ", n, "\nk = ", k)
if d <= 0 or d > k:
    print("Некорректное кодовое расстояние!")
else:
    print("\nd =", d, "\nt =", t)

v = np.array([1, 0, 1, 1, 1, 0, 1, 0, 0, 1, 0])
print("v =", v)
e1 = np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0])
print("e1 =", e1)
error = add_error(v, e1, H)
print("(v + e1)@H =", error, "- error")
e2, v_e2, error = find_e2(v, H)
print("\ne2 =", e2)
print("v+e2 =", v_e2)
print("(v + e2)@H =", error, "- no error")