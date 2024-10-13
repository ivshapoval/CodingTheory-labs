import copy
import random
import numpy as np

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

def generate_X(k, n):
    flag = True
    X = []
    while flag:
        flag = False
        X = []
        for i in range(0, k):
            x = []
            for i in range(0, n):
                if random.random() > 0.5:
                    x.append(1)
                else:
                    x.append(0)
            X.append(x)

        for i in range(0, k):
            if sum(X[i]) < 4:
                flag = True

        for i in range(0, k - 1):
            for j in range(i + 1, k):
                s2 = np.add(X[i], X[j])
                if sum(s2) < 3:
                    flag = True

        for i in range(0, k - 2):
            for j in range(i + 1, k - 1):
                for m in range(j + 1, k):
                    s3 = np.add(X[i], X[j])
                    s3 = np.add(s3, X[m])
                    if sum(s3) < 2:
                        flag = True

        for i in range(0, k - 3):
            for j in range(i + 1, k - 2):
                for m in range(j + 1, k - 1):
                    for l in range(m + 1, k):
                        s4 = np.add(X[i], X[j])
                        s4 = np.add(s4, X[m])
                        s4 = np.add(s4, X[l])
                        if sum(s4) < 1:
                            flag = True
    return np.array(X, dtype=int)

def first_task():
    print("________________________1 Часть________________________")
    K = 4
    N = 7

    X = np.array(([[1, 1, 1],
                    [1, 1, 0],
                    [1, 0, 1],
                    [0, 1, 1]]), dtype=int)
    print("X:\n", X)

    Ik = np.eye(4, dtype=int)
    G = np.concatenate((Ik, X), axis=1)
    G.dtype = int
    print("G: \n", G)

    Ink = np.eye(3)
    H = np.concatenate((X, Ink), axis = 0)
    print("H: \n", H)

    errors = np.eye(7)
    syndroms = dict()
    for i in range(6, -1, -1):
        syndroms[tuple(np.matmul(errors[i, :], H))] = errors[i, :]
    print("Таблица синдромов:\n", syndroms)

    word = np.array([1, 0, 0, 1])
    print("Cлово\n", word)

    coding_word = np.dot(word, G)%2
    print("Закодированное слово\n", coding_word)

    error1 = gen_error(N, 1)
    print("Error 1 = \n", error1)

    word_with_error = coding_word + error1
    word_with_error %= 2
    print("Слово с 1 ошибкой = \n", word_with_error)

    sindrom_word_with_error = np.dot(word_with_error, H) % 2
    print("Cиндром кодового слова с одной ошибкой = \n", sindrom_word_with_error)

    try: 
        err_synd = syndroms[tuple(sindrom_word_with_error)]
        correct_word = word_with_error + err_synd
        correct_word %= 2
        print("Исправленное кодовое слово c 1 ошибкой = \n", correct_word)
        if np.array_equal(correct_word, coding_word):
            print("Совпадают\n\n-----------------\n\n")
        else:
            print("Не совпадают\n\n-----------------\n\n")
    except:
        print("В таблице синдромов такого нет")
    

    print("Слово = \n", coding_word)

    error2 = gen_error(N, 2)
    print("Error 2 = \n", error2)

    word_with_error2 = coding_word + error2
    word_with_error2 %= 2
    print("Слово с 2 ошибкой = \n", word_with_error2)

    sindrom_word_with_error2 = np.dot(word_with_error2, H) % 2
    print("Cиндром кодового слова с 2 ошибками = \n", sindrom_word_with_error2)

    try: 
        err_synd2 = syndroms[tuple(sindrom_word_with_error2)]
        correct_word2 = word_with_error2 + err_synd2
        correct_word2 %= 2
        print("Исправленное кодовое слово c 2 ошибками = \n", correct_word2)
        if np.array_equal(correct_word2, coding_word):
            print("Совпадают\n\n-----------------\n\n")
        else:
            print("Не совпадают\n\n-----------------\n\n")
    except:
        print("В таблице синдромов такого нет")

def second_task():
    N = 11
    K = 4
    X = generate_X(K, N-K)
    print("X:\n", X)

    Ik = np.eye(K, dtype=int)
    G = np.concatenate((Ik, X), axis=1)
    G.dtype = int
    print("G: \n", G)

    Ink = np.eye(N-K)
    H = np.concatenate((X, Ink), axis = 0)
    print("H: \n", H)

    errors = np.eye(11, dtype=int)
    for i in range(0, 11):
        for j in range(i+1, 11):
            if i!=j:
                new_line = np.add(errors[i], errors[j])
                new_line = np.reshape(new_line, (1, 11))
                errors = np.concatenate((errors, new_line), axis = 0)
    syndroms = dict()
    for i in range(len(errors)-1, -1, -1):
        syndroms[tuple(np.dot(errors[i, :], H)%2)] = errors[i, :]
    print("Таблица синдромов с 2 ошибками:\n", syndroms)

    word = [1, 0, 0, 1]

    coding_word = np.dot(word, G)%2
    print("Закодированное слово\n", coding_word)

    error1 = gen_error(N, 1)
    print("Error 1 = \n", error1)

    word_with_error = coding_word + error1
    word_with_error %= 2
    print("Слово с 1 ошибкой = \n", word_with_error)

    sindrom_word_with_error = np.dot(word_with_error, H) % 2
    print("Cиндром кодового слова с 1 ошибкой = \n", sindrom_word_with_error)

    try: 
        err_synd = syndroms[tuple(sindrom_word_with_error)]
        correct_word = word_with_error + err_synd
        correct_word %= 2
        print("Исправленное кодовое слово c 1 ошибкой = \n", correct_word)
        if np.array_equal(correct_word, coding_word):
            print("Совпадают\n\n-----------------\n\n")
        else:
            print("Не совпадают\n\n-----------------\n\n")
    except:
        print("В таблице синдромов такого нет")
    
    


    print("Слово = \n", coding_word)

    error2 = gen_error(N, 2)
    print("Error 2 = \n", error2)

    word_with_error2 = coding_word + error2
    word_with_error2 %= 2
    print("Слово с 2 ошибкой = \n", word_with_error2)

    sindrom_word_with_error2 = np.dot(word_with_error2, H) % 2
    print("Cиндром кодового слова с 2 ошибками = \n", sindrom_word_with_error2)

    try: 
        err_synd2 = syndroms[tuple(sindrom_word_with_error2)]
        correct_word2 = word_with_error2 + err_synd2
        correct_word2 %= 2
        print("Исправленное кодовое слово c 2 ошибками = \n", correct_word2)
        if np.array_equal(correct_word2, coding_word):
            print("Совпадают\n\n-----------------\n\n")
        else:
            print("Не совпадают\n\n-----------------\n\n")
    except:
        print("В таблице синдромов такого нет")



    print("Слово = \n", coding_word)

    error3 = gen_error(N, 3)
    print("Error 3 = \n", error3)

    word_with_error3 = coding_word + error3
    word_with_error3 %= 2
    print("Слово с 3 ошибкой = \n", word_with_error3)

    sindrom_word_with_error3 = np.dot(word_with_error3, H)% 2
    print("Cиндром кодового слова с 3 ошибками = \n", sindrom_word_with_error3)

    try: 
        err_synd3 = syndroms[tuple(sindrom_word_with_error3)]
        correct_word3 = word_with_error3 + err_synd3
        correct_word3 %= 2
        print("Исправленное кодовое слово c 3 ошибками = \n", correct_word3)
        if np.array_equal(correct_word3, coding_word):
            print("Совпадают\n\n-----------------\n\n")
        else:
            print("Не совпадают\n\n-----------------\n\n")
    except:
        print("В таблице синдромов такого нет")

first_task()
second_task()