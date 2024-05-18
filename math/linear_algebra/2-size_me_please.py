
#!/usr/bin/env python3
def matrix_shape(matrix):
    new_array = []
    while isinstance(matrix, list):
        new_array.append(len(matrix))
        matrix = matrix[0]
    return new_array

result = matrix_shape([[[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15]],
        [[16, 17, 18, 19, 20], [21, 22, 23, 24, 25], [26, 27, 28, 29, 30]]])
print(result)
