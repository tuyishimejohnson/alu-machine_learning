
#!/usr/bin/env python3

def matrix_shape(matrix):
    new_array = []
    rows = len(matrix)
    columns = len(matrix[0])
    print(columns)
    print(rows)

    if columns <= 2 and rows <= 2:
        new_array = [rows, columns]
    else:
        for m in matrix:
            for i in m:
                new_array = [rows, columns, len(i)]
            
            
    return new_array

result = matrix_shape([[1, 2], [3, 4]])


print(result)
