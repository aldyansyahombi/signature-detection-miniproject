def array_multiplication_3d(A, B):
    if len(A) != len(B):
        raise ValueError("Array sizes do not match!")

    if len(A[0]) != len(B[0]):
        raise ValueError("Array sizes do not match!")

    if len(A[0][0]) != len(B[0][0]):
        raise ValueError("Array sizes do not match!")

    result = []

    for i in range(len(A)):
        layer = []

        for j in range(len(A[i])):
            row = []

            for k in range(len(A[i][j])):
                row.append(A[i][j][k] * B[i][j][k])

            layer.append(row)

        result.append(layer)

    return result

def display(array):
    for i, layer in enumerate(array):
        print(f"Layer {i + 1}:")
        for row in layer:
            print(row)
        print()

def main():
    print("=== 3D ARRAY MULTIPLICATION ===")

    A = [
        [
            [1, 2],
            [3, 4]
        ],
        [
            [5, 6],
            [7, 8]
        ]
    ]

    B = [
        [
            [2, 3],
            [4, 5]
        ],
        [
            [6, 7],
            [8, 9]
        ]
    ]

    print("\nArray A:")
    display(A)

    print("Array B:")
    display(B)

    result = array_multiplication_3d(A, B)

    print("The product of A x B:")
    display(result)


if __name__ == "__main__":
    main()