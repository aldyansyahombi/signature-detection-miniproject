def input_array_3d(name, layers, rows, cols):
    array = []

    print(f"\nEnter elements for Array {name}:")

    for i in range(layers):
        layer = []

        print(f"Layer {i + 1}:")

        for j in range(rows):
            row = []

            for k in range(cols):
                value = int(input(f"Element [{i}][{j}][{k}]: "))
                row.append(value)

            layer.append(row)

        array.append(layer)

    return array


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

    layers = int(input("Enter number of layers: "))
    rows = int(input("Enter number of rows: "))
    cols = int(input("Enter number of columns: "))

    A = input_array_3d("A", layers, rows, cols)
    B = input_array_3d("B", layers, rows, cols)

    print("\nArray A:")
    display(A)

    print("Array B:")
    display(B)

    result = array_multiplication_3d(A, B)

    print("The product of A × B:")
    display(result)


if __name__ == "__main__":
    main()