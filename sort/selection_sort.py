def selection_sort(arr):
    """
    选择排序 (Selection Sort)
    
    原理：每次从未排序部分选择最小值，放到已排序部分的末尾。
    时间复杂度: O(n^2)
    空间复杂度: O(1)
    """
    # TODO: Implement selection sort
    pass

if __name__ == "__main__":
    test_arr = [64, 25, 12, 22, 11]
    print(f"Original: {test_arr}")
    print(f"Sorted: {selection_sort(test_arr)}")
