def heap_sort(arr):
    """
    堆排序 (Heap Sort)
    
    原理：利用大顶堆性质。1. 建堆 2. 交互堆顶与末尾 3. 重新成堆。
    时间复杂度: O(n log n)
    空间复杂度: O(1)
    """
    # TODO: Implement heap sort
    pass

if __name__ == "__main__":
    test_arr = [3, 6, 8, 10, 1, 2, 1]
    print(f"Original: {test_arr}")
    print(f"Sorted: {heap_sort(test_arr)}")
