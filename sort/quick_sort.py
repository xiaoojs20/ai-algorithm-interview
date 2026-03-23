def quick_sort(arr):
    """
    快速排序 (Quick Sort)
    
    原理：分治法，通过 Pivot 将数组分为两部分。
    时间复杂度: 平均 O(n log n), 最坏 O(n^2)
    空间复杂度: 平均 O(log n)
    """
    # TODO: Implement quick sort
    pass

if __name__ == "__main__":
    test_arr = [3, 6, 8, 10, 1, 2, 1]
    print(f"Original: {test_arr}")
    print(f"Sorted: {quick_sort(test_arr)}")
