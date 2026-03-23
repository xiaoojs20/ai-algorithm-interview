def radix_sort(arr):
    """
    基数排序 (Radix Sort)
    
    原理：非比较型排序。按位数进行稳定排序（LSD）。
    时间复杂度: O(d * (n + k))
    空间复杂度: O(n + k)
    """
    # TODO: Implement radix sort
    pass

if __name__ == "__main__":
    test_arr = [170, 45, 75, 90, 802, 24, 2, 66]
    print(f"Original: {test_arr}")
    print(f"Sorted: {radix_sort(test_arr)}")
