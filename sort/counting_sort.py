def counting_sort(arr):
    """
    计数排序 (Counting Sort)
    
    原理：非比较排序，通过统计每个值出现的次数进行排序。
    时间复杂度: O(n + k)
    空间复杂度: O(k)
    """
    # TODO: Implement counting sort
    pass

if __name__ == "__main__":
    test_arr = [4, 2, 2, 8, 3, 3, 1]
    print(f"Original: {test_arr}")
    print(f"Sorted: {counting_sort(test_arr)}")
