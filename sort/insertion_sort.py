def insertion_sort(arr):
    """
    插入排序 (Insertion Sort)
    
    原理：将待排序元素逐一插入已排序的前缀中。
    时间复杂度: 平均/最坏 O(n^2), 最好 O(n)
    空间复杂度: O(1)
    """
    # TODO: Implement insertion sort
    pass

if __name__ == "__main__":
    test_arr = [12, 11, 13, 5, 6]
    print(f"Original: {test_arr}")
    print(f"Sorted: {insertion_sort(test_arr)}")
