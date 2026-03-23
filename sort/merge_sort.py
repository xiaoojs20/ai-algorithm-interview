def merge_sort(arr):
    """
    归并排序 (Merge Sort)
    
    原理：分治法，拆分数组至单元素，然后合并有序子序列。
    时间复杂度: O(n log n) 稳定
    空间复杂度: O(n)
    """
    # TODO: Implement merge sort
    pass

if __name__ == "__main__":
    test_arr = [3, 6, 8, 10, 1, 2, 1]
    print(f"Original: {test_arr}")
    print(f"Sorted: {merge_sort(test_arr)}")
