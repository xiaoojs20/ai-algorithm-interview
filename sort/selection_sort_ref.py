def selection_sort(arr):
    """
    选择排序 (Selection Sort)
    
    原理：
    每一次从待排序的数据元素中选出最小的一项，存放在序列的起始位置，直到全部待排序的数据元素排完。
    
    复杂度分析：
    - 时间复杂度: O(n^2) (两个嵌套循环，无论原序列是否有序)
    - 空间复杂度: O(1) (原地排序)
    - 稳定性: 不稳定排序
    """
    n = len(arr)
    for i in range(n):
        # 寻找 [i, n-1] 区间内的最小值的索引
        min_idx = i
        for j in range(i + 1, n):
            if arr[j] < arr[min_idx]:
                min_idx = j
        
        # 将最小值与起始位置交换
        arr[i], arr[min_idx] = arr[min_idx], arr[i]
    return arr

if __name__ == "__main__":
    test_arr = [64, 25, 12, 22, 11]
    print(f"Original: {test_arr}")
    print(f"Sorted: {selection_sort(test_arr)}")
