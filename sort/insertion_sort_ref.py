def insertion_sort(arr):
    """
    插入排序 (Insertion Sort)
    
    原理：
    每一步将一个待排序的记录，按其关键码值的大小插入前面已经排序的文件中适当位置，直到全部插入完为止。
    
    复杂度分析：
    - 时间复杂度: 
        - 平均情况: O(n^2)
        - 最坏情况: O(n^2)
        - 最好情况: O(n) (数组已有序)
    - 空间复杂度: O(1) (原地排序)
    - 稳定性: 稳定排序
    """
    # 从第二个元素开始，因为第一个元素默认是有序的
    for i in range(1, len(arr)):
        key = arr[i]
        
        # 将 arr[i] 移动到它在有序区间中的正确位置
        j = i - 1
        while j >= 0 and key < arr[j]:
            arr[j + 1] = arr[j] # 逐个后移
            j -= 1
        arr[j + 1] = key
        
    return arr

if __name__ == "__main__":
    test_arr = [12, 11, 13, 5, 6]
    print(f"Original: {test_arr}")
    print(f"Sorted: {insertion_sort(test_arr)}")
