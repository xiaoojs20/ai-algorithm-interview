def shell_sort(arr):
    """
    希尔排序 (Shell Sort)
    
    原理：
    插入排序的改进版。通过设定一个增量 (gap)，将序列分成若干子序列分别进行直接插入排序。
    随着增量逐渐减少，整个序列趋于有序，最后增量为1时完成一次完全的插入排序。
    由于初期 gap 较大，元素移动距离较远，效率比普通插入排序更高。
    
    复杂度分析：
    - 时间复杂度: 取决于增量序列 (gap sequence)
        - 最坏情况: O(n^2)
        - 最好情况: O(n log n)
        - 平均情况: 约 O(n^1.3)
    - 空间复杂度: O(1)
    - 稳定性: 不稳定排序
    """
    n = len(arr)
    gap = n // 2
    
    while gap > 0:
        # 对各分组进行插入排序
        for i in range(gap, n):
            temp = arr[i]
            j = i
            while j >= gap and arr[j - gap] > temp:
                arr[j] = arr[j - gap]
                j -= gap
            arr[j] = temp
        gap //= 2
        
    return arr

if __name__ == "__main__":
    test_arr = [12, 34, 54, 2, 3]
    print(f"Original: {test_arr}")
    print(f"Sorted: {shell_sort(test_arr)}")
