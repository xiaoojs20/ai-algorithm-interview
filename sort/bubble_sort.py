def bubble_sort(arr):
    """
    冒泡排序 (Bubble Sort)
    
    原理：
    通过重复遍历要排序的数列，一次比较两个元素，如果它们的顺序错误就交换过来。
    遍历数列的工作重复进行，直到没有再需要交换，即该数列已经排序完成。
    
    复杂度分析：
    - 时间复杂度: 
        - 平均情况: O(n^2)
        - 最坏情况: O(n^2)
        - 最好情况: O(n) (数组已有序，且通过 swapped 标志提前退出)
    - 空间复杂度: O(1) (原地排序)
    - 稳定性: 稳定排序
    """
    n = len(arr)
    # 遍历所有数组元素
    for i in range(n):
        swapped = False
        # 最后 i 个元素已经排好序
        for j in range(0, n - i - 1):
            # 比较两个相邻元素
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
                swapped = True
        
        # 如果在一趟遍历中没有发生任何交换，序列已排序
        if not swapped:
            break
            
    return arr

if __name__ == "__main__":
    test_arr = [64, 34, 25, 12, 22, 11, 90]
    print(f"Original: {test_arr}")
    print(f"Sorted: {bubble_sort(test_arr)}")
