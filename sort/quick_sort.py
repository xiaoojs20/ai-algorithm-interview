def quick_sort(arr):
    """
    快速排序 (Quick Sort)
    
    原理：
    采用分治法 (Divide and Conquer)。通过一趟排序将要排序的数据分割成独立的两部分，
    其中一部分的所有数据都比另外一部分的所有数据都要小，然后再按此方法对这两部分数据分别进行快速排序，
    整个排序过程可以递归进行。
    
    复杂度分析：
    - 时间复杂度:
        - 平均情况: O(n log n)
        - 最坏情况: O(n^2) (当数组已经有序或完全逆序时)
    - 空间复杂度: 
        - 平均情况: O(log n) (递归栈深度)
        - 最坏情况: O(n)
    - 稳定性: 不稳定排序
    """
    if len(arr) <= 1:
        return arr
    
    # 选取基准值 (Pivot)
    pivot = arr[len(arr) // 2]
    
    # 分成三部分：小于基准、等于基准、大于基准
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    
    return quick_sort(left) + middle + quick_sort(right)

# 原地排序版本 (In-place Quick Sort)
def quick_sort_inplace(arr, low, high):
    """
    原地快速排序
    使用双指针分区，节省空间复杂度。
    """
    if low < high:
        pivot_idx = partition(arr, low, high)
        quick_sort_inplace(arr, low, pivot_idx - 1)
        quick_sort_inplace(arr, pivot_idx + 1, high)

def partition(arr, low, high):
    pivot = arr[high]
    i = low - 1  # 小于基准的元素索引
    
    for j in range(low, high):
        if arr[j] <= pivot:
            i += 1
            arr[i], arr[j] = arr[j], arr[i]
            
    arr[i + 1], arr[high] = arr[high], arr[i + 1]
    return i + 1

if __name__ == "__main__":
    test_arr = [3, 6, 8, 10, 1, 2, 1]
    print(f"Original: {test_arr}")
    print(f"Sorted: {quick_sort(test_arr)}")
