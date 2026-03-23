def merge_sort(arr):
    """
    归并排序 (Merge Sort)
    
    原理：
    典型的分治思想 (Divide and Conquer)。
    1. 分解：将序列拆成两部分，直到每个部分只有一个元素。
    2. 递归排序：两部分各自排序好。
    3. 合并 (Merge)：将两个有序序列合并为一个大的有序序列。
    
    复杂度分析：
    - 时间复杂度: O(n log n) (性能稳定，无论数组是否有序都是 O(n log n))
    - 空间复杂度: O(n) (合并过程需要辅助数组)
    - 稳定性: 稳定排序 (Stable Sort)
    """
    if len(arr) <= 1:
        return arr
        
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    
    return merge(left, right)

def merge(left, right):
    """
    合并两组有序序列
    """
    result = []
    i = j = 0
    
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
            
    # 追加剩余部分
    result.extend(left[i:])
    result.extend(right[j:])
    return result

if __name__ == "__main__":
    test_arr = [3, 6, 8, 10, 1, 2, 1]
    print(f"Original: {test_arr}")
    print(f"Sorted: {merge_sort(test_arr)}")
