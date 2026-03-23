def heapify(arr, n, i):
    # 构建大顶堆
    largest = i
    l = 2 * i + 1
    r = 2 * i + 2
    
    # 比较左子节点
    if l < n and arr[l] > arr[largest]:
        largest = l
        
    # 比较右子节点
    if r < n and arr[r] > arr[largest]:
        largest = r
        
    # 如果根不是最大的，交换并递归处理
    if largest != i:
        arr[i], arr[largest] = arr[largest], arr[i]
        heapify(arr, n, largest)

def heap_sort(arr):
    """
    堆排序 (Heap Sort)
    
    原理：
    利用堆这种数据结构。大顶堆 (Max Heap) 的根节点总是最大值。
    1. 构建大顶堆：使整个序列满足堆的性质。
    2. 最大元素下沉：将堆顶 (最大) 与最后一个元素交换。
    3. 重新调整：将剩下的 n-1 个元素重新调整为大顶堆，直到排序完成。
    
    复杂度分析：
    - 时间复杂度: O(n log n) (建堆 O(n), 调整 log n 次)
    - 空间复杂度: O(1) (原地排序)
    - 稳定性: 不稳定排序
    """
    n = len(arr)
    
    # 1. 建立大顶堆 (从最后一个非叶子节点开始向上调整)
    for i in range(n // 2 - 1, -1, -1):
        heapify(arr, n, i)
        
    # 2. 逐一提取元素 (交换堆顶到数组末尾并维持堆性质)
    for i in range(n - 1, 0, -1):
        arr[i], arr[0] = arr[0], arr[i]
        heapify(arr, i, 0)
    
    return arr

if __name__ == "__main__":
    test_arr = [3, 6, 8, 10, 1, 2, 1]
    print(f"Original: {test_arr}")
    print(f"Sorted: {heap_sort(test_arr)}")
