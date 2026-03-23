def radix_sort(arr):
    """
    基数排序 (Radix Sort)
    
    原理：
    非比较型整数排序。将整数按位数切割成不同的数字，然后按每个位数分别比较。
    一般采用 LSD (Least Significant Digit)，即从低位到高位依次对每个数位执行一次稳定排序。
    
    复杂度分析：
    - 时间复杂度: O(d * (n + k)) (d 是最大数字的位数, k 是基数 0-9)
    - 空间复杂度: O(n + k) (需存储桶或计数数组)
    - 稳定性: 稳定排序 (依赖于底座排序的稳定性)
    """
    if not arr:
        return arr
        
    # 获取最大值确定位数
    max_num = max(arr)
    digit = 1
    
    # 逐位进行稳定排序 (这里内部使用计数排序的思想)
    while max_num // digit > 0:
        # 使用桶 (Buckets) 或计数排序
        buckets = [[] for _ in range(10)]
        for x in arr:
            idx = (x // digit) % 10
            buckets[idx].append(x)
            
        # 展平桶
        arr = []
        for b in buckets:
            arr.extend(b)
            
        digit *= 10
        
    return arr

if __name__ == "__main__":
    test_arr = [170, 45, 75, 90, 802, 24, 2, 66]
    print(f"Original: {test_arr}")
    print(f"Sorted: {radix_sort(test_arr)}")
