import time
from functools import wraps


def get_func_params(func,*args,**kwargs):
    dict_params = {}
    if(len(args)>0):
        var_names = func.__code__.co_varnames
        if(len(args) == len(var_names)):
            for i in range(len(var_names)):
                dict_params.update({var_names[i]:args[i]})
    if(len(kwargs)>0):
        dict_params.update(kwargs.items())
    return dict_params
def with_method_logging(func):
    @wraps(func)
    def wrapper(*args,**kwargs):
        t_begin = time.time()
        print('%s method start at %0.4f' % (func.__name__, t_begin))
        print('LOG: Running method "%s"' % func.__name__)
        func_params = get_func_params(func, *args, **kwargs)
        print(func.__name__ + "params dict:======>" + str(func_params))
        result = func(*args, **kwargs)
        print('LOG: Method "%s" completed' % func.__name__)
        print('LOG: Method "%s" result = %s' % (func.__name__, str(result)))
        t_end = time.time()
        print('%s method end at %0.4f' % (func.__name__, t_end))
        print('%s method executed in %s ms' % (func.__name__, t_end - t_begin))
        print("===============================================================")
        return result

    return wrapper

class quicksort:

    def quick_sort(self,data):
        """快速排序"""
        if len(data) >= 2:
            mid = data[len(data) // 2]  # 选取基准值，也可以选取第一个或最后一个元素
            left, right = [], []  # 定义基准值左右两侧的列表
            data.remove(mid)  # 从原始数组中移除基准值
            for num in data:
                if num >= mid:
                    right.append(num)
                else:
                    left.append(num)
            return self.quick_sort(left) + [mid] + self.quick_sort(right)
        else:
            return data

    def getPivot(self,data, low, high):
        pivot = data[low]
        pointer = low
        for i in range(low + 1, high + 1):
            if (data[i] <= pivot):
                pointer += 1
                if (i != pivot):
                    data[i], data[pointer] = data[pointer], data[i]
        data[pointer], data[low] = data[low], data[pointer]
        return pointer

    def sortdata(self,data, low, high):
        if (low < high):
            pivot = self.getPivot(data, low, high)
            self.sortdata(data, low, pivot - 1)
            self.sortdata(data, pivot + 1, high)
        else:
            return data

    def quick_sort2(self,data):
        self.sortdata(data, 0, len(data) - 1)

    def quick_sort3(self,data):
        n = len(data)
        def sort(data, low, high):
            if (low < high):
                pivot = data[low]
                left = low
                right = high
                while (low < high):
                    while (data[low] < pivot and low < high):
                        low += 1
                    while (data[high] > pivot and low < high):
                        high -= 1
                    data[low], data[high] = data[high], data[low]
                data[left], data[low] = data[low], data[left]
                sort(data, left, low - 1)
                sort(data, high, right)
            return data

        return sort(data, 0, n - 1)

class mergesort:

    def merge_sort(self,data):
        if(len(data)<=1):
            return data
        mid = len(data)//2
        left = self.merge_sort(data[:mid])
        right = self.merge_sort(data[mid:])
        return self.merge(left,right)


    def merge(self,left,right):
        target =[]
        i=j=0
        while i <len(left) and j <len(right):
            if(left[i]<right[j]):
                target.append(left[i])
                i+=1
            else:
                target.append(right[j])
                j+=1
        target+=left[i:]
        target+=right[j:]
        return target


def printfunc(func):
    print("after cal")