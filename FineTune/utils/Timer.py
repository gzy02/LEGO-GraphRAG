import time
from functools import wraps


def timer_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()  # 记录开始时间
        result = func(*args, **kwargs)  # 执行被装饰的函数
        end_time = time.time()  # 记录结束时间
        execution_time = end_time - start_time  # 计算执行时间
        print(
            f"Function '{func.__qualname__}' executed in {execution_time:.4f} seconds")
        return result
    return wrapper


class TimedClass:
    def __init__(self):
        # 为process方法添加计时装饰器
        for attr_name in dir(self):
            if attr_name == "process":
                attr_value = getattr(self, attr_name)
                if callable(attr_value):
                    setattr(self, attr_name, timer_decorator(attr_value))
