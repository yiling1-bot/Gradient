import numpy as np
def gradient_descent(func, grad_func, initial_point, learning_rate=0.01, tolerance=0.000001, max_iterations=1000):

    point = np.array(initial_point, dtype=float)
    iteration = 0

    while iteration < max_iterations:
        gradient = grad_func(point)
        gradient_norm = np.linalg.norm(gradient)

        if gradient_norm < tolerance:
            print(f"梯度范数小于容忍度 {tolerance}，停止迭代。")
            break

        point = point - learning_rate * gradient
        iteration = iteration+1

    function_value = func(point)
    print(f"达到最大迭代次数 {max_iterations}。")

    return point, function_value, iteration
def gradient_ascent(func, grad_func, initial_point, learning_rate=0.01, tolerance=0.000001, max_iterations=1000):
    point = np.array(initial_point, dtype=float)
    iteration = 0

    while iteration < max_iterations:
        gradient = grad_func(point)
        gradient_norm = np.linalg.norm(gradient)

        if gradient_norm < tolerance:
            print(f"梯度范数小于容忍度 {tolerance}，停止迭代。")
            break

        point = point + learning_rate * gradient
        iteration += 1

    function_value = func(point)
    print(f"达到最大迭代次数 {max_iterations}。")

    return point, function_value, iteration
#定义默认目标函数
def objective_function(x):
    return (x - 2)**2
#定义默认目标函数的梯度
def gradient_objective_function(x):
    # 假设 x 是一个包含一个元素的 NumPy 数组
    x_scalar = x[0]
    return np.array([2 * (x_scalar - 2)])
# 运行函数
dimension = input("请输入维度：（1，2，3或更高维度数，但更高维度数没有示例函数定义代码）")
if dimension =="1":
    user_input_func = input("请输入新的函数定义代码（例如：def objective_function(x): return (x - 2)**2）：")
    user_input_grad = input("请输入新的函数梯度定义代码（例如：def gradient_objective_function(x): return np.array([2 * (x[0] - 2)])）：")
    
elif dimension == "2":
    user_input_func = input("请输入新的函数定义代码（例如：def objective_function_2d(x): return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2）：")
    user_input_grad = input("请输入新的函数梯度定义代码（例如：def gradient_objective_function_2d(x): return np.array([-2 * (1 - x[0]) - 400 * x[0] * (x[1] - x[0]**2), 200 * (x[1] - x[0]**2)])）：")
elif dimension == "3":
    user_input_func = input("请输入新的函数定义代码（例如：def objective_function(x): return (x[0] - 1)**2 + (x[1] - 2)**2 + (x[2] - 3)**2）：")
    user_input_grad = input("请输入新的函数梯度定义代码（例如：def gradient_objective_function(x): return np.array([2 * (x[0] - 1), 2 * (x[1] - 2), 2 * (x[2] - 3)])）：")
else:
    print("这是更高维的函数，请注意输入准确性。\n")
    print("请注意，高维度空间中，过大的学习率很容易导致震荡和发散，推荐学习率为0.001，容忍度1e-6，最大容忍度10000。")
    user_input_func = input("请输入函数定义代码")
    user_input_grad = input("请输入函数梯度定义代码")

try:
    exec(user_input_func)
    func_name = user_input_func.split('def ')[1].split('(')[0]
    objective_function = globals()[func_name]

    exec(user_input_grad)
    grad_name = user_input_grad.split('def ')[1].split('(')[0]
    gradient_objective_function = globals()[grad_name]
except Exception as e:
    print(f"输入的函数定义代码有误，错误信息: {e}")
    exit()
initial_point = np.array([5.0])
learning_rate = 0.1
tolerance = 1e-8
max_iterations = 1000

if dimension == "1":
    initial_point = np.array([float(input("请输入初始点（例如：5.0）："))])
    learning_rate = float(input("请输入学习率（例如0.1）："))
    tolerance = float(input("请输入容忍度（例如1e-8）："))
    max_iterations = int(input("请输入最大迭代次数："))
    # 验证 initial_point 是一维数组
    if initial_point.ndim != 1:
        raise ValueError("初始点必须是一维数组。")

    min_point, min_value, iterations_min = gradient_descent(
        objective_function,
        gradient_objective_function,
        initial_point,
        learning_rate,
        tolerance,
        max_iterations
    )
elif dimension == "2":
    initial_point_str = input("请输入初始点（例如：[1.0, 2.0]）：")
    initial_point = np.array(eval(initial_point_str), dtype=float)
    if initial_point.ndim != 1 or initial_point.shape[0] != 2:
        raise ValueError("初始点必须是包含两个元素的一维数组。")
    
    learning_rate = float(input("请输入学习率（例如0.1）："))
    tolerance = float(input("请输入容忍度（例如1e-8）："))
    max_iterations = int(input("请输入最大迭代次数："))
elif dimension == "3":
    initial_point_str = input("请输入初始点（例如：[1.0, 2.0, 3.0]）：")
    initial_point = np.array(eval(initial_point_str), dtype=float)
    if initial_point.ndim != 1 or initial_point.shape[0] != 3:
        raise ValueError("初始点必须是包含三个元素的一维数组。")
    learning_rate = float(input("请输入学习率（例如0.1）："))
    tolerance = float(input("请输入容忍度（例如1e-8）："))
    max_iterations = int(input("请输入最大迭代次数："))
else:
    initial_point_str = input("请输入初始点：")
    initial_point = np.array(eval(initial_point_str), dtype=float)
    learning_rate = float(input("请输入学习率（例如0.1）："))
    tolerance = float(input("请输入容忍度（例如1e-8）："))
    max_iterations = int(input("请输入最大迭代次数："))
if initial_point is not None:
    optimize_type = input("请选择优化类型（下降/上升）：").lower()
    if optimize_type == "下降":
        min_point, min_value, iterations_min = gradient_descent(
            objective_function,
            gradient_objective_function,
            initial_point,
            learning_rate,
            tolerance,
            max_iterations
        )
        print(f"找到的最小值点: {min_point}")
        print(f"最小值: {min_value}")
        print(f"迭代次数: {iterations_min}\n")
    elif optimize_type == "上升":
        max_point, max_value, iterations_max = gradient_ascent(
            objective_function,
            gradient_objective_function,
            initial_point,
            learning_rate,
            tolerance,
            max_iterations
        )
        print(f"找到的最大值点: {max_point}")
        print(f"最大值: {max_value}")
        print(f"迭代次数: {iterations_max}\n")
    else:
        print("无效的优化类型。")