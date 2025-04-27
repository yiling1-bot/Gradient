import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # 用于三维绘图

def gradient_descent_with_history(func, grad_func, initial_point, learning_rate=0.01, tolerance=0.000001, max_iterations=1000):
    point = np.array(initial_point, dtype=float)
    history = [point.copy()]  # 记录优化过程中的点
    iteration = 0

    while iteration < max_iterations:
        gradient = grad_func(point)
        gradient_norm = np.linalg.norm(gradient)

        if gradient_norm < tolerance:
            print(f"梯度范数小于容忍度 {tolerance}，停止迭代。")
            break

        point = point - learning_rate * gradient
        history.append(point.copy())
        iteration += 1

    function_value = func(point)
    print(f"达到最大迭代次数 {max_iterations}。")

    return point, function_value, iteration, np.array(history)

def gradient_ascent_with_history(func, grad_func, initial_point, learning_rate=0.01, tolerance=0.000001, max_iterations=1000):
    point = np.array(initial_point, dtype=float)
    history = [point.copy()]  # 记录优化过程中的点
    iteration = 0

    while iteration < max_iterations:
        gradient = grad_func(point)
        gradient_norm = np.linalg.norm(gradient)

        if gradient_norm < tolerance:
            print(f"梯度范数小于容忍度 {tolerance}，停止迭代。")
            break

        point = point + learning_rate * gradient
        history.append(point.copy())
        iteration += 1

    function_value = func(point)
    print(f"达到最大迭代次数 {max_iterations}。")

    return point, function_value, iteration, np.array(history)

# 定义默认目标函数和梯度 (单变量)
def objective_function_1d(x):
    return (x - 2)**2

def gradient_objective_function_1d(x):
    return np.array([2 * (x - 2)])

# 定义默认目标函数和梯度 (双变量)
def objective_function_2d(x):
    return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2

def gradient_objective_function_2d(x):
    df_dx = -2 * (1 - x[0]) - 400 * x[0] * (x[1] - x[0]**2)
    df_dy = 200 * (x[1] - x[0]**2)
    return np.array([df_dx, df_dy])

# 定义默认目标函数和梯度 (三变量)
def objective_function_3d(x):
    return (x[0] - 1)**2 + (x[1] - 2)**2 + (x[2] - 3)**2

def gradient_objective_function_3d(x):
    df_dx = 2 * (x[0] - 1)
    df_dy = 2 * (x[1] - 2)
    df_dz = 2 * (x[2] - 3)
    return np.array([df_dx, df_dy, df_dz])

# 运行函数
dimension = input("请输入维度（1、2或3）：")

if dimension == "1":
    user_input_func = input("请输入新的函数定义代码（例如：def objective_function(x): return (x - 2)**2）：")
    user_input_grad = input("请输入新的函数梯度定义代码（例如：def gradient_objective_function(x): return np.array([2 * (x[0] - 2)])）：")
elif dimension == "2":
    user_input_func = input("请输入新的函数定义代码（例如：def objective_function(x): return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2）：")
    user_input_grad = input("请输入新的函数梯度定义代码（例如：def gradient_objective_function(x): return np.array([-2 * (1 - x[0]) - 400 * x[0] * (x[1] - x[0]**2), 200 * (x[1] - x[0]**2)])）：")
elif dimension == "3":
    user_input_func = input("请输入新的函数定义代码（例如：def objective_function(x): return (x[0] - 1)**2 + (x[1] - 2)**2 + (x[2] - 3)**2）：")
    user_input_grad = input("请输入新的函数梯度定义代码（例如：def gradient_objective_function(x): return np.array([2 * (x[0] - 1), 2 * (x[1] - 2), 2 * (x[2] - 3)])）：")
else:
    print("不支持该维度。")
    exit()

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

initial_point = None
learning_rate = 0.1
tolerance = 1e-8
max_iterations = 1000

if dimension == "1":
    initial_point = np.array([float(input("请输入初始点（例如：5.0）："))])
    learning_rate = float(input("请输入学习率（例如0.1）："))
    tolerance = float(input("请输入容忍度（例如1e-8）："))
    max_iterations = int(input("请输入最大迭代次数："))
elif dimension == "2":
    initial_point_str = input("请输入初始点（例如：[1.0, 2.0]）：")
    initial_point = np.array(eval(initial_point_str), dtype=float)
    learning_rate = float(input("请输入学习率（例如0.1）："))
    tolerance = float(input("请输入容忍度（例如1e-8）："))
    max_iterations = int(input("请输入最大迭代次数："))
elif dimension == "3":
    initial_point_str = input("请输入初始点（例如：[1.0, 2.0, 3.0]）：")
    initial_point = np.array(eval(initial_point_str), dtype=float)
    learning_rate = float(input("请输入学习率（例如0.1）："))
    tolerance = float(input("请输入容忍度（例如1e-8）："))
    max_iterations = int(input("请输入最大迭代次数："))

if initial_point is not None:
    optimize_type = input("请选择优化类型（下降/上升）：").lower()
    if optimize_type == "下降":
        final_point, min_value, iterations, history = gradient_descent_with_history(
            objective_function,
            gradient_objective_function,
            initial_point,
            learning_rate,
            tolerance,
            max_iterations
        )
        print(f"找到的最小值点: {final_point}")
        print(f"最小值: {min_value}")
        print(f"迭代次数: {iterations}\n")

        # 绘制优化过程
        plt.figure(figsize=(10, 6))
        if dimension == "1":
            x_vals = np.linspace(final_point[0] - 5, final_point[0] + 5, 100)
            y_vals = [objective_function(np.array([x])) for x in x_vals]
            plt.plot(x_vals, y_vals, label='Objective Function')
            plt.plot(history[:, 0], [objective_function(np.array([p])) for p in history], 'o-', color='red', label='Optimization Path')
            plt.scatter(final_point[0], min_value, color='green', marker='*', s=200, label='Minimum')
            plt.xlabel('x')
            plt.ylabel('f(x)')
            plt.title('1D Gradient Descent')
            plt.legend()
            plt.grid(True)
            plt.show()
        elif dimension == "2":
            x_vals = np.linspace(final_point[0] - 5, final_point[0] + 5, 100)
            y_vals = np.linspace(final_point[1] - 5, final_point[1] + 5, 100)
            X, Y = np.meshgrid(x_vals, y_vals)
            Z = np.array([objective_function(np.array([x, y])) for x, y in zip(X.flatten(), Y.flatten())]).reshape(X.shape)
            contour = plt.contour(X, Y, Z, levels=50, cmap='viridis')
            plt.colorbar(contour, label='f(x, y)')
            plt.plot(history[:, 0], history[:, 1], 'o-', color='red', label='Optimization Path')
            plt.scatter(final_point[0], final_point[1], color='green', marker='*', s=200, label='Minimum')
            plt.xlabel('x')
            plt.ylabel('y')
            plt.title('2D Gradient Descent')
            plt.legend()
            plt.grid(True)
            plt.show()
        elif dimension == "3":
            fig = plt.figure(figsize=(12, 8))
            ax = fig.add_subplot(111, projection='3d')
            ax.plot(history[:, 0], history[:, 1], history[:, 2], 'o-', color='red', label='Optimization Path')
            ax.scatter(final_point[0], final_point[1], final_point[2], color='green', marker='*', s=200, label='Minimum')
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel('z')
            ax.set_title('3D Gradient Descent')
            ax.legend()
            plt.show()

    elif optimize_type == "上升":
        final_point, max_value, iterations, history = gradient_ascent_with_history(
            objective_function,
            gradient_objective_function,
            initial_point,
            learning_rate,
            tolerance,
            max_iterations
        )
        print(f"找到的最大值点: {final_point}")
        print(f"最大值: {max_value}")
        print(f"迭代次数: {iterations}\n")

        # 绘制优化过程 (与下降类似，只需修改标签和标题)
        plt.figure(figsize=(10, 6))
        if dimension == "1":
            x_vals = np.linspace(final_point[0] - 5, final_point[0] + 5, 100)
            y_vals = [objective_function(np.array([x])) for x in x_vals]
            plt.plot(x_vals, y_vals, label='Objective Function')
            plt.plot(history[:, 0], [objective_function(np.array([p])) for p in history], 'o-', color='red', label='Optimization Path')
            plt.scatter(final_point[0], max_value, color='green', marker='*', s=200, label='Maximum')
            plt.xlabel('x')
            plt.ylabel('f(x)')
            plt.title('1D Gradient Ascent')
            plt.legend()
            plt.grid(True)
            plt.show()
        elif dimension == "2":
            x_vals = np.linspace(final_point[0] - 5, final_point[0] + 5, 100)
            y_vals = np.linspace(final_point[1] - 5, final_point[1] + 5, 100)
            X, Y = np.meshgrid(x_vals, y_vals)
            Z = np.array([objective_function(np.array([x, y])) for x, y in zip(X.flatten(), Y.flatten())]).reshape(X.shape)
            contour = plt.contour(X, Y, Z, levels=50, cmap='viridis')
            plt.colorbar(contour, label='f(x, y)')
            plt.plot(history[:, 0], history[:, 1], 'o-', color='red', label='Optimization Path')
            plt.scatter(final_point[0], final_point[1], color='green', marker='*', s=200, label='Maximum')
            plt.xlabel('x')
            plt.ylabel('y')
            plt.title('2D Gradient Ascent')
            plt.legend()
            plt.grid(True)
            plt.show()
        elif dimension == "3":
            fig = plt.figure(figsize=(12, 8))
            ax = fig.add_subplot(111, projection='3d')
            ax.plot(history[:, 0], history[:, 1], history[:, 2], 'o-', color='red', label='Optimization Path')
            ax.scatter(final_point[0], final_point[1], final_point[2], color='green', marker='*', s=200, label='Maximum')
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel('z')
            ax.set_title('3D Gradient Ascent')
            ax.legend()
            plt.show()