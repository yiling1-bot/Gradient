# Gradient

Gradient 是一个适合初学者阅读和练习的 Python 梯度优化项目，用简单代码演示梯度下降、梯度上升和数值梯度。

## 这个项目用来做什么

这个项目的目标不是成为工业级优化库，而是帮助刚学习 Python 和数学优化的同学理解：

- 梯度是什么；
- 梯度下降怎样寻找函数最小值；
- 梯度上升怎样寻找函数最大值；
- 如果没有手写梯度，怎样用数值方法近似梯度。

## 简单理解梯度下降和梯度上升

梯度可以理解为函数增长最快的方向。

- 梯度下降：沿着梯度的反方向走，用来寻找最小值。
- 梯度上升：沿着梯度的方向走，用来寻找最大值。

例如函数 `f(x) = (x - 2)^2` 的最低点在 `x = 2`。从 `x = 0` 开始，梯度下降会一步步靠近 `2`。

## 安装

```bash
git clone https://github.com/yiling1-bot/Gradient.git
cd Gradient
pip install -e ".[test]"
```

## 可运行示例

```python
import numpy as np

from gradient_optimizer import gradient_descent


def objective(x: np.ndarray) -> float:
    return float((x[0] - 2.0) ** 2)


def gradient(x: np.ndarray) -> np.ndarray:
    return np.array([2.0 * (x[0] - 2.0)])


result = gradient_descent(
    objective,
    start=[0.0],
    gradient=gradient,
    learning_rate=0.1,
    max_iterations=200,
)

print(result.point)
print(result.value)
```

也可以直接运行 examples：

```bash
python examples/quadratic_1d.py
python examples/quadratic_2d.py
python examples/rosenbrock.py
```

## 项目结构

```text
src/gradient_optimizer/
  __init__.py       对外导出的 API
  core.py           梯度下降、梯度上升、数值梯度
  examples.py       示例函数
  cli.py            简单命令行入口
examples/           可以直接运行的入门示例
tests/              pytest 测试
.github/            GitHub Actions 和协作模板
```

## 运行测试

```bash
pip install -e ".[test]"
pytest
```

## 关于安全性

项目避免使用不安全的 `eval()` 来执行用户输入。直接执行用户输入的字符串可能会运行任意 Python 代码，例如导入模块、读取文件或执行其他危险操作。

本项目采用更安全的方式：

- 用户直接传入 Python 函数；
- 示例代码正常定义目标函数和梯度函数；
- 如果未来支持字符串表达式，应使用受限制的解析器，而不是原始 `eval()`。

## 路线图

- 增加更多适合初学者的示例。
- 增加可选绘图示例。
- 增加安全的数学表达式解析器。
- 补充更多关于数值梯度和收敛的说明。

## 贡献

欢迎贡献，但请保持改动简单、清晰，适合初学者学习。你可以：

- 改进文档；
- 添加示例；
- 添加测试；
- 简化代码。

## 许可证

本项目使用 MIT License。
