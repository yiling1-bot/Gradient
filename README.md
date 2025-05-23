# 用梯度求给定函数的极大极小值的python实现
## author:yiling1-bot
## last update:2024.04.27
#### -------以下是项目的说明和前置要求-------
该项目利用python实现了用梯度求给定函数的极大极小值
**请注意，在使用该项目之前，请确保您的python环境正常，且已经安装了numpy和matplotlib库。**

如未下载，请使用以下命令下载：
```bash
pip install numpy
pip install matplotlib
```

#### -------以下是项目的使用方法-------

跟随提示输入函数表达式，函数梯度表达式即可

-------

#### -------以下是项目的注意事项-------
- 1.在函数中，**变量名不能是x，y，z等，而是相应的替换为x[0],x[1],x[2]** 等

- 2.请注意，**函数的输入形式必须是一个完整的函数定义表达式**，例如：
```python
def objective_function(x): return (x - 2)**2）
```
而不能是：
```python
x[0]**2 + x[1]**2 + x[2]**2
```
> 这是因为我们使用了eval()函数来将字符串转换为函数对象，所以必须保证输入的字符串是一个完整的函数定义表达式。如果不是，会导致错误。
> 
- 3.请注意，在更高维度时，过大的学习率很容易导致震荡和发散。所以**建议学习率不要过大**（0.001~0.00001左右）,容忍度建议不要过大（0.00000001到0.000000001左右），迭代次数适量（1000到100000左右）

# 注意 drawingAI.py为纯AI生成的代码，可能存在错误，仅作为娱乐用途