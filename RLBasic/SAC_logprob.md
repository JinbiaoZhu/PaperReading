# 强化学习 SAC算法 对数概率推导

先上原论文：

![SAC](https://github.com/JinbiaoZhu/PaperReading/blob/main/pic/sac.png?raw=true)

首先对公式 $(20)$ 做推导。

公式 $(20)$ 的数据流应该是这样的：
$$
\mathbf{s}\rightarrow \pi(\mathbf{u}|\mathbf{s}) \rightarrow \mathbf{u}\rightarrow \mathbf{a}=\tanh(\mathbf{u})\rightarrow \mathbf{a}
$$
求 $\mathbf{a}$ 的概率密度，我们先可以这样写出 $\mathbf{a}$ 的分布函数表达式：
$$
\begin{align*}
Pr_{A}(a) &= Pr(A\le a) \\
		 &\ (A表示随机变量,a表示A的某个观察值,也就是实际产生的“action”) \\
         &= Pr(\tanh(U)\le a) \\
         &\ (带入\tanh 函数,将随机变量A用U的函数来表示) \\
         &= Pr(U\le\tanh^{-1}(a)) \\
         &\ (求解Pr里面的不等式,\tanh 的反函数用\tanh^{-1}表示) \\
         &= F_{U}(\tanh^{-1}(a)) \\
         &\ (根据分布函数的定义,化简这个表达式) \\
\end{align*}\tag{1}
$$
求 $\mathbf{a}$ 的概率密度，我们可以由 $\mathbf{a}$ 的分布函数求导得到：
$$
\begin{align*}
p(A) &= \frac{\mathbf{d}Pr_{A}(\mathbf{a})}{\mathbf{d}\mathbf{a}} \\
     &= \frac{\mathbf{d}F_{U}(\tanh^{-1}(\mathbf{a}))}{\mathbf{d}\mathbf{a}} \\
     &\ (代入(1)中的表达式) \\
     &= \frac{\mathbf{d}F_{U}(\tanh^{-1}(\mathbf{a}))}{\mathbf{d}\tanh^{-1}(\mathbf{a})}\cdot\frac{\mathbf{d}\tanh^{-1}(\mathbf{a})}{\mathbf{d}\mathbf{a}} \\
     &\ (对表达式做链式求导) \\
     &= \frac{\mathbf{d}F_{U}(\mathbf{u})}{\mathbf{d}\mathbf{u}}\cdot\frac{\mathbf{d}\mathbf{u}}{\mathbf{d}\mathbf{a}} \\
     &\ (这是因为\mathbf{a}=\tanh(\mathbf{u}),则\mathbf{u}=\tanh^{-1}(\mathbf{a})) \\
     &= p(U)\cdot\big(\det\frac{\mathbf{d}\mathbf{a}}{\mathbf{d}\mathbf{u}}\big)^{-1} \\
     &\ (左边一部分根据概率密度函数的定义,右边一部分根据：) \\
     &\ (“反函数的导数等于原函数导数的倒数”) \\
     &\ (另外,向量对向量求导得到的结果是矩阵形式,因为\mathbf{a}=\tanh(\mathbf{u})) \\
     &\ (是逐个对应元素做计算,那么得到的矩阵就是一个对角阵) \\
     &\ (最后的结果是：原函数的导数的对角阵的逆) \\
     &= \mu(\mathbf{u}|\mathbf{s})\cdot \big|\det\frac{\mathbf{d}\mathbf{a}}{\mathbf{d}\mathbf{u}}\big|^{-1} \\
\end{align*}\tag{2}
$$
我们得到了论文的公式 $(20)$ ，但是后面导数的对角阵的逆还需要进一步处理。
$$
\begin{align*}
y&=\tanh(x)=\frac{\sinh(x)}{\cosh(x)}\tag{3} \\
y^{\prime}&=\frac{\cosh^{2}(x)-\sinh^{2}(x)}{\cosh^{2}(x)} \\
&= 1-\tanh^{2}(x)\tag{4} \\
&其中，[\cosh(x)]^{\prime}=\sinh(x)，[\sinh(x)]^{\prime}=\cosh(x)
\end{align*}
$$
得到了这样一个等式之后，我们可以把这个等式用到向量之间：
$$
\begin{align*}
&\ \ \ \ \big|\det\frac{\mathbf{d}\mathbf{a}}{\mathbf{d}\mathbf{u}}\big|^{-1} \\
&= \begin{vmatrix}
   \frac{\mathbf{d}a_{1}}{\mathbf{d}u_{1}} &  &  \\
    & \ddots &  \\
    &  & \frac{\mathbf{d}a_{n}}{\mathbf{d}u_{n}}
  \end{vmatrix}^{-1} \\
&= \begin{vmatrix}
   1-\tanh^{2}(u_{1}) &  &  \\
    & \ddots &  \\
    &  & 1-\tanh^{2}(u_{n})
  \end{vmatrix}^{-1} \\
&=\frac{1}{( 1-\tanh^{2}(u_{1}))\cdots(1-\tanh^{2}(u_{n}))}
\end{align*}\tag{5}
$$
最后计算对数概率密度：
$$
\begin{align*}
\log\pi(\mathbf{a}|\mathbf{s}) &= \log\mu(\mathbf{u}|\mathbf{s})+\log\frac{1}{( 1-\tanh^{2}(u_{1}))\cdots(1-\tanh^{2}(u_{n}))} \\
&\ (把公式(5)带进来并根据对数等式拆分) \\
&= \log\mu(\mathbf{u}|\mathbf{s}) - \log( 1-\tanh^{2}(u_{1}))-\cdots\log( 1-\tanh^{2}(u_{n})) \\
&\ (根据对数等式拆分) \\
&= \log\mu(\mathbf{u}|\mathbf{s}) -\sum\limits_{i=1}^{n}\log(1-\tanh^{2}(u_{i})) \\
&\ (化简一下) \\
\end{align*}\tag{6}
$$
最后我想说的是：

1. 这就是为什么在SAC的策略更新代码中，计算重采样之后的概率密度，还要再加上一串很奇怪的项。

   这一串很奇怪的项就是公式 $(6)$ 的第二项。

2. 一般是重采样之后通过 $\tanh()$ 计算实际作用于环境的动作，然后对这个动作按元素做平方计算，最后用数字1减去这个平方计算值（内含广播机制），然后与前面的重采样直接计算的概率密度相减！

3. 为什么会有 $\epsilon$ 小量？我认为这是因为公式 $(5)$ 的除法导致的。小量一般是10的-7次方，其实是忽略不计的。

OK，清楚了，撒花~~~