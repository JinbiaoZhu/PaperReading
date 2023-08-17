# 【论文笔记】Ground Manipulator Primitive Tasks to Executable Actions using Large Language Models

## Abstract

1. 研究目标：解决从**高层任务**到**低层机器人执行指令**的转换问题。

2. 研究问题：在分层架构的机器人系统中，如何将规划层的高级任务直接转换为执行层的低级运动指令。

3. 作者的研究思路：提出一种将**操作原语任务**与**机器人低级动作**联系起来的新方法，利用大型语言模型（LLMs）实现。

4. 具体技术路线：

   设计基于**形式化任务框架**（Task Framework Formalism）的**类编程**的提示。

   使LLMs能够为生成混合**控制位力集点**。

5. 评估方法：对几种最先进的LLMs进行评估。

## Introduction

### 分层化机器人系统

- 分层架构的发展可以追溯到斯坦福研究院的 Shakey 移动机器人。

  其系统由高级功能编程，用一阶逻辑解决规划问题，并通过低级功能指定电机命令。

- 被总结为 “感知 —— 建模 —— 规划 —— 执行” （Sense - Model - Plan - Action）范式。

  通常分层的范式是：一个规划层和一个执行层。

- 分层存在的问题是：从规划层到执行层的转换存在困难。

  **【仿生包容结构】** 存在，但是没有很多进展。

  **【使用原语任务】** 它引入了一组中间层动作库，例如 “推动物体” 或 “通过门” ，以在**高级语言逻辑**和**低级电机命令**之间进行中介转换。成果：高层规划，中层执行和低层硬件控制层。<font color=red>现有的基于原始任务的方法需要繁琐的手动规定。它们通常产生一些包含少量原始任务的库，从而限制了机器人的任务适应性和通用性。</font>

---

### 自然语言处理 for 机器人任务规划

列举了经典的几篇论文。。。真的很经典。。。

利用大型语言模型以 zero-shot 或 few-shot 的方式生成高级任务计划。

生成的抽象计划是用自然语言描述的，因此执行层无法理解。

**【设计固定的语法实现高层-底层之间的转换】** 

- 早期的方法使用严格的语法规则，将任务规范转换为线性时间逻辑形式；
- 语法解析器用于将句子指令分解为不同的语法元素，主要涉及动词和对象；
- 基于单词的统计机器翻译模型，以实现对新环境的泛化；
- 单词嵌入（Word Embedding）和循环神经网络（RNN）成为主流，并在后来引入机器人任务分解中；
- 现在，LLMs。

---

### 形式化任务框架

形式化任务框架的概念起源于对顺应运动（compliant motions）的研究（Mason 1981）。

形式化任务框架被明确地制定为**任务规划**和**力/位置控制**之间的中间转换。

<font color=red>任务框架（也称为顺应框架）是**附加在被操纵对象上**的**局部坐标系**。</font>

<font color=red>它的**平移**和**旋转方向**可以配置为**力控制**或**位置控制**。</font>

传统工作的不足之处：主要关注低级行动表示的分类，并很少考虑与高级任务命名的关联。

---

### 作者提出的方法概述

提出了一种方法，使用LLMs将文本 *操作原语任务* 转化为 *可执行的动作* 。

- 设计一个**形式化任务框架**（Task Framework Formalism）的**类编程**提示，这是一种面向对象的规范。

- 该提示将原始任务的文本作为输入，并输出任务框架中的一组位置/力矩设定点。

## Porposed Approach

### Preprocess: Identify Manipulator Primitive Tasks

将 “insert peg” 或 “open bottle” 等任务视为原始任务，通过**单个控制策略完成**而**无需更改坐标设置**。

将 “assembly GPU” 或 “make coffee” 是非原始任务的示例，必须进一步分解为多个步骤，这些步骤使用不同的控制策略和坐标。如果某个任务不是原始任务，需要进行任务分解。

### TFF-based Prompt Design

使用LLMs以形式化任务框架的方式将其转换为低级执行程序。

LLMs倾向于对**非正式问题**提供**冗长的答案**和**详细的解释**，需要设计**特定的提示**来将输出调节为结构化形式。

受到使用代码块生成机器人动作计划的启发，在计算机编程中以函数形式设计提示词。

> ![1](https://github.com/JinbiaoZhu/PaperReading/blob/main/pic/ground_primitive_1.png?raw=true)
>
> - 在函数外部，使用程序注释来指示函数是用于源函数还是目标函数。
>
>   建议使用具有不同形式化任务框架配置的多个源函数，以为LLMs提供更好的指导。
>
> - 函数名就是原语任务名。
>
> - 函数参数以任务坐标系中的六个运动方向的进行规定。
>
> - 如果一个方向不激活，只需将其值指定为0。使用 $\mathbf{translational}_z=-5\mathbf{N}$ 和 $\mathbf{angular}_z = 5 \mathbf{rad / sec}$ 来激活六个方向中的两个，控制信号是并行实现的。

### Evaluation

#### Baselines

| LLM              | temperature hyperparameter | top P | frequency penalty | presence penalty hyperparameters |
| ---------------- | -------------------------- | ----- | ----------------- | -------------------------------- |
| `GPT-3.5-turbo`  | 0                          | 1     | 0                 | 0                                |
| `GPT-4`          | 0                          | 1     | 0                 | 0                                |
| `Bard` (Web GUI) | None                       | None  | None              | None                             |
| `LLaMA-2`        | 0.01                       | 1     | 0                 | 0                                |

#### Evaluation Methods

zero-shot; one-shot; three-shot; five-shot;

> ![2](https://github.com/JinbiaoZhu/PaperReading/blob/main/pic/ground_primitive_2.png?raw=true)

#### Results

> ![3](https://github.com/JinbiaoZhu/PaperReading/blob/main/pic/ground_primitive_3.png?raw=true)

1. 在 zero-shot 测试中，没有一个LLM能够产生有效的响应。
2. 对于 `GPT-3.5-turbo` 、 `GPT-4` 和 `Bard` 模型，它们从 one-shot 开始就严格遵循定义的代码格式。
3. 对于 one-shot 提示的 `Bard` ，它的大多数目标任务规范本质上是从源任务的方向设置中复制的。
4. 失败原因分析：这些失败的任务与源任务语义相差很大，对LLMs提出了更多的挑战。
5. LLaMA-2-70B只是反复地调用源函数。它不能生成任何**新的坐标设置**或**位置/力设置点**。

## Conclusions and Discussions

- 建议构建一个更全面的提示，包括所有接触和运动类型的任务。
- 可以进一步引导 `LLaMA-2` 模型生成正确规范的操作。



