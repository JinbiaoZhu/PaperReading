# 【论文笔记】Task and Motion Planning with Large Language Models for Object Rearrangement

## Abstract

1. 研究目标：

   多目标重排是服务机器人的关键技能，这个过程中通常需要常识推理。

   实现常识性的排列需要关于物体的知识，这对机器人来说很难转移。

   大型语言模型（LLMs）是获取这种知识的一种潜在来源。

   用LLMs实现机器人多目标重排。

2. 关键问题：

   它们并不能**天生地/自然地捕捉**有关世界的、可能的、物理的排列信息。

3. 研究思路：

   提出了一个模型，LLM-GROP

4. 具体技术路线：

   使用提示来从LLMs中提取有关语义上有效的物体配置的常识知识，并使用任务和运动规划器实例化它们，以便推广到不同的场景几何形状。

   LLM-GROP使人类能够**在不同的环境中**从**自然语言命令**到**待对齐的物体** *重新排列* 。

5. 实验设置：

   设计 —— 仿真实验 + 实际实验

   指标 —— 人类评估 + 成功率 + 累计动作消耗

6. 实验结论：

   根据人类评估，提出的方法在成功率方面表现优异，同时保持可比的累计动作成本，而且超越了竞争基线，获得了最高评分。

7. 是否开源：

   [链接](https://sites.google.com/view/llm-grop)

## I. INTRODUCTION

多目标重排列涉及的有关任务：整理桌子、整理书架、放置（碗盘）到洗碗机中。

机器人需要：

1. 将**餐具物品**以**语义**上有意义的配置正确地放置
2. 避免像椅子或人这样的障碍物的情况下高效地室内导航，这些障碍物的位置事先不知道。

---

机器人移动重排列工作：

1. Semantically grounded object matching for robust robotic scene rearrangement.
2. Structformer: Learning spatial structure for language-guided semantic rearrangement of novel objects.
3. Lego-net: Learning regular rearrangements of objects in rooms.
4. Large-scale multi-object rearrangement.
5. Multi-skill mobile manipulation for object rearrangement.
6. Rearrangement planning using object-centric and robot-centric action spaces.
7. Where to relocate?: Object rearrangement inside cluttered and confined environments for robotic manipulation.
8. Reactive planning for mobile manipulation tasks in unexplored semantic environments
9. Structdiffusion Object-centric diffusion for semantic rearrangement of novel objects.
10. Visually grounded task and motion planning for mobile manipulation.

作者指出：

这些工作中的大多数需要**明确的指令**，例如将 **<font color=red>相似颜色</font>物品** 排成 **一行** 或将它们放在桌子上 **<font color=red>特定形状</font>** 的位置中。

然而，现实世界中的**用户请求**往往是**不充分的**：有许多不同的方式可以摆放桌子，这些方式并**不是同样受欢迎的**（not equally preferred）。

---

作者分析：如何确定叉子应放在盘子的左侧，刀应放在右侧？

需要相当多的**常识知识** —— LLMs

常规方法一般是使用机器学习获得语义信息，列举如下：

1. Structformer: Learning spatial structure for language-guided semantic rearrangement of novel objects.
2. Lego-net: Learning regular rearrangements of objects in rooms.
3. Structdiffusion Object-centric diffusion for semantic rearrangement of novel objects.

这些方法需要收集训练数据，这限制了它们对从事复杂服务任务的机器人的适用性。

---

**<font size=5>作者的工作：LLM-GROP —— 利用常识知识进行规划，以完成物体重新排列的任务。</font>**

1. 使用LLMs生成物体之间的符号空间关系。
2. 这些**空间关系**可以与**不同的几何空间关系**相对应，其**可行性水平**由运动规划系统评估。
3. LLMs使用不同任务运动计划的**可行性**和**效率**来优化自身，以最大化长期效用。

---

实验设置：

场景 —— 在餐厅里，移动机器人必须根据用户的指示摆放餐具。

目标 —— 机器人的任务是生成那些符合常识的对象的桌面配置，并生成一个任务运动计划来实现配置。

评估 —— 让用户评价不同的餐桌布置，以获取主观评价。

结论 —— 与现有的物体重新排列方法相比，LLM-GROP可以提高用户满意度，同时保持类似或更低的累计动作成本。

## II. RELATED WORK

### A. Object Rearrangement

#### rearrangement with known infos

两个比赛项目：[the Habitat Rearrangement Challenge](https://aihabitat.org/challenge/rearrange_2022/)、[the AI2-THOR Rearrangement Challenge](https://ai2thor.allenai.org/rearrangement/)

这些方法中的一个常见假设是，目标安排是输入的一部分，并且机器人知道物体的精确期望位置。

---

#### language-based + high-level skill

ALFRED提出了一种基于语言的多步物体重新排列任务，已经提出了许多解决方案，结合了高级技能，并最近扩展为使用LLMs作为输入。

1. Alfred: A benchmark for interpreting grounded instructions for everyday tasks.
2. A persistent spatial semantic representation for high-level natural language instruction execution.
3. Film: Following instructions in language with modular methods.
4. Prompter: Utilizing large language model prompting for a data efficient embodied instruction following.

这些方法在**非常粗糙**、**离散的层面**上操作，而不是进行运动级别和放置决策，因此无法对常识物体排列做出精细的决策。

---

#### Author's proposed methods

- 接受来自人类的一般性指示
- 有能力从LLMs抽取共识性知识来做普适性的物体重排列
- 在高层和底层运动层共同设计出决策

### B. Predicting Complex Object Arrangements

物体重排列任务不仅需要计算物体的位置，还需要普适性常识，比如在整理桌面时将叉放在左边，刀放在右边。

以往在这一领域的研究，都集中于基于**模糊指令**预测复杂的物体排列。

**作者举例**：StructFormer 是一种基于transformer的神经网络，根据自然语言指令将物体排列语义意义结构。

**作者对比**：LLM-GROP利用LLMs进行常识获取，避免了为计算物体位置而需要演示数据的需要。优化放置餐具物品计划的可行性和效率。

---

使用网页级别规模的扩散模型。

**作者举例**：DALL-E-Bot使机器人能够使用DALL-E基于文本描述生成图像，并相应地在桌面场景中安排物体。

**作者对比**：LLM-GROP使用预先训练的模型实现了零样本性能，但不限于桌子的 *单一自上而下* 视图。考虑了操作和导航的不确定性，并优化了规划的效率和可行性。

#### C. Robot Planning with Large Language Models

**作者举例**：LLMs可以通过迭代地强化prompt来用于家庭领域的任务规划。

**作者举例**：SayCan使机器人的规划能够考虑到**行动的可行性**，其中服务请求是用自然语言指定的

**作者对比**：LLM-GROP在计算语义上有效的几何配置时，同时优化了可行性和效率。

## III. THE LLM-GROP APPROACH

机器人提前了解表格的形状和位置，具备装卸餐具物品的技能。

考虑导航和操作行为中的不确定性。

1. 当机器人的目标太靠近桌子或椅子时，在规划或执行时机器人可能会在导航方面失败；
2. 当机器人与目标位置距离不足时，机器人可能会在操作方面失败；

作者主要用的方法是：LLMs用于生成**物体之间的符号空间关系**和**几何空间关系**；TAMP（任务和运动规划）用于计算**最佳**任务运动计划的路线图。

### A. Generating Symbolic Spatial Relationships

LLMs用于提取关于**放置在桌子上**的**物体之间符号空间关系**的常识知识。

这是通过利用基于模板的提示来完成的。

> <img src="https://github.com/JinbiaoZhu/PaperReading/blob/main/pic/llm_rearrangement_1.png?raw=true" alt="1" style="zoom:50%;" />

作者设计了模板提示，如果这个模板提示不带例子，那么就是 zero-shot ，反之就是 few-shot 。few-shot 提示可以确保LLM的响应遵循预定义的格式。

---

**如何防止LLMs输出语义冲突的动作？**

自动生成的结果可能会产生矛盾。

为了防止逻辑错误，研究人员开发一种基于**逻辑推理的方法**来评估**生成的候选项**与**显式符号约束**的一致性。

这种方法是在 Answer Sets Programming (ASP) 上实现的，它是一种声明式编程语言，将问题表达为一组逻辑规则和约束。

ASP可以进行递归推理，其中规则和约束可以根据其他规则和约束定义，提供了一种模块化的问题解决方法。

ASP特别适用于确定在给定上下文中规则和约束集是否为真或假。

### B. Generating Geometric Spatial Relationships

1. 选择一个坐标原点。这个原点可以是一个与桌面有明确空间关系并位于中心位置的物体。
2. 使用推荐的距离和物体之间的空间关系来确定其他物体的坐标。
3. 通过分别在水平和垂直方向上加减推荐的距离来计算物体的坐标。

---

这样做的不足是：它们不考虑物体的属性，例如形状和大小，包括桌子的限制。

设计了一种自适应采样的方法，在获得推荐的物体位置后，将物体属性纳入考虑。

涉及使用**<font color=red>二维高斯采样技术</font>**对每个物体的位置进行采样序列，以 $(x_i, y_i)$ 为均值向量，协方差矩阵描述概率密度函数的形状。

> **<font color=red>二维高斯采样技术</font>** —— 概率路线图规划器（PRMs）是一种相对较新的运动规划技术，显示出巨大的潜力。PRM的一个关键方面是用于采样自由配置空间的概率策略。在本文中，我们提出了一种新的、简单的采样策略，称为高斯采样器，它可以更好地覆盖自由配置空间的困难部分。该方法只使用基本操作，适用于许多不同的规划问题。实验表明，该技术非常高效。
>
> The gaussian sampling strategy for probabilistic roadmap planners.

---

如何增强大语言模型的精确性？

应用多个规则来确定它们的可接受性，受到**<font color=red>拒绝抽样</font>**的启发。

> **<font color=red>拒绝抽样</font>** —— 作者提出了一种从任何单峰对数凹概率密度函数中进行拒绝抽样的方法。该方法是自适应的：随着抽样的进行，拒绝包络和挤压函数收敛于密度函数。拒绝包络和挤压函数是分段指数函数，拒绝包络在先前抽样点处接触密度，挤压函数在这些接触点之间形成弧线。该技术适用于评估密度计算成本高昂的情况，特别是对于具有非共轭的贝叶斯模型的吉布斯抽样应用。我们将该技术应用于单克隆抗体反应的吉布斯抽样分析。
>
> Adaptive rejection sampling for gibbs sampling.

这些规则包括验证采样的几何位置在高层次上遵循符号关系，避免物体重叠，并确保物体保持在桌子边界内。

### C. Computing Task-Motion Plans

当物理属性确定完之后，机器人必须决定物体放置的顺序以及如何接近桌子。

机器人必须在运动和任务层之间连接，计算 $2D$ 导航目标(表示为 $loc$ )。

机器人将自己放置在物体附近进行放置，而不是站在远处并伸手去拿物体，这可能更为可取。

最近提出了一种称为 **<font color=red>GROP</font>** 的方法，用于计算最佳导航目标 $loc$ ，它可以在给定物体配置 $(x_{i,j}, y_{i,j})$ ，其中 $0\le j\le M$ 时，在可行性和效率方面为每个物体放置计算最大效用的任务-运动计划。

> **<font color=red>GROP</font>** —— 任务和运动规划（TAMP）算法旨在帮助机器人实现任务级目标，同时保持运动级别的可行性。本文重点研究涉及机器人行为需要较长时间（例如长距离导航）的TAMP领域。我们开发了一种视觉定位方法，以帮助机器人概率地评估动作可行性，并介绍了一种TAMP算法，称为GROP，它优化了可行性和效率。我们收集了一个数据集，其中包括96,000个模拟试验，用于学习符号空间关系的动作可行性评估。与竞争性TAMP基线相比，GROP表现出更高的任务完成率，同时保持较低或可比较的动作成本。除了这些广泛的模拟实验外，GROP已完全实现并在真实机器人系统上进行了测试。
>
> Visually grounded task and motion planning for mobile manipulation

## IV. EXPERIMENTS

### 1. Baseline

- Task Planning with Random Arrangement (TPRA).
- LLM-based Arrangement and Task Planning (LATP).
- GROP.

### 2. Experimental Setup

> <img src="https://github.com/JinbiaoZhu/PaperReading/blob/main/pic/llm_rearrangement_2.png?raw=true" alt="1" style="zoom:50%;" />

在模拟环境中，机器人需要从各个位置检索多个物体并将它们放置在中央桌子上。

一个障碍物(即一把椅子)将随机放置在桌子周围。

### 3. Using LLMs Engine

> <img src="https://github.com/JinbiaoZhu/PaperReading/blob/main/pic/llm_rearrangement_3.png?raw=true" alt="1" style="zoom:50%;" />

GPT-3

### 4. Rating Critera

> <img src="https://github.com/JinbiaoZhu/PaperReading/blob/main/pic/llm_rearrangement_4.png?raw=true" alt="1" style="zoom:50%;" />

从四种方法(三个基线和LMM-GROP)中生成了640张图像，用于八个任务。

每个图像都需要所有志愿者进行评估，共得到3200张图像的样本量。

志愿者在我们提供的网站上逐个查看图片，并根据评分规则对每张图片进行1到5的评分。

### 5. Results

> <img src="https://github.com/JinbiaoZhu/PaperReading/blob/main/pic/llm_rearrangement_5.png?raw=true" alt="1" style="zoom:50%;" />

与其他方法相比，LLM-GROP获得了最高的用户评分和最短的执行时间。

其他基线缺乏导航能力，无法在复杂环境中进行高效导航。

> <img src="https://github.com/JinbiaoZhu/PaperReading/blob/main/pic/llm_rearrangement_6.png?raw=true" alt="1" style="zoom:50%;" />

### 6. Real World Demonstration

> <img src="https://github.com/JinbiaoZhu/PaperReading/blob/main/pic/llm_rearrangement_7.png?raw=true" alt="1" style="zoom:50%;" />

## V. CONCLUSION AND FUTURE WORK

在未来，可能会从像 **<font color=red>M0M</font>** 这样的方法中获取更多信息，以便在未知场景中对完全未知的物体进行抓取和操作，并扩大到更广泛的放置问题集。

