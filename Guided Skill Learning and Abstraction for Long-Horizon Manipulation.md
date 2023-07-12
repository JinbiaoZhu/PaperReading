# 【论文笔记】Guided Skill Learning and Abstraction for Long-Horizon Manipulation

更多笔记（在耕）：[这里](https://github.com/JinbiaoZhu/PaperReading)

## Abstract

1. 研究背景？

   > To assist with everyday human activities, robots must solve complex long-horizon tasks and generalize to new settings.

   为了协助日常人类活动，机器人必须解决复杂的长期任务并推广到新环境。

   > Recent deep reinforcement learning (RL) methods show promises in fully autonomous learning, but they struggle to reach long-term goals in large environments. 

   最近的深度强化学习方法在完全自主学习方面表现出了一定的优势，但在大型环境中很难达到长期目标。

   > On the other hand, Task and Motion Planning (TAMP) approaches excel at solving and generalizing across long-horizon tasks, thanks to their powerful state and action abstractions.

   另一方面，任务和动作规划（TAMP）方法在解决和推广长期任务方面表现出色，这要归功于它们强大的状态和动作抽象化。

2. 任务和动作规划（TAMP）的不足之处是什么？

   > But they assume predefined skill sets, which limits their real-world applications.

   它们需要假设预定义好的的技能集

3. 针对这个不足，本文的研究思路是什么？

   > In this work, we combine the benefits of these two paradigms and propose an integrated task planning and skill learning framework named LEAGUE.

   作者将强化学习和 TAMP 这两种范式相结合，提出一个集成任务规划和技能学习的框架 LEAGUE。

4. 作者的具体技术路线是什么？

   > LEAGUE leverages symbolic interface of a task planner to guide RL-based skill learning and creates abstract state space to enable skill reuse. 

   首先，LEAGUE 利用任务规划器的**符号接口**来指导基于强化学习的技能学习，并创建抽象状态空间以实现技能重用。

   > More importantly, LEAGUE learns manipulation skills in-situ of the task planning system,  continuously growing its capability and the set of tasks that it can solve.

   更重要的是，LEAGUE 在任务规划系统的运行中能持续学习操纵技能，不断增强其能力和可以解决的任务集合。

5. 作者的实验设置？

   three challenging simulated task domains built on the Robosuite simulation framework

6. 作者的实验指标？

   在摘要中暂无提到

7. 作者的结论？

   - LEAGUE在基线测试中表现优异。
   - 学习到的技能可以被重复使用，以加速在新任务和领域中的学习。

8. 作者是否有效果展示？

   有，但是没有开放源代码。[总体网站](https://sites.google.com/view/guidedskilllearning/)

## I. INTRODUCTION

- 基于强化学习的家庭机器人目前还有哪些不足？

  作者认为主要存在两点困难。

  > First, complex real-world tasks are often long-horizon. This requires a learning agent to explore a prohibitively large space of possible action sequences that scales exponentially with the task horizon.

  首先，复杂的现实世界任务通常具有长期的。强化学习要求智能体探索可能的行动序列空间，而该空间随任务广度增长而呈指数级增长。

  > Second, effective home robots must carry out diverse tasks in varying environments.

  其次，有效的家庭机器人必须在不同环境中执行各种任务，这就意味着必须具有泛化能力。

  所以作者的结论：A learner must **generalize** or **quickly adapt** its knowledge to new settings.

- 针对作者提出的困难，作者做了哪些综述？

  1. 课程学习：课程学习中的自动生成目标通过**中间子目标**指导学习过程，使智能体能够有效地探索并朝着长期目标不断取得进展。

     > “Automatic goal generation for reinforcement learning agents,” the 35th ICML.

  2. 分层+运动基元学习：使用预定义的**行为基元**或学习分层策略，以实现时域的决策制定。

     因为想了解技能相关的论文，正好这里列举了很多，那就展示出来~

     | 名称列表                                                     |
     | ------------------------------------------------------------ |
     | Augmenting reinforcement learning with behavior primitives for diverse manipulation tasks |
     | Accelerating reinforcement learning with learned skill priors |
     | Accelerating robotic reinforcement learning via parameterized action primitives |
     | Bottom-up skill discovery from unsegmented demonstrations for long-horizon robot manipulation |
     | Discovery of options via meta-learned subgoals               |
     | Skill-based meta-reinforcement learning （读过啦）           |
     | Reset-free lifelong learning with skill-space planning       |
     | Data-efficient hierarchical reinforcement learning           |
     | The option-critic architecture                               |
     | Opal: Offline primitive discovery for accelerating offline reinforcement learning |
     | Learning to coordinate manipulation skills via skill behavior diversification |
     | Efficient bimanual manipulation using learned task schemas   |

  3. 作者的点评：low sample effificiency, lack of interpretability, and fragile generalization; task-specifific and fall short in cross-task and cross-domain generalization.

- 作者是如何介绍自己的源方法 TAMP 呢？

  作者先阐述了 TAMP 的概念：“leverages symbolic action abstractions to enable tractable planning and strong generalization”——利用符号行动抽象实现可处理的规划和强大的泛化能力。“Specifically, the symbolic action operators divide a large planning problem into pieces that are each easier to solve.”——具体而言，符号动作运算符将一个大型规划问题分成多个更容易解决的部分。“The ‘lifted’ action abstraction allows skill reuse across tasks and even domains.”——“被重点强化/强调”的行动抽象允许在任务甚至领域之间重复使用技能。

  > Comments: 感觉也是一种拆分任务和动作的方法，相比于上一篇《Skills Regularized Task Decomposition for Multi-task Offline Reinforcement Learning》这一篇的任务/动作拆分更强调 symbolic 。

  作者给出的例子是：`grasp` 这个操作符被抽象之后，可以用在不同的任务和不同的领域上。

  ---

  作者分析了 TAMP 方法的特征：强调 symbolic 同时还强调了 complete set of skills ，需要一个完备的技能集 

  但是，这样的完备技能集导致在部署时候变得很不切实际：首先，为所有可能的任务准备技能很难；机器人必须能够按任务需求扩展其技能集。其次，在复杂或接触丰富的任务（即插入/轴孔装配）手工设计操作技能很困难，像这种 插入/轴孔装配 动作，除了末端位姿要考虑，机械臂末端力矩也要考虑，而“力”这种东西在实践性的强化学习论文中比较难设计/处理。

- 作者分析完强化和 TAMP ，作者的创新点是？

  LEAGUE (LEarning and Abstraction with GUidancE) —— an integrated task planning and skill learning framework that learns to solve and generalize across long-horizon tasks

  从装备了易于实现的技能（例如 `reach`）的任务规划器开始，LEAGUE 使用基于深度强化学习的学习器在现场持续扩展技能集。任务计划中的中间目标被指定为奖励，以便学习器获取和完善技能，并且掌握的技能用于到达新技能的初始状态。

  此外，LEAGUE 利用行动运算符的定义，即前置条件 precondition 和效果 effect，为每个学习的技能确定减少的状态空间，类似于联邦学习中信息隐藏的概念。

  > 联邦学习我比较少见，这里列举他的引用论文~
  >
  > Feudal reinforcement learning;
  >
  > Feudal networks for hierarchical reinforcement learning;

  关键思想是将与任务无关的特征抽象出来，使学习的技能具有模块化和可重用性。总体而言，这形成了一个良性循环，其中任务规划器指导技能学习和抽象，而学习器不断扩展整个系统可以执行的任务集。

  > Comments:  经过作者的介绍感觉想法是不错的，考虑到了技能集的不完备性，并通过强化学习不断扩展技能集。想到了《Skill-based Meta Reinforcement Learning》，从当前文章的视角看，《Skill-based》默认了技能集是完备的（from 离线数据集），然后强化学习不处理技能集合，而是对技能集增、删、改和查，最后实际得到的是针对当前任务的子集~~~

- 作者最后得到了什么结论？

  We show that LEAGUE is able to outperform state-of-the-art hierarchical reinforcement learning  methods by a large margin. 

  > Augmenting reinforcement learning with behavior primitives for diverse manipulation tasks

  > Comments:  它最后是跟分层强化学习比，那么就应该拿分层的视角看这个方法。

  We also highlight that our method can achieve strong generalization to new task goals and even task domains by reusing and adapting learned skills.

  LEAGUE can solve a challenging simulated coffee making task where competitive baselines fall flat.

## II. RELATED WORK

### TAMP and Learning for TAMP.

作者在此处继续补充了第一部分关于 TAMP 的阐述： TAMP方法需要**高级技能**，和**其运动学或动力学模型先验知识**，这些假设导致在手工工程操纵技能困难的领域，例如接触丰富的任务，使用 TAMP 是很困难的。

通过描述**技能前置条件**和**效果**来学习TAMP的动力学模型，也就是让模型学到 ”这个基础技能要在什么条件下实施“ 以及 ”实施这些技能后有什么效果？“。

> Comments:  学习这两部分内容其实有点像一些游戏的描述。比如说xx英雄他有一些技能，然后在英雄介绍技能描述页面就会说，这个技能在什么时候触发，会造成多少伤害/控制等等。那么对于我们玩家来说，了解/学习这些技能的**前置条件**和**造成的效果**是最关键的，而英雄技能机理，也就是怎么利用这些技能造成这样的效果，这些对于玩家来说**不必可知**。

作者列举了相关的工作，还是指出了这种方法的不足：技能集是静态的，难以在开放任务上做泛化适应。相反，我们的工作旨在逐步学习新技能，以扩展类似TAMP系统的能力。

### Curriculum for RL.

关键思想是在掌握目标任务之前，让学习代理逐渐接触到更加困难的中间任务。这些中间任务可以采用状态初始化、环境和子目标的形式。现有的课程计划侧重于教授**任务**或**特定领域**的策略。相比之下，作者的方法利用任务规划器的符号抽象来学习一系列模块化和可组合的技能。

### State and Action Abstractions.

> State abstraction allows agents to focus on task-relevant features of the environment. Action abstraction enables temporally-extended decision-making for long-horizon tasks.

状态抽象使智能体能够专注于环境的任务相关特征。行为抽象使得智能体能够进行时间上扩展的决策，以完成长期任务。

状态动作抽象，这个技术在基于技能的强化学习上看起来挺重要的，这里列举工作：

Jonschkowski explores different representation learning objectives for effective state abstraction. 

> Learning state representations with robotic priors, Autonomous Robots, 2015

Abel introduces a theory for value preserving state-action abstraction.

> Value preserving state-action abstractions, ICAIS, 2020

作者指出了不足之处：**自主发现合适的抽象**仍然是一个未解决的挑战。

作者分析了自己的模型，动作运算符的符号接口定义了前置条件和效果（动作抽象），以及与动作相关的状态子空间（状态抽象）。这些抽象使我们能够训练与任务规划器兼容的技能，并防止学习到的技能被无关对象分散注意力，从而实现跨任务和领域的技能重用。

### Hierarchical Modeling in Robot Learning. 

本文工作的关系：Our method inherits the bi-level hierarchy of a TAMP framework.

TAMP 在分层上的技术路线是：hierarchical task networks 分层任务网络、logical-geometric programming 逻辑几何规划 和 hierarchical reinforcement learning (HRL)  分层强化学习

---

关于 分层强化学习 + TAMP符号运动表征，有一些小的工作：

> Peorl: Integrating symbolic planning and hierarchical reinforcement learning for robust decision-making; 
>
> Symbolic plans as high-level instructions for reinforcement learning;

然而，这些方法需要使用表格化状态表示，因此仅限于简单的网格世界领域。

---

关于 分层强化学习 + TAMP符号运动表征 + 机器人学习，有一些小的工作（其实就是基于运动原语）：

> Augmenting reinforcement learning with behavior primitives for diverse manipulation tasks;
>
> Accelerating robotic reinforcement learning via parameterized action primitives;

作者与上述文章的相比，优势在于，他能自己不断的扩展运动原语集合。

## III. METHOD

### A. Background

#### MDP

其实就是一般的强化学习MDP框架啦

$$
<\chi,A,R(x,a),T(x^{\prime}|x,a),p(x^{0}),\gamma>
$$

$$
J=E_{x^{0},a^{0},a^{1},\cdots,a^{t-1},s^{T}\sim\pi,p(x^{0})}\big[ \sum_{t}\gamma^{t}R(x^{t},a^{t}) \big]
$$

#### Task planning space

$$
<O,\Lambda,\hat{\Psi},\hat{\Omega},G>
$$

$O$ 对象集； $\Lambda$ 对象类型的有限集合； 对于每一个对象 $o\in O$ ，都存在一个向量 $\lambda\in\Lambda$ ；向量的维度 $\text{dim}(\lambda)$ 是这个对象携带的特征信息的含量（3维的位置、rpy角度等等...）

存在以下映射： $x\in\chi$ 则存在 $x(o)\in R^{\text{dim}(\text{type}(o))}$ ，其实可以用条件概率风格这样改写公式： $x|o=x(o)\in R^{\text{dim}(\text{type}(o))}$ 这就说明，当一个对象 $o\in O$ 作用于环境的某个状态 $x\in\chi$ 时，环境状态信息会被表征为与对象 $o\in O$ 种类维度相一致的实数向量。

$\hat{\Psi}$ 描述多个对象之间的关系。谓词 $\psi\in\hat{\Psi}$ 描述了多个对象 $o\in O$ 之间的关系。每个谓词 $\psi\in\hat{\Psi}$  （例如，`Holding`）由一个对象类型元组 $(\lambda_{1},\cdots,\lambda^{m})$ 和一个二元分类器组成，确定关系是否成立。

$$
c_{\phi}:\chi\times O^{m}\rightarrow \{True,False\}
$$

其中每个下标 $o_{i}\in O$ 都有自己的向量 $\lambda_{i}\in\Lambda$  

通过替换相应的对象实体，在状态上评估谓词将产生一个基本组件（例如，`Holding(peg1)` ），其中一个被强调的组件是将映射到类型化对象变量的谓词，可以被视为占位符placeholder（例如， `Holding(?object)`）。

一个任务目标 $g \in G$ 可以表示为一组基本组件，其中符号状态 $x_{\Psi}$ 可以通过评估一组谓词 $\hat{\Psi}$ 并保留所有正向组件来获得。

![引导技能学习和抽象](https://github.com/JinbiaoZhu/PaperReading/blob/main/pic/%E5%BC%95%E5%AF%BC%E6%8A%80%E8%83%BD%E5%AD%A6%E4%B9%A0%E5%92%8C%E6%8A%BD%E8%B1%A1_1.png?raw=true)

> Comments:  分析到这里，其实也比较好理解一些了。
>
> 每一个对象 $o\in O$ 实际就是场景中的物体，比如轴、孔以及小物块这样的，那么这些物体在场景中必然有其自身的属性，于是就有了向量 $\lambda\in\Lambda$ 来描绘这个物体的属性（位姿......），因为不同物体需要使用的属性信息不一样，那么每个向量 $\lambda\in\Lambda$ 的维度也就不一样。
>
> 接下来，映射关系也好分析了。物体在MDP里面的状态中，都会映射出一个当前时刻的属性表现，也就是 $R^{\text{dim}(\text{type}(o))}$ 。接下来就是考虑谓词，这里强调了“多个目标之间的关系”，我的理解为“A目标+谓词，可以转化/实现/与目标B发生联动”。但是谓词和对象之间的关系要符合基本事实，那么就要通过“二元分类器组成，确定关系是否成立”。
>
> 后面的“占位符placeholder”不太理解，应该是让这个谓词的表达更加普适性？
>
> 最后任务目标 $g \in G$ 可以这样理解：完成某些任务（堆叠、轴孔装配）需要在每个状态下判断合适的谓词，也即是 $True$ 的谓词，并将他们收集起来做一个集合。
>
> 一个小 part 看了我半小时 5555555555555555555555555555555

#### Symbolic skill operators

这一部分主要是承接上文，当得到一组技能后，如何描述这个lifted atoms？

> Comments:  "lifted atoms" 这个意思怎么翻译成中文呢？请教了一下GPT-4的Monica，它的回答是：
>
> 1. **量子物理学**：在量子物理学中，"lifted atoms" 可能指的是原子在能量状态上的转换，即原子从一个能量较低的状态跃迁到一个能量较高的状态。
> 2. **哲学**：在哲学领域，"lifted atoms" 可能是一种隐喻，用来描述事物的本质或组成部分在更高层次上的整合。
> 3. **技术**：在技术领域，"lifted atoms" 可能是一种新技术或概念，它涉及到原子级别的操作或操控，以实现某种目的。
>
> 根据作者的描述，“A task goal $g \in G$ is represented as a set of ground **atoms**, where a symbolic state $x_{\Psi}$ can be obtained by evaluating a set of predicates $\hat{\Psi}$ and keeping all **positive ground atoms**” 
>
> 我的理解是：ground **atoms** 是谓词逻辑中的基本单位，它们描述了事物之间的关系或事物的属性。ground **atoms** 是通过将变量替换为特定常量来实例化谓词的结果。这里提到通过评估谓词集合 $\hat{\Psi}$ 来获得符号状态，这意味着需要检查每个谓词在当前状态下是否成立/合理。在评估谓词后，接下来保留所有正 ground **atoms** 。正ground **atoms** 是成立的 ground **atoms** ，即那些在当前状态下为真的 atoms。
>
> 结合 GPT-4 的 Monica 回答，以及对作者论文的解读，这里的 “atoms” 应该指的是哲学领域的“组成部分在更高层次上的整合。”，比较像“技能”这个概念。
>
> 所以接下来的 “lifted atoms” 就说是 “技能” 啦~~~

$$
<PAR,PRE,EEF^{+},EEF^{-}>
$$

PRE 表示技能的前置条件，它定义了技能可执行的条件。EFF+ 和 EFF- 描述了成功执行技能后预期的效果（条件变化）。PAR 是一个有序的参数列表，定义了 PRE、EFF+ 和 EFF- 中使用的所有对象类型。

一个基本技能运算符 $\omega$ 来表达一种抽象的技能描述的“类”（就是面向对象编程的那种“类”），然后通过 $\delta$ 赋值具体的、实际的内容来实例化这个技能运算符 $\omega$ 。给定一个任务目标，符号任务计划是一个基本运算符列表，当实例化技能成功执行时，会导致环境状态满足目标条件。

$$
\omega=<\hat{\omega},\delta>=<PRE,EEF^{+},EEF^{-}>, \delta:PAR\rightarrow O
$$

---

作者感兴趣的是学习原始操作技能，以完成由相应技能的预期效果引起的单个子目标 - 组成符号任务计划的构建块。在作者的设置中，每个技能都将有一个相应的需要学习的技能策略 $\pi$ ，而在执行过程中，属于同一技能的基本技能共享相同的技能策略。我们假设可以访问环境的谓词和运动基元（技能），并专注于有效地学习实现效果的技能。

作者提到，存在一些发明和学习谓词和运算符的工作，但这些主题超出了本工作的范围。

### B. Skill Learning and Abstraction with Operator Guidance

> ![引导技能和抽象_2](https://github.com/JinbiaoZhu/PaperReading/blob/main/pic/%E5%BC%95%E5%AF%BC%E6%8A%80%E8%83%BD%E5%AD%A6%E4%B9%A0%E5%92%8C%E6%8A%BD%E8%B1%A1_2.png?raw=true)
>
> 我们利用 TAMP 风格的技能运算符来指导技能学习（使用期望效果作为奖励）和状态抽象（强制执行与技能相关的状态空间）。这些抽象，以符号动作运算符的形式，可以很容易地引导强化学习训练的策略获得类似的能力。

> Specifically, for action abstraction, we train temporally-extended skills to reach desired effects of a skill operator by prescribing the **effect condition** as **shaped reward**. 

首先，对于动作抽象，通过将**效果条件**作为**形状奖励**来训练时延技能，以达到技能运算符的期望效果。

> Comments: 我的理解是，从传统的强化学习角度出发，评价一个“动作-状态”的价值很重要的信息就是奖励，但是这种技能描述符的方法跟强化学习不沾边，没有提供奖励信息，那么作者就考虑这个技能施加之后造成的影响（成功/失败/促进目标实现/阻碍目标实现）并将这些影响与强化学习的奖励函数相关联，让智能体可学。

> For state abstraction, we take inspiration from the idea of *information hiding* in feudal learning [35, 36] and use the precondition and effect signature of an operator to determine a *skill-relevant* state space for its corresponding learned policy. This allows the policy to be robust against domain shift and achieve generalization, especially in large environments where most elements are impertinent to a given skill.

其次，对于状态抽象，从联邦学习中的**“信息隐藏”**思想中获得启示，并使用运算符的**前置条件**和**效果表达**来确定相应学习策略的“技能相关”状态空间。这使得策略能够对领域转移具有鲁棒性，并实现泛化，特别是在大型环境中，其中大多数元素与给定技能无关。

> To further accelerate skill learning, we also leverage the existing motion planning capability of a TAMP system to augment the learned skill with a transition primitive.

最后，为了进一步加速技能学习，利用 TAMP 系统的现有运动规划能力，通过转换基元来增强学习到的技能。

#### Symbolic operators as reward guidance.

在这项工作中，SAC作为技能学习的基础。

![引导技能和抽象_3](https://github.com/JinbiaoZhu/PaperReading/blob/main/pic/%E5%BC%95%E5%AF%BC%E6%8A%80%E8%83%BD%E5%AD%A6%E4%B9%A0%E5%92%8C%E6%8A%BD%E8%B1%A1_3.png?raw=true)

#### Enhance skill reuse with feudal state abstraction. 

由于一个技能描述符的前置条件和造成效果确定了，可以确定一个与技能相关的状态空间，以进一步防止学习的策略被任务无关对象所干扰。

![引导技能和抽象_4](https://github.com/JinbiaoZhu/PaperReading/blob/main/pic/%E5%BC%95%E5%AF%BC%E6%8A%80%E8%83%BD%E5%AD%A6%E4%B9%A0%E5%92%8C%E6%8A%BD%E8%B1%A1_4.png?raw=true)

作者给了一个例子就很好理解了。作者认为，这种设计回应了之前的研究，即学习对状态施加约束，只不过这些约束是直接由任务规划器提供的。

> For the skill `Pick(peg1)`, the skill-relevant state $\dot{x}$  includes the 6D pose of `peg1` and the end-effector, the offset between the gripper and `peg1`, the joint parameters of the robot.
>
> 很明显提取任务相关的信息，像其他信息（其他物体的位姿、光照和纹理）等都被排除了。

#### Accelerate learning with transition motion primitives.

使用基于运动规划器的转换基元（transition primitives）来增强我们的策略。关键思想是，在进行基于强化学习的技能学习之前，首先使用现成的运动规划器接近与技能操作符相关的对象。该组件可以**显著加速探索**，同时仍允许系统学习闭环接触丰富的操作技能。

> Comments: 我的理解是，作者也是考虑到了强化学习训练前期比较缓慢，然后通过现成的运动规划器的转换基元来提供一些先验知识，加速探索。

### C. Integrated Task Planning and Skill Learning

1. How LEAGUE performs task planning and execution at inference time?
2. Introduce an algorithm that uses task plans as an autonomous curriculum to schedule skill learning.

#### Task planning and skill execution.

1. 对连续环境状态 $x$ 进行解析，以获取符号状态 $x_{\Psi}$ ，这为使用基本技能的符号搜索提供了可能。

2. 通过在前置条件和效果中替换对象实体来对每个技能 $\hat{\omega}\in\hat{\Omega}$ 进行实例化，从而将其应用于对象集 $O$ ，从而得到支持与符号状态一起操作的技能算子 $\omega =< PRE，EFF^{+}，EFF^{−} >$ 。只有在满足其前置条件时，才认为技能是可执行的： $PRE \in x_{\Psi}$ 。

3. 这些算子引导了一个抽象转换模型 $F(x_{\Psi}, \omega)$ ，该模型允许在符号空间中进行规划。
   
$$
x_{0}^{\Psi} = F(x_{\Psi}, \omega) = (x_{\Psi} / EFF^{−}) \cup EFF^{+}
$$

4. 使用 PDDL 来构建符号规划器，并使用 $A^{\ast}$ 搜索生成高级计划。

5. 通过生成的任务计划，按顺序调用对应的技能 $\pi_{l}$，以达到符合计划中每个技能 $\omega_{l}$ 的效果的子目标。对每个技能控制器进行跑数据，直到其满足运算符的效果或达到最大技能视野 $H$ 。

6. 为了验证技能是否成功执行，通过解析结束的环境状态 $x_l$ ，获取相应的符号状态 $x_{l}^{\Psi}$ 。只有当环境状态 $x_l$ 符合预期的效果时，才认为执行是成功的： $F(x_{l}^{\Psi-1}, ω) \in x_{l}^{\Psi}$ 。

   此外，跟踪失败的技能，以进行课程学习。

#### Task planner as autonomous curriculum

在 $N$ 次任务执行期间跟踪失败的技能，并采用严格的调度标准，即如果在 $N￥ 次任务中某个技能失败，则该技能将被安排进行课程学习。

值得注意的是，对于属于同一技能的不同技能实例（例如 `Pick(peg1)` 和 `Pick(peg2)`），共享回放缓冲区，以便相关经验可以被重复使用，从而进一步提高学习效率和泛化能力。

<img src="https://github.com/JinbiaoZhu/PaperReading/blob/main/pic/%E5%BC%95%E5%AF%BC%E6%8A%80%E8%83%BD%E5%AD%A6%E4%B9%A0%E5%92%8C%E6%8A%BD%E8%B1%A1_5.png?raw=true" alt="引导技能和抽象_5" style="zoom:50%;" />

## IV. EXPERIMENTS

### A. Experimental Setup

![引导技能和抽象_6](https://github.com/JinbiaoZhu/PaperReading/blob/main/pic/%E5%BC%95%E5%AF%BC%E6%8A%80%E8%83%BD%E5%AD%A6%E4%B9%A0%E5%92%8C%E6%8A%BD%E8%B1%A1_6.png?raw=true)

**HammerPlace**, **PegInHole**, and **MakeCoffee** on Robosuite with Mujoco

We use a Franka Emika Panda robot arm that controlled at frequency 20Hz with an operational space controller (OSC), which has 5 degrees of freedom: the position of the end-effector, the yaw angle, and the position of the gripper.

| name            | description                                                  |
| --------------- | ------------------------------------------------------------ |
| **HammerPlace** | requires the robot to place two hammers into different closed cabinets, where four skill operators are applicable in the environment: `Pick(?object)` , `Place(?object)` , `Pull(?handle)` , `Push(?handle)` . |
| **PegInHole**   | is to pick up and insert two pegs into two target holes. The applicable operators are  `Pick(?object)`  and `Insert(?object, ?hole)` . |
| **MakeCoffee**  | is the most challenging task that requires the robot to pick up a coffee pod from a closed cabinet and insert it into the holder of the coffee machine. The applicable operators are `Pick(?object)` , `Pull(?handle)` , `Push(?handle)` , `CloseLid(?machine)` and `InsertHolder(?object, ?machine)` . |

### B. Visualize Progressive Skill Learning

![技能引导和抽象_7](https://github.com/JinbiaoZhu/PaperReading/blob/main/pic/%E5%BC%95%E5%AF%BC%E6%8A%80%E8%83%BD%E5%AD%A6%E4%B9%A0%E5%92%8C%E6%8A%BD%E8%B1%A1_7.png?raw=true)

$y$ 轴显示了每个迭代中技能接收到的平均归一化奖励。每个技能的相应任务进展在图表顶部的快照中可视化。

最终，在训练结束时，所有技能都变得熟练，可以用于执行整个任务。

结果定性地表明，LEAGUE 的自动课程表在逐步学习技能以实现长期目标方面是有效的。

#### C. Quantitative Evaluation

baselines:

| name                          | description                                                  |
| ----------------------------- | ------------------------------------------------------------ |
| **MAPLE**                     | a recent state-of-the-art hierarchical RL method that learns a task controller to invoke parametric action primitives or atomic actions. |
| **a variant of our approach** | a model without the proposed state abstraction.              |
| **SAC**                       | trained with the staged dense reward.                        |

评价标准： 分数 = 已执行好的任务数 / 总的任务数

---

![引导技能和抽象_8](https://github.com/JinbiaoZhu/PaperReading/blob/main/pic/%E5%BC%95%E5%AF%BC%E6%8A%80%E8%83%BD%E5%AD%A6%E4%B9%A0%E5%92%8C%E6%8A%BD%E8%B1%A1_8_new.png?raw=true)

1.  SAC 可以通过利用奖励函数中的意外行为（即，抓住锤子的头部而不是把手）来达到第二阶段。

2. 当切换到第二个锤子时，作者的方法会出现性能下降。

   这是因为拉动左抽屉和右抽屉之间存在运动学结构差异。

   当为新目标微调 RL 策略时，也观察到这种现象，并已在文献中报道过。

3. In the most challenging MakeCoffee, LEAGUE is able to make reasonable progress but plateaus at inserting the pod and closing the cabinet and lid. Note that because this task does not facilitate in-domain skill reuse, *LEAGUE performs on par with its full-state baseline*.

#### D. Generalization to New Tasks and Domains

![引导技能和抽象_9](https://github.com/JinbiaoZhu/PaperReading/blob/main/pic/%E5%BC%95%E5%AF%BC%E6%8A%80%E8%83%BD%E5%AD%A6%E4%B9%A0%E5%92%8C%E6%8A%BD%E8%B1%A1_9.png?raw=true)

LEAGUE 在没有额外训练的情况下，推广到新的任务目标时性能下降很少，展示了强大的组合泛化能力和技能模块化。

![引导技能和抽象_10](https://github.com/JinbiaoZhu/PaperReading/blob/main/pic/%E5%BC%95%E5%AF%BC%E6%8A%80%E8%83%BD%E5%AD%A6%E4%B9%A0%E5%92%8C%E6%8A%BD%E8%B1%A1_10.png?raw=true)

We adapt skills `Pick(?object)`, `Pull(?cabinet)`, and `Push(?cabinet)` learned in the **HammerPlace** domain by slightly modifying the preconditions and effects and integrate the skills into learning the **MakeCoffee** task. 

As shown in Fig. 5, compared to learning from scratch, transferring learned skills can significantly accelerate learning (the $x$-axis is shorter) and enables the robot to solve the entire task. This highlights  LEAGUE’s strong potential for continual learning.

> Comments: 作者做的实验太成功惹 5555555555555555555555 但是在github上没有开源代码？？？

## V. CONCLUSION

未来工作进展：Relatedly, our assumptions pertaining to skill relevant state abstraction, although empirically effective, may not hold in certain cases (e.g. unintended consequences during exploration). A possible path to address both challenges is to learn skill operators with sparse transition models from experience. 作者认为这个方向的发展在于“探索中意外出现的结果”，感觉是动态场景下的泛化？然后作者给出的思路是具有更加稀疏的转移模型的技能描述符。
