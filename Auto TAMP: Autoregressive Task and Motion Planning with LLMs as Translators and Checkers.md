# 【论文笔记】Auto TAMP: Autoregressive Task and Motion Planning with LLMs as Translators and Checkers

更多笔记（在耕）：[这里](https://github.com/JinbiaoZhu/PaperReading)

本文已开源！[这里](https://github.com/yongchao98/AutoTAMP)

## Abstract

1. 研究背景

   > For effective human-robot interaction, robots need to understand, plan, and execute complex, long-horizon tasks described by natural language. The recent and remarkable advances in large language models (LLMs) have shown promise for translating natural language into robot action sequences for complex tasks.

   为了实现有效的人机交互，机器人需要理解、规划和执行由自然语言描述的复杂、长期任务。最近大型语言模型（LLMs）的显著进展表明，它们有望将自然语言转化为机器人执行复杂任务的动作序列。

   > However, many existing approaches either translate the natural language directly into robot trajectories, or factor the inference process by decomposing language into task sub-goals, then relying on a motion planner to execute each sub-goal.

   现有的方法要么直接将**自然语言翻译为机器人轨迹**，要么通过将自然语言**分解为任务子目标**来**分解推理过程**，然后**依靠运动规划器**执行每个子目标。

2. 研究问题

   > When complex environmental and temporal constraints are involved, inference over planning tasks must be performed jointly with motion plans using traditional task and-motion planning (TAMP) algorithms, making such factorization untenable.

   当涉及到复杂的环境和时间限制时，必须使用传统的**任务和运动规划（TAMP）算法**联合**执行规划好的任务**和**运动规划的推理**，使得这种子任务分解变得不可行。

3. 研究思路

   > Rather than using LLMs to directly plan task sub-goals, we instead perform few-shot translation from natural language task descriptions to an **intermediary task representation** that can then be consumed by a TAMP algorithm to jointly solve the task and motion plan.

   作者的思路是：不直接使用LLMs来规划任务子目标，而是将自然语言任务描述进行**few-shot翻译**，转换为**中介任务表示**，然后再由TAMP算法使用，以共同解决任务和运动规划。

4. 具体技术路线

   > To improve translation, we automatically detect and correct both syntactic and semantic errors via autoregressive re-prompting, resulting in significant improvements in task completion.

   为了提高翻译质量，作者通过**自回归重新提示**自动检测和纠正句法和语义错误，从而显著提高任务完成度。

5. 效果

   > We show that our approach outperforms several methods using LLMs as planners in complex task domains.

   在复杂的任务领域中，作者的方法优于使用**LLMs作为规划器**的几种方法。

6. 关键词

   Large Language Models, Task and Motion Planning, Human-Robot Interaction

7. 是否开源？

   是，[这里](https://github.com/yongchao98/AutoTAMP)。

---

## 1 Introduction

> Robots not only need to reason about the task in the environment and find a satisfying sequence of actions but also verify the feasibility of executing those actions according to the robot’s motion capabilities.

机器人不仅需要对环境中的任务进行推理并找到令人满意的动作序列，还需要根据机器人的运动能力验证执行这些动作的可行性。这个问题被称为**任务和动作规划（Task And Motion Planning, TAMP）**，已经有大量的研究致力于寻找高效的算法。传统的解决方案依赖于**在专门的规划表示（方法/规则）中指定（实例）任务**，以适应这样的算法。

---

虽然这种任务规范化方法相当成功，但直接使用这些表示（方法/规则）需要培训和经验，使它们成为**非专业用户**的不良候选方案。作为替代方案，**自然语言**提供了一种直观和灵活的任务描述范式。预训练的LLMs已经在许多与语言相关的任务上表现出很好的性能，并且已经有一系列研究关于预训练的LLMs在TAMP中的使用。

---

早期的研究使用LLMs作为**直接的**任务规划器，即基于一组**自然语言指令**生成**子任务序列**，取得了良好结果，但由于**缺乏反馈**和**验证子任务序列的可执行性**而受到限制。

> 《Language models as zero-shot planners: Extracting actionable knowledge for embodied agents》
>
> 使用LLMs生成整个子任务序列，而**不检查可执行性**。

其他研究通过将**子任务连接到控制策略可负担的功能**（关键词：affordance），提供机器人行动的**环境反馈**，并将**动作可行性检查与LLMs动作建议交错出现/发生**来解决可执行性问题。

> 《 Do as i can, not as i say: Grounding language in robotic affordances》可负担的功能；
>
> 可以**自回归地**提示LLM以在序列中的**前一个子任务**的条件下生成**每个后续子任务**。可以通过将**语言模型似然性**与**可行性似然性**相结合，从前 $K(=5\text{, in this work})$ 个候选项中选择下一个子任务。
>
> 《Inner monologue: Embodied reasoning through planning with language models》环境反馈；
>
> 《Text2motion: From natural language instructions to feasible plans》可行性检查和LLMs动作建议；

这些方法在处理**各种任务复杂性时**存在困难，例如**时间依赖的多步动作**，**动作序列优化**，**任务约束**等。已提出的框架将规划问题分解，并使用LLMs单独推断任务计划和运动计划，但是两者的优化过程需要同时执行。

> For instance, when the task is `‘reach all locations via the shortest path’` , the order of places to be visited  (task planning) depends on the geometry of the environment and the related motion optimization.

> Comments: 作者给的这个例子比上一篇文章好一些。作者的意思是说，对于“使用最短路径方法遍历所有地点”这样的问题，当使用语言模型设计子任务和运动方案时，都要考虑到环境的几何信息，这作为一种“任务约束”反过来影响了任务规划和运动方案设计的调优。那么，需要在提示词里面强调这些几何信息，也可以称为限制，来让规划更加合理。
>
> Comments: 这个任务，emmmmmm，使用强化学习不就能实现了？？？

LLMs似乎无法直接生成轨迹，这可能与复杂的**空间和数值推理能力**有关。

> 《 ChatGPT empowered long step robot control in various environments: A case application》
>
> 《Large language models still can’t plan (a benchmark for llms on planning and reasoning about change)》

---

值得注意的是，经典TAMP中使用的任务表示，例如PDDL或Temporal Logics，具有**足够的表达能力**，可以**指定**任务复杂性，从而**现有的规划算法**可以**找到并验证满足这些规范的**动作序列。

为了充分利用自然语言的**用户友好性**和现有TAMP算法的功能，使用LLMs将**高级任务描述**转换为**形式化任务规范**来解决问题。

> 这样的工作之前就已经有了。
>
> 《 Llm+ p: Empowering large language models with optimal planning proficiency》
>
> 这篇论文介绍了LLM+P框架，该框架将经典规划器的优势与LLMs相结合，以便在自然语言输入的情况下，快速找到解决问题的正确或最优计划。通过对一系列基准问题的实验，发现LLM+P能够为大多数问题提供最优解决方案，而LLMs则无法为大多数问题提供可行的计划。
>
> 《Translating natural language to planning goals with large-language models》
>
> 语言模型无法进行准确的推理或解决规划问题，这可能限制它们在机器人相关任务中的实用性。本文的核心问题是，**LLMs能否将自然语言中指定的目标翻译成结构化的规划语言**。如果可以，LLMs可以作为规划器和人类用户之间的自然接口；翻译后的目标可以交给非领域特定的AI规划器进行处理。对GPT 3.5变体的实证结果表明，**LLMs更适合于翻译而不是规划**。**LLMs能够利用常识知识和推理来提供未指定目标中缺失的细节（通常在自然语言中出现）**。**实验还揭示了LLMs在涉及数值或物理（例如空间）推理的任务中可能无法生成目标，并且LLM对使用的提示敏感**。因此，这些模型在翻译到结构化规划语言方面具有潜力，但在使用时应注意。

相比于其他的工作，本文工作的优势在于：

1. 以前的工作集中于将自然语言翻译为PDDL目标或线性的 Temporal Logics，仅考虑了任务规划问题。

   而作者利用 Signal Temporal Logic (STL) 作为中间表示，使用规划器直接优化整个轨迹，即**任务和运动规划一起**，从而提高规划成功率。

2. 还使用**重新提示**来**自动纠正任务规范错误**，既通过现有的语法错误纠正框架，也通过使用LLMs的新颖的语义错误检查和纠正循环，从而实现了显着的性能改进。

---

作者主要设计的实验如下：

> Comprehensive experiments in challenging task domains, including several multi-agent tasks. 

此外，作者团队发布一个数据集，包括1400个测试用例，包括语言指令、环境、生成的 STL 规范和生成的轨迹。

对比对象：a fine-tuned  `NL-to-STL` model

作者的结论是：

1. GPT-4 few-shot learning is competitive with fine-tuning.
2. 使用预训练的LLMs进行上下文学习非常适合于语言到任务规范的翻译，以解决TAMP问题。

## 2  Problem Description

将包括空间和时间约束的自然语言指令转换为机器人的运动计划，编码为一组与时间有关的路点，例如 $(x_i,y_i,t_i)$ 。环境状态被编码为一组**被命名的障碍物**，描述为多边形，并作为**附加上下文提供**。

作者的任务是根据**给定的指令**和**环境状态**生成**满足约束的轨迹**。

机器人不能超过其最大速度，并且总操作时间不应超过任务时间限制。

假设完整轨迹是路点之间的线性插值，可以通过密集的航点序列指定复杂轨迹。

## 3 Methods

![autotamp_2](https://github.com/JinbiaoZhu/PaperReading/blob/main/pic/autotamp_2.png?raw=true)

### 3.1 LLM End-to-end Motion Planning

一种自然的想法是使用LLM通过直接为**给定的语言指令**生成**轨迹**来处理任务和运动规划。

如果生成的轨迹违反约束条件，将使用**约束违规重新提示**模型以产生另一条轨迹。

最多允许五次这样的重新提示。

> ![autotamp_1](https://github.com/JinbiaoZhu/PaperReading/blob/main/pic/autotamp_1.png?raw=true)
>
> GPT-4 直接端到端轨迹规划的失败案例。
>
> 橙色线显示遵守指令的正确路径。
>
> 紫色和灰色虚线分别显示 GPT-4 在第一次和第二次提示后的轨迹。
>
> GPT-4 生成一个带有关联时间点的 $(x,y)$ 位置列表。初始提示描述了语言建模任务、环境状态和指令。每个对象都由 $(x,y)$ 边界描述为矩形。

### 3.2 LLM Task Planning

使用LLM通过直接从**给定的语言指令**生成**子任务序列**来处理任务规划。

为了生成最终轨迹，子任务由**独立的运动规划器**处理。

---

在本文中，这些子任务仅限于导航动作，并且**运动规划**由作者提出的方法使用的STL**规划器处理**，这样相对公平。

作者评估并与**使用LLMs进行任务规划的三种方法**（Naive任务规划、SayCan 和 LLM任务规划+反馈）进行比较。

> LLM任务规划+反馈 的论文是：《Text2motion: From natural language instructions to feasible plans》 在上一篇笔记中做了简单的摘要。作者是这样描述他们的工作的：将**完整序列生成**与**可行性检查**相结合，旨在执行之前找到**满足完整任务的子任务序列**并**验证其可行性**。对于任何可行的子任务，可以向LLM提供反馈信息以生成新的子任务序列。

### 3.3 Autoregressive LLM Specification Translation&Checking + Formal Planner

包括两种**再提示技术**来提高翻译性能：一种用于**语法错误**，另一种用于**语义错误**。

#### Signal Temporal Logic (STL) Syntax

> 《Monitoring temporal properties of continuous signals》

STL根据以下语法递归地定义STL表达式：（很抽象，我直接截图了......

![autotamp_3](https://github.com/JinbiaoZhu/PaperReading/blob/main/pic/autotamp_3.png?raw=true)

语义谓词、对象之间的关系、对象之间在时间上的描述。

为避免括号匹配的问题，生成的STL遵循前序形式。

#### STL Trajectory Planner

使用一种最好的STL规划器，之前用于多智能体领域。它根据轨迹定义STL公式的有效性，然后优化轨迹以最大化有效性。规划器只考虑区域内或区域外的状态，并输出位置和时间对的序列。

> 《Multi-agent motion planning from signal temporal logic specifications》

#### Syntactic Checking & Semantic Checking

开环翻译可能会出现语法和语义错误。使用两种重新提示技术来自动纠正此类错误。

1. 与Skreta等人类似，使用**验证器**检查语法错误。

   在重新提示LLM生成**更正的STL**时，找到的任何错误都会作为反馈提供。重复此过程，直到找不到错误为止（最多五次迭代）。

2. 对于语义错误，提出了一种新颖的**自回归重新提示**技术，将STL规划器生成的状态序列（即 `[[in（road），0]` ， `[in（red kitchen），0.5]` ， `[in（blue restroom2），1.2]……]` ）反馈到LLM中，以检查它是否满足原始指令。

   如果不满足，则提示LLM修改STL，这将重复语法和语义重新提示。

   此过程在未检测到错误或STL未发生变化的情况下终止（最多三次迭代）。

## 4 Experimental Design

每个任务场景都设置在二维环境中，涉及到一个或多个机器人的导航；机器人在环境中具有限制区域，并以不同的起始位置初始化机器人。每个环境由**形状**、**位置**和**特征（例如颜色、名称、功能）**组成。

对于评估的每种方法，LLM最初会提示语言**描述任务（例如任务规划）**和**五个上下文示例（这些取决于语言任务）**。为了减少提示之间的差异，最初为每种方法测试了**六个不同的示例集**，并选择了表现最佳的一个。通过这个测试，作者发现相对于整体性能来说，**提示的差异是微不足道的**。

在**六个不同的任务场景**（三个单智能体和三个多智能体）中评估了第3部分中描述的不同方法，这些场景具有**不同的几何和时间约束组合**。对于下面每个场景描述，分别使用G和T表示这些约束的存在。

> ![autotamp_4](https://github.com/JinbiaoZhu/PaperReading/blob/main/pic/autotamp_4.png?raw=true)
>
> `HouseWorld` 和 `Chip's Challenge` 是单智能体场景。 `Overcooked` 、 `Rover` 和 `Wall` 是多智能体场景。
>
> 在 `Overcooked` 中，黑色正方形是不可达的。
>
> 线表示按照指令的正确轨迹。对于 `HouseWorld` 和 `Chip's Challenge` 环境，黑色圆点和五边形点分别表示起始和结束位置。

对于每种方法，使用 `GPT-3` 和 `GPT-4` 作为LLM来评估性能。

在多智能体场景中，不测试SayCan或LLM任务规划+反馈，因为这些方法**不容易适应多个智能体**。

对于需要超过90分钟的测试用例，将会终止并报告失败。

通过硬编码检查器自动检查生成的轨迹。

整个实验集使用四个16核CPU进行了两周，LLM API调用的成本约为1500美元。

### HouseWorld1 (single-agent)

这是来自Finucane等人的工作的房屋环境。

> 《Experimenting with language, temporal logic and robot control》

首先手动构建了10个不同复杂度的指令，然后提示GPT-4将**每个指令**改述为**意思相同但措辞不同的9个指令**，从而为该环境生成了**总共100个指令**。

对于每个指令，随机初始化两个**起始-结束位置对**，共进行**200个测试用例**。

对于这种情况，没有对计划轨迹施加硬时间约束。

|     指令     |                     内容                      |
| :----------: | :-------------------------------------------: |
| 自然语言指令 | “参观两个颜色最接近红色但不是纯红色的房间。”  |
| STL语言指令  | `(finally room_purple and finally room_pink)` |

### HouseWorld2 (T, single-agent)

这个场景与 `HouseWorld1` 相同，但每个计划的轨迹都受到**困难的时间约束**。

这个时间限制是通过以**0.8的最大速度**完成正确的轨迹来预先确定的。

|     指令     |                             内容                             |
| :----------: | :----------------------------------------------------------: |
| 自然语言指令 | “Visit two rooms with color closest to red, but not the pure red color. The task should be completed within 10 seconds.” |
| STL语言指令  | `(finally[0, 10] room_purple and finally[0, 10] room_pink)`  |

> 其余的任务场景是为智能体制定了特定的规则和目标。
>
> 对于每个场景，使用GPT-4将**原始描述**改述**为意思相同但措辞不同的20个变体**。
>
> 为每个场景实例化了三个不同的环境实例，并随机化了五个不同的**起始-结束位置对**，共进行了300个测试用例。请注意，多个智能体之间的避免碰撞已经**内在地编码在STL规划器**中。

### Chip’s Challenge (G, single-agent)

这是一个受 `Chip's Challenge` 游戏中某个关卡启发的场景，该游戏具有严格的几何和逻辑约束条件。机器人必须到达所有目标区域（蓝色），但必须获取独特的钥匙才能通过相应的门。

|     指令     |                             内容                             |
| :----------: | :----------------------------------------------------------: |
| 自然语言指令 | Try to reach all the goals but you have to reach the corresponding key first to open the specific door. <br/>For example, you have to reach key1 ahead to open door1. <br/> Also remember always do not touch the walls. |
| STL语言指令  | `finally enter(goal_i) (i = 1,2,3...) and ( negation enter(door_j) until enter(key_j ) (j= 1,2,3...) ) and globally not_enter(walls)` |

### Overcooked (G & T, multi-agent)

这是一个受 `Overcooked` 游戏启发的场景，该游戏具有严格的时间限制的烹饪模拟游戏。

智能体必须在有限的时间内协作收集食材并返回烹饪室。

在这种情况下，多智能体运动规划受到了智能体机动空间的限制。

|     指令     |                             内容                             |
| :----------: | :----------------------------------------------------------: |
| 自然语言指令 | Enter all the ingredient room to pick up food. Once entered ingredient rooms,<br/>go to cooking room within 3 seconds. And all the agents should not collide each other and black obstacles. |
| STL语言指令  | `finally enter(ingredient_i) (i = 1,2,3...) and ( enter(ingredient_j ) imply finally[0, 3]` |

### Rover (G & T, multi-agent)

多个智能体必须在时间和能量的限制下，在进入红色区域之前到达每个观察区域（蓝色目标）并传输观察结果。

|     指令     |                             内容                             |
| :----------: | :----------------------------------------------------------: |
| 自然语言指令 | All rovers must reach the blue charging station within 5 units of time each time they exit it. <br/> Once they reach their destination, they need to get to a yellow transmitter within 2 time units to send the collected information to the remote control. <br/>Rovers must keep clear of black walls and other rovers.<br/> All target areas need to be visited |
| STL语言指令  | ` finally enter(goal_i) (i = 1,2,3...) and ( enter(goal_j) imply finally[0, 2] (enter(Transmitter_1) or enter(Transmitter_2) ) (j = 1,2,3...) ) and ( not_enter(Charging) imply finally[0,5] enter(Charging) and globally not_enter(walls) )` |

### Wall (G & T, multi-agent) 

多个智能体必须在时间限制和机动瓶颈的限制下占据每个目标区域（蓝色）。

|     指令     |                             内容                             |
| :----------: | :----------------------------------------------------------: |
| 自然语言指令 | Reach all the goals and stay there. Take care of the collision among each agent and to the walls. |
| STL语言指令  | `finally globally enter(goal_i) (i = 1,2,3...) and globally not_enter(walls)` |

## 5 Results

### 5.1 Task Success Rates

在表1和表2中报告了单智能体和多智能体场景的任务成功率。

对于没有严格时间限制的 `HouseWorld1` ，所有使用LLMs作为任务规划器的方法都优于我们的方法；该环境允许直接在任意两个位置之间进行轨迹规划，因此缺乏这些方法将难以应对的几何挑战。

当添加严格的时间限制（`HouseWorld2`）时，这些方法的表现要差得多，而 `AutoTAMP` 的成功率仍然持续。对于包括几何约束的其余任务，LLM端到端运动规划和Naive任务规划都表现相当差。观察到GPT-4在所有方法中都优于GPT-3的总体趋势。

![autotam_5](https://github.com/JinbiaoZhu/PaperReading/blob/main/pic/autotamp_5.png?raw=true)

### 5.2 Failure examples for LLM Task Planning and AutoTAMP Methods

#### LLM Task Planning

LLM任务规划方法可能会在需要长期规划的任务中失败。

![autotamp_6](https://github.com/JinbiaoZhu/PaperReading/blob/main/pic/autotamp_6.png?raw=true)

在（a）中，这些方法不是在解锁门1之前，先获取靠近起始位置的key1和key2，而是将动作序列化为获取key1，解锁door1，然后获取key2。这种**低效率**可能会导致违反任务规范中的时间限制，从而导致任务失败。在（b）中，ingredient1被两个智能体**不必要地（且低效地）**访问。

#### AutoTAMP

AutoTAMP（以及消融实验）主要因为翻译错误而失败。虽然使用的**重新提示技术**可以改善翻译，但仍然存在翻译错误导致任务失败的情况。

### 5.3 Planning Time

![autotamp_7](https://github.com/JinbiaoZhu/PaperReading/blob/main/pic/autotamp_7.png?raw=true)

在表3中报告了AutoTAMP与GPT-4的时间信息。对于每个场景，报告了以下步骤的第 `10/50/90` 个百分位运行时成本：GPT-4 API调用，STL规划器规划时间，代码执行时间，GPT-4翻译成STL，语法检查循环，语义检查循环。

作者的结论：STL规划器运行时间是主要瓶颈。在GPT-4 API调用需要过多时间（即超过100秒）的情况下，这可能是由于不稳定的延迟造成的，并且可能不是解码运行时的良好智能体。

### 5.4 Ablation Studies

#### Syntactic and Semantic Checkers

没有错误修正的翻译在各种任务场景中取得了适度的成功，但是**语法和语义错误修正均显著提高**了性能；这种趋势在所有场景中都存在。

#### Fine-tuned NL2TL Translation Model

之前的工作 `NL2TL` 已经通过使用经过**精细调整的LLM的模块化流水线**研究了从自然语言到STL的翻译。虽然将 NL2TL 整合到 AutoTAMP 框架中并不是主要工作，但在表4中报告了它对任务成功的影响。

![autotamp_8](https://github.com/JinbiaoZhu/PaperReading/blob/main/pic/autotamp_8.png?raw=true)

前三行报告了带有语法和语义重新提示的AutoTAMP + NL2TL的消融结果。底部三行报告了没有NL2TL的AutoTAMP的原始消融结果。

整合NL2TL通常会带来适度的改进。但是，由于它**不依赖于有针对性的数据集或额外的离线训练**，AutoTAMP似乎在不使用经过微调的模型的情况下也具有竞争力。

## 6 Related Work

### LLMs for TAMP

对LLMs在任务和运动规划中的研究，其中一种方法是直接将LLMs用作规划器。最初的工作表明，从高级任务描述中进行纯粹的的 zero-shot 生成基本动作序列的**可执行性相对较差**，但是**少量的上下文学习**、将**输出限制为可接受的动作**以及**自回归逐步动作**生成显著提高了性能。

随后的工作通过采用基于运动控制策略的**原始动作**，并**使用可负担性函数**来指导基于LLMs的任务和运动规划。还有工作是：通过在查询**下一个动作之前提示反馈**（例如动作成功指示）来形成闭环。

其他工作集中于提示如何通知任务执行，例如**用户任务偏好的少量摘要**或**提取未明确定义的对象放置的常识**安排。尽管取得了这些成功，但是有证据表明LLMs在更现实的任务上表现不佳。

作者的工作并没有直接将LLMs用作规划器。

### Translating Language to Task Representations

一种自然的替代方案是依靠专用的规划器，通过从自然语言到规划表示的映射来实现。

将语言映射到 $\lambda$ 演算、运动规划约束、线性时间逻辑和信号时间逻辑等。

为了解决数据可用性、任务泛化和语言复杂性等挑战，最近的工作已将LLMs应用于这个翻译问题。

模块化方法，使用LLMs提取**相应逻辑命题的指称表达式**，然后构建**完整的时间逻辑规范**。依靠LLMs进行直接翻译，其他工作已将语言映射到PDDL目标或完整的PDDL问题。

作者的工作类似地将其转换为任务规范，但可以表示复杂的约束（例如时间），并引入了一种新颖的机制来自动检测和纠正语义错误。

### Re-prompting of LLMs

有用的上下文（例如用于新任务的少量上下文学习）极大地提高了LLM输出的质量。LLMs通常还提供与任务相关的信息，例如环境状态或可接受的动作。基于LLMs输出的附加上下文重新提示已被证实有效，例如迭代动作生成、环境反馈、不可接受的动作、未满足的动作前提条件、代码执行错误以及结构化输出中的语法错误。

> 《ProgPrompt: Generating situated robot task plans using large language models》
>
> 《Language models are few-shot learners》
>
> 《Planning with large language models via corrective re-prompting》
>
> 《Errors are useful prompts: Instruction guided task programming with verifier-assisted iterative prompting》

作者的工作利用了与Skreta等人相同的语法纠正重新提示技术，但作者进一步通过重新提示引入了自动检测和纠正语义错误的方法。

## 7 Conclusion and Limitations

1. 尽管结果依赖于从几个候选项中**选择最佳提示**，但所使用的提示**可能不是引出LLMs最佳性能的最佳提示**。因此，每种方法的个体结果可能存在改进空间，且期望这样的改进趋势将持续存在。

2. 虽然AutoTAMP的任务成功率很高，但规划时间的成本很高，特别是对于多次语法和语义重新提示迭代。

   需要进一步的工作，集中研究如何**加速规划器**并**减少LLM推理延迟**。 

3. 由于翻译错误，一些任务仍然失败。

   作为未来工作的一部分，探索通过**人在环中框架的反馈**是否可以改善翻译并减少AutoTAMP迭代次数。

## Appendix

### A Recursive LLM Prompts

在这项工作中，一些方法需要使用提示历史的上下文重新提示LLMs。

与提供此功能模板的GPT-4 API不同，GPT-3 API模板仅适用于单个推理历史记录。

因此，有必要包括任何提示历史记录作为上下文，以指示先前的用户输入和LLM响应。

![autotamp_9](https://github.com/JinbiaoZhu/PaperReading/blob/main/pic/autotamp_9.png?raw=true)

### B Convert Sub-task Sequence to STL

为了评估LLMs任务规划方法，使用STL规划器（用于AutoTAMP方法的相同规划器）作为生成轨迹的运动规划器。

LLMs任务规划方法生成的输出形式为 `[[enter(room1), t1], [enter(room2), t2]，...，[enter(roomi), ti]，[enter(roomi+1)，ti+1]]` 。自动将其转换为STL，以供规划器使用，即 `finally[t1, t1] enter(room1) and finally[t2, t2] enter(room2) ... and finally[ti, ti] enter(roomi) and globally not_enter(walls)` 。

如果LLMs任务规划方法还需要可执行性检查（例如SayCan、LLM任务规划+反馈），则对每个子任务应用相同的转换；例如，给定一个子轨迹 $i$ ，我们将其转换为 `finally[ti+1−ti, ti+1−ti] enter(roomi+1) and globally not_enter(walls)` ，然后检查其可执行性。

### C STL Syntax

![autotamp_10](https://github.com/JinbiaoZhu/PaperReading/blob/main/pic/autotamp_10.png?raw=true)

### D Pre-order and In-order STL Formats

每个STL都可以视为一个二叉树，树中的**每个节点**最多有**两个子节点**。

在请求NL-to-STL翻译时，STL的目标标记可以按照**中序（左子树、根、右子树）**或**前序（根、左子树、右子树）**的方式进行线性化。

对于这项工作，提示LLMs以**前序格式**生成STL，以避免括号匹配错误，并与STL规划器兼容。

使用GPT-4进行翻译时，两种格式的翻译性能相似。

![autotamp_11](https://github.com/JinbiaoZhu/PaperReading/blob/main/pic/autotamp_11.png?raw=true)

