# Reviewer #1
1. The framework claims to reduce "hallucinations," but it does not clearly explain how the "Anomaly Backtracking" logic works in the architecture section.
2. The framework relies on specific large language models (LLMs), like GPT-4, but it does not explore how changing temperature settings (t_i) affects its performance.
3. There is not enough discussion about the delay caused by the multi-agent interaction layers compared to traditional manual modeling timelines.
4. The claim that the system reduces barriers for non-experts lacks user-study data to support the effectiveness of the "Business Analyst Agent."
5. In Section 2.4 (Challenges). It is suggested to refer from the paper: " A novel permissioned blockchain approach for scalable and privacy-preserving IoT authentication". This citation addresses "privacy and trust issues" in decentralized environments, which is important for discussing the security of data shared between port agents in the ECR network.
6. The paper should include a clear, step-by-step description of how the PM-Agent detects a failure in the TE-Agent and triggers a state transition back to either the ME-Agent or the PY-Agent.
7. The paper should add an experiment that changes the LLM temperature (t_i) and top_p values to show how stable the generated Mixed Integer Programming (MIP) formulations are.
8. The terminology must be consistent; the terms "SOP-MAS" and "LLM-agent system" should be used in the same way throughout.



## Reviewer #2
A) Define "accuracy" precisely. Distinguish between successful code execution and reaching the global optimum (optimality gap). High-impact journals require measuring the mathematical quality of the solution, not just syntax.

B) Explicitly confirm if all baselines (OptiMUS, CoE, etc.) used the same DeepSeek-v3 backbone. Paradigm effectiveness cannot be validated if the underlying language models differ across comparisons.

C) Re-evaluate Equation 1's probabilistic notation. If agent selection is governed by deterministic prompts rather than actual probability distributions, simplify the math to avoid unnecessary "math-washing."

D) Provide efficiency metrics correlated to problem size. Show how runtime and accuracy scale as the number of ports and time periods increase to prove industrial-grade scalability.

E) Support "reduced modeling time" claims with a human baseline. Compare the 48-second SOP-MAS runtime against the actual hours an expert requires to manually model the Liaoning case.

F) Clarify the Intel i5-4210U hardware usage. If an API was used for DeepSeek-v3, separate network latency from agent reasoning time to ensure the efficiency analysis is meaningful.

G) Propose a qualitative metric for the Business Analyst agent. Since it does not affect mathematical accuracy, demonstrate its value through user-interpretability scores or decision-making speed improvements.



Reviewer #3:
本文介绍了一种在多智能体协作领域内应用大语言模型（LLM）的框架，旨在解决基准测试场景和实际应用问题，重点关注空箱调运（ECR）问题。在基准评估中，所提出的方法与现有技术相比表现出了显著的提升。论文进行了全面的分析，包括对特定参数的敏感性分析、消融研究以及计算效率评估。文中描述了 ECR 案例研究，相关数据和开发的模型在附录中进行了详细说明。

为了提高论文质量，我提出以下建议：

1. 在第 314-315 行，未对 top_p 和 max_tokens 这两个术语进行定义。
2. 在第 332 行，“将第 i 步的输出记为 yi。”这一表述引发了一个问题：这里的 i 与“A = {ai} (i=0 到 n)”中的 i 是同一个吗？在该集合中，i 是否同时代表了迭代步数（如图 1 所示）和集合 A 中的智能体编号？
3. 公式 (1) 中变量的符号没有明确定义。e、Si、s 和 Ai 的定义是什么？
4. 在图 1 中，变量 N 缺乏定义。最好能详细说明应该如何确定 N 的值。
5. 在图 1 中，异常回溯部分有一个从“error: ai?”框指出的 Y 箭头，这表示 ai 出现错误会导致 ai 被加入堆栈 H 中。这种理解是否准确？还是说过程应该反过来？
6. 在第 345-346 行，“如果当前迭代次数 k 超过了最大允许选择次数 K”这句话，并未引用图 1 中 k 和 K 的值。正如作者在第 319 行所指出的，本节的 SOP 与图 1 是一致的。但本节中的几个变量似乎与图 1 中描绘的变量不一致。
7. 在第 3.2 节中，i 和 k 的作用区分得不够清楚。必须阐明这两个参数之间的区别。
8. 在第 3.4 节的第 420-432 行，作者旨在使用图 2 描述集装箱供应链管理中经典 ECR 案例的 SOP-MAS 过程。然而，该图实际上展示的是飞机着陆问题。此外，图中使用的是 PD 和 MA 智能体，而不是之前确立的 PM 和 BA 智能体。
9. 考虑到图 2 是该框架的一个关键示例，对图中说明的过程提供更全面的解释将是有益的，并且将大大增进读者的理解。例如，可以在正文中对异常回溯组件进行更深入的讨论。
10. 在第 501 行，“……相比 SPM 准确率提升了 8.89%。”这一断言与 COT 和 SPM 的准确率（分别为 52.26% 和 48.08%）不一致。因此，提升幅度并不是 8.89%（*译者注：审稿人此处原文写的8.69%，结合数据（52.26-48.08）/48.08 约为 8.69%，作者需核实原文笔误*）。
11. 在第 534 行，“……被停用 (is_backward = False)……”这一短语中，应像第 369 行那样，将“is_backward = False”加粗强调。
12. 在第 581-582 行，提供了符号“+”和“-”的定义，但在稿件中任何地方都没有应用它们。
13. 在第 689 行，“St_i”中的变量“t”缺乏定义。
14. 在第 719-729 行，展示了由 SOP-MAS 框架得出的解决方案，但讨论缺乏数值证据的支持。加入带有示例的更全面的讨论将显著提高论文的质量。
15. 在第 745 行，结论中突然提到了调运成本降低的百分比。这一改进内容应该在分析模型解决方案的第 5.3 节中进行讨论。
16. 在结论部分，探讨潜在的未来研究方向将是有益的。鉴于运筹学这一领域的快速发展和浓厚兴趣，这样的讨论将是一个有价值的补充。
17. 一个小建议：作者可以考虑将问题的数学规划模型整合到论文的正文中，而不是将 LaTeX 版本放在附录中。虽然可以理解作者是想突出 LLM 和 MAS 方法得出的结果，但这种呈现方式可能会增加理解问题和模型的难度。不过，如果作者选择保留目前的 LaTeX 格式将模型放在附录，也是可以理解的。

本文提出了一种解决复杂现实挑战的创新方法，并通过空箱调运（ECR）问题的案例研究进行了说明。作者旨在以清晰易懂的方式呈现该方法，并有实验证据表明，所提出的方法优于现有替代方案。我认为这篇论文引人入胜，并相信它为该领域引入了新的视角。然而，某些部分需要进一步澄清，特别是关于图表的解释。此外，论文标题《结合大语言模型与多智能体系统的空箱调运优化》暗示了会对模型和案例研究结果进行详细讨论，但目前这方面有所欠缺，特别是与基准测试场景相比。




- 人工建模和SOP-MAS建模的讨论不够充分的问题，涉及意见有Comment 1.3、Comment 2.E
  - Comment 1.3 Reviewer: “There is not enough discussion about the delay caused by the multi-agent interaction layers compared to traditional manual modeling timelines.”
  - “Support ‘reduced modeling time’ claims with a human baseline. Compare the 48-second SOP-MAS runtime against the actual hours an expert requires to manually model the Liaoning case.”
  - 回溯思路：量化人工建模的时间是个难点，引用其他论文作为回复
- BA-Agent的定量分析，涉及意见有Comment 1.4、Comment 2.G：
  - BA-Agent 主要是做的是AI可解释性，但是在实验中没有量化分析BA-Agent的功能。审稿人指出需要定量分析BA-Agent的作用，“建议为业务分析师智能体（BA-Agent）提出一项定性指标。鉴于它不影响数学准确率，请通过‘用户可解释性评分’或‘决策速度的提升’来证明其价值。”
  - 回复思路：回头看下参考文献 LI B, MELLOU K, ZHANG B, 等. Large Language Models for Supply Chain Optimization[A/OL]. arXiv, 2023[2026-04-07]. http://arxiv.org/abs/2307.03875. DOI:[10.48550/arXiv.2307.03875](https://doi.org/10.48550/arXiv.2307.03875).
- Top-p和temperature参数的说明，涉及意见有Comment 1.2， 1.7，Comment 3.1
  - **Top-p（也叫核采样，Nucleus Sampling）是大模型用来控制生成文本“创造力（随机性）”的一个核心参数。**
  - **Temperature** 是通过拉平或放大概率差距来控制随机性。
  - 回复思路是：移除Top-p
- 回溯， Comment 1.1， 1.6， 

