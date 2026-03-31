# Large Language Models Simply Achieve Explainable and Training-Free One-shot HRRP ATR :sheep:

<div align="right">
  <a href="#english">English</a> | <a href="#chinese">中文</a>
</div>

---

<a name="english"></a>
## English

This repository provides the codes for HRRPLLM, IEEE SPL 2025.

> Diverging from conventional methods requiring extensive task-specific training or fine-tuning, our approach converts one-dimensional HRRP signals into textual scattering center representations. These are then directly processed by an LLM, which performs target recognition via few-shot in-context learning, effectively leveraging its vast pre-existing knowledge without any parameter update. As the first work to utilize general-purpose LLMs directly to HRRP target recognition, our simple but effective approach generates competitive results among current HRRP ATR baselines. This opens new avenues for the domain of few-shot radar target recognition.

<p align="center">
  <img src="LLMsATR.jpg" width="50%">
</p>

---

### Platform :pushpin:

We wrote, ran and tested our scripts on PyCharm IDE in a Conda environment, which we recommend for reproduction.
We also recommend running the code on Linux (Our testing was done on Ubuntu 20.04).

---

### Dependencies :wrench:

You actually don't need PyTorch, Tensorflow and etc. because our HRRPLLM is training-free and based on API calling.

Set up the environment with the `requirements.txt`.

---

### A DEMO Toy Example

An interactive, client-side demonstration of the HRRPLLM prompt structure and simulated reasoning process is available:

This demo allows you to:
- View the fixed contextual information and few-shot exemplars provided to the LLM.
- Input or modify the scattering center data for a test sample.
- See a *simulated* LLM prediction and a generated rationale based on simple heuristics.

**Important:** This demo runs entirely in your browser using JavaScript. It does **not** make actual calls to any Large Language Model API. The "LLM" responses are simulated based on a simplified comparison of the input scattering centers to the predefined prototypes. It serves to illustrate the concept and the type of information an LLM would process.

#### How to Run the DEMO Locally
1. Clone this repository.
2. Open the `index.html` file in your web browser.

---

<a name="chinese"></a>
## 中文

本仓库提供了 HRRPLLM 的代码，该工作已发表于 IEEE SPL 2025。

> 与传统方法需要大量任务特定的训练或微调不同，我们的方法将一维 HRRP 信号转换为文本散射中心表示。然后由 LLM 直接处理这些表示，通过少样本上下文学习进行目标识别，有效利用其庞大的预存知识而无需任何参数更新。作为首个将通用 LLM 直接应用于 HRRP 目标识别的工作，我们简单而有效的方法在当前 HRRP ATR 基线方法中取得了有竞争力的结果。这为少样本雷达目标识别领域开辟了新的途径。

<p align="center">
  <img src="LLMsATR.jpg" width="50%">
</p>

---

### 平台 :pushpin:

我们在 PyCharm IDE 的 Conda 环境中编写、运行和测试了脚本，我们推荐您使用此环境进行复现。
我们还建议在 Linux 上运行代码（我们的测试在 Ubuntu 20.04 上完成）。

---

### 依赖 :wrench:

实际上您不需要 PyTorch、Tensorflow 等，因为我们的 HRRPLLM 是无训练的，基于 API 调用。

使用 `requirements.txt` 设置环境。

---

### DEMO 玩具示例

我们提供了一个交互式的客户端演示，展示 HRRPLLM 的提示结构和模拟推理过程：

该演示允许您：
- 查看提供给 LLM 的固定上下文信息和少样本示例。
- 输入或修改测试样本的散射中心数据。
- 查看基于简单启发式的*模拟* LLM 预测和生成的推理依据。

**重要提示：** 此演示完全在您的浏览器中使用 JavaScript 运行。它**不会**实际调用任何大型语言模型 API。"LLM" 响应是基于输入散射中心与预定义原型的简化比较进行模拟的。它用于说明概念和 LLM 将处理的信息类型。

#### 如何在本地运行 DEMO
1. 克隆本仓库。
2. 在您的网页浏览器中打开 `index.html` 文件。

---

## Citation / 引用

If you find our work useful in your research, please consider citing:

如果您在研究中觉得我们的工作有用，请考虑引用：

```bibtex
@ARTICLE{11122886,
  author={Chen, Lingfeng and Hu, Panhe and Pan, Zhiliang and Liu, Qi and Zhang, Shuanghui and Liu, Zhen},
  journal={IEEE Signal Processing Letters}, 
  title={Large Language Models Can Achieve Explainable and Training-Free One-Shot HRRP ATR}, 
  year={2025},
  volume={32},
  number={},
  pages={3395-3399},
  keywords={Indexes;Target recognition;Scattering;Radar;Training;Large language models;Frequency-domain analysis;Data mining;Frequency modulation;Transforms;High-resolution range profile;automatic target recognition;large language models;in-context learning},
  doi={10.1109/LSP.2025.3598220}}
```

---

## Acknowledgements :small_red_triangle:

This project is released under :page_facing_up: the MIT license.

本项目基于 :page_facing_up: MIT 许可证发布。
