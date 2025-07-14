# Large Language Models Simply Achieve Explainable and Training-Free One-shot HRRP ATR :sheep:
This repository provides the codes for HRRPLLM submitted to IEEE SPL 2025.

 ---

> Diverging from conventional methods requiring extensive task-specific training or fine-tuning, our approach converts one-dimensional HRRP signals into textual scattering center representations. These are then directly processed by an LLM, which performs target recognition via few-shot in-context learning, effectively leveraging its vast pre-existing knowledge without any parameter update. As the first work to utilize general-purpose LLMs directly to HRRP target recognition, our simple but effective approach generates competitive results among current HRRP ATR baselines. This opens new avenues for the domain of few-shot radar target recognition.
><p align="center">
  > <img src="LLMsATR.jpg" width="50%">
</p>

 ---

## Platform :pushpin:
We wrote, ran and tested our scripts on PyCharm IDE in a Conda environment, which we recommend for reproduction.
We also recommend running the code on Linux (Our testing was done on Ubuntu 20.04).

 ---

## Dependencies :wrench:
You actually don't need PyTorch, Tensorflow and etc. because our HRRPLLM is training-free and based on API calling.

Set up the environment with the `requirements.txt`. 

 ---

## A DEMO Toy Example

An interactive, client-side demonstration of the HRRPLLM prompt structure and simulated reasoning process is available:

This demo allows you to:
- View the fixed contextual information and few-shot exemplars provided to the LLM.
- Input or modify the scattering center data for a test sample.
- See a *simulated* LLM prediction and a generated rationale based on simple heuristics.

**Important:** This demo runs entirely in your browser using JavaScript. It does **not** make actual calls to any Large Language Model API. The "LLM" responses are simulated based on a simplified comparison of the input scattering centers to the predefined prototypes. It serves to illustrate the concept and the type of information an LLM would process.

### How to Run the DEMO Locally
1. Clone this repository.
2. Open the `index.html` file in your web browser.

---

## Acknowledgements :small_red_triangle:	
This project is released under :page_facing_up: the MIT license.