# Working Memory Identifies Reasoning Limits in Language Models

[Paper Link]([https://github.com/suzgunmirac/BIG-Bench-Hard](https://aclanthology.org/2024.emnlp-main.938.pdf))

This repository contains datasets, prompts, code, and supplementary materials from our **EMNLP 2024 paper**. Our study explores the cognitive limitations of large language models (LLMs) concerning working memory and autonomous reasoning capabilities. While pretraining can scale the working memory capacity of models like GPT-3.5 and GPT-4 to levels comparable with humans, their reasoning is still constrained by limitations in their autonomous reasoning and planning abilities.

Our findings highlight the importance of enhancing LLMs’ internal capabilities, such as planning and search mechanisms, to improve their autonomy and reasoning performance.

## Overview

The primary objective of this study is to explore and mitigate the inherent reasoning limitations of LLMs such as GPT-3.5 and GPT-4. Inspired by cognitive science, we develop CoT+ (Chain of Thought enhanced with cognitive science insights) prompts, which aim to improve LLMs' ability to identify and process complex reasoning patterns. Our work pushes the boundaries of LLMs toward more general artificial intelligence by addressing working memory constraints and enhancing reasoning autonomy.

## Contents

This repository is continuously updated with the following materials:

- **Datasets/**: Contains the Big-Bench Hard (BBH) Task Card dataset, which evaluates the reasoning capabilities of LLMs across various tasks.
- **Prompts/**: Includes original CoT prompts and our enhanced CoT+ prompts designed to address failure cases and improve working memory performance.
- **Code/**: Provides scripts and Jupyter notebooks for executing reasoning tasks, analyzing model responses, and evaluating performance improvements using CoT+ prompts.

## Acknowledgments

We gratefully acknowledge the authors of the [Big-Bench Hard](https://github.com/suzgunmirac/BIG-Bench-Hard) paper for providing the BBH dataset and original CoT prompts, which served as a foundation for this research and for the benefit of the wider research community. We also greatly appreciate the authors who created [ChatGPT-WM] (https://github.com/Daniel-Gong/ChatGPT-WM/tree/main/datasets).

## License

This project is licensed under the [MIT License](https://opensource.org/licenses/MIT).
