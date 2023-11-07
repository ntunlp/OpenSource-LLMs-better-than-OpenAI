# OpenSource-LLMs-better-than-ChatGPT


## Datasets

### Evaluation datasets

#### Question-answering (QA)

1. [**Natural Questions**: A Benchmark for Question Answering Research](https://aclanthology.org/Q19-1026/). Tom Kwiatkowski et al, TACL 2019. 307,373 training examples with single annotations; 7,830 development examples with 5-way annotations and 7,842 test examples with 5-way annotations. Questions are real anonymized, aggregated queries issued to the Google search engine. Each question is paired with an entire Wikipedia page. 
2. [**BoolQ**: Exploring the Surprising Difficulty of Natural Yes/No Questions](https://aclanthology.org/N19-1300/). Christopher Clark et al, NAACL 2019. 16,000 naturally occurring yes/no questions, split in 9.4k train, 3.2k dev, 3.2k test. Each question is paired with a Wikipedia passage. 
3. [**TruthfulQA**: Measuring How Models Mimic Human Falsehoods](https://arxiv.org/abs/2109.07958). Stephanie Lin et al, ACL 2022. 817 questions spanning 38 categories. Question and answers are hand-written by human annotators and designed to elicit imitative falsehoods.

#### Long-input capacity

1. [**ZeroSCROLLS**: A Zero-Shot Benchmark for Long Text Understanding](https://arxiv.org/abs/2305.14196). Uri Shaham et al, EMNLP 2023. Long-input benchmark containing 10 datasets: 2 summarization tasks ([GovReport](https://arxiv.org/abs/2104.02112) and [SummScreenFD](https://arxiv.org/abs/2104.07091)), 2 query-based summarization tasks ([QMSum](https://arxiv.org/abs/2104.05938) and [SQuALITY](https://arxiv.org/abs/2205.11465)), 4 question-answering datasets ([Qasper](https://arxiv.org/abs/2105.03011), [NarrativeQA](https://arxiv.org/abs/1712.07040), [QuALITY](https://arxiv.org/abs/2112.08608), [MuSiQue](https://arxiv.org/abs/2108.00573)), 2 aggregation tasks ([SpaceDigest](https://arxiv.org/abs/2012.04443), [BookSumSort](https://arxiv.org/abs/2105.08209)). 

#### Code

1. [Evaluating Large Models Trained on Code (**HumanEval** benchmark)](https://arxiv.org/abs/2107.03374). Mark Chen et al, 2021. 164 hand-written programming problems. Each problem includes a function signature, docstring, body, and several unit tests, with an average of 7.7 tests per problem.

#### Math

1. [Training Verifiers to Solve Math Word Problems (**GSM8K** benchmark)](https://arxiv.org/abs/2110.14168). Karl Cobbe et al, 2021. 8.5K (7.5k training + 1k test) high quality grade school math problems created by human problem writers. Each problem takes between 2 and 8 steps to solve, and solutions primarily involve performing a sequence of elementary calculations using basic arithmetic operations.

### Fine-tuning / instruction-tuning datasets

1. [AgentTuning: Enabling Generalized Agent Abilities For LLMs](https://arxiv.org/pdf/2310.12823.pdf). Zeng et al., 2023. Introduce **AgentInstruct** dataset: 1,866 high quality interaction trajectories generated by GPT-4 and verified by human.
2. [Enhancing Chat Language Models by Scaling High-quality Instructional Conversations](https://arxiv.org/abs/2305.14233). Ning Ding et al, 2023. Introduce **UltraChat** dataset: 1.5 million high-quality multi-turn dialogues covering a wide range of topics and instructions.
3. [**OpenAssistant Conversations** -- Democratizing Large Language Model Alignment](https://arxiv.org/abs/2304.07327). Andreas Köpf et al, 2023. 161,443 messages (91,829 prompter and 69,614 assistant messages) distributed across 66,497 conversation trees, in 35 different languages, annotated with 461,292 quality ratings.


## High-performing open-source LLMs

### Better pre-training techniques

| **LLM**                                           | **Date released** | **Pre-training** | **LLM size** | **Eval Dataset (metric)** | **OpenAI model** | **OpenAI result** | **LLM result** |
|---------------------------------------------------|-------------------|------------------|--------------|-------------|------------------|-------------------|----------------|
| **Phi-1** [[paper](https://arxiv.org/abs/2306.11644)] | June 20th, 2023 | From scratch | 1.3B | HumanEval (pass@1) | GPT-3.5 | 47.0 | 50.6 (+7.7%) |
| **Llama-2-Long-Chat** [[paper](https://arxiv.org/abs/2309.16039)] | Sept 27th, 2023 | Llama-2 + 400B tokens | 70B | ZeroScrolls (average) | GPT-3.5-turbo-16k | 36.7 | 37.7 (+2.7%) |
| **Lemur** [[paper](https://arxiv.org/abs/2310.06830)] | Oct 10th, 2023 | Llama-2 + 90B tokens | 70B | HumanEval (pass@1) | GPT-3.5-turbo | 37.78 | 46.67 (+23.5%) |
| | | | | GSM-8K | GPT-3.5-turbo | 43.75 | 58.33 |

### Better fine-tuning / instruction-tuning techniques

| **LLM**                                           | **Date released** | **Backbone LLM** | **LLM size** | **Eval Dataset** | **OpenAI model** | **OpenAI result** | **LLM result** |
|---------------------------------------------------|-------------------|------------------|--------------|-------------|------------------|-------------------|----------------|
| **FireAct** [[paper](https://arxiv.org/abs/2310.05915)] | October 9th, 2023 | LLama-2 | 13B | HotPotQA (EM) | GPT-3.5 | 31.4 | 34.4 (+9.6%) |
| **AgentLM** [[paper](https://arxiv.org/abs/2310.12823)] | October 19th, 2023 | LLama-2 | 70B | ALFWorld (SR) | GPT-4 | 78.0 | 86.0 (+10.3%) |
| | | | | Knowledge Graph (F1) | GPT-3.5 | 27.2 | 47.0 (+72.8%) |
| | | | | Database (SR) | GPT-4 | 33.7 | 37.7 (+11.9%) |
| | | | | HotPotQA (EM) | GPT-3.5 | 37.4 | 41.6 (+11.2%) |
| | | | | GSM-8K (SR) | GPT-3.5 | 57.1 | 59.7 (+4.6%) |

3. [Enhancing Chat Language Models by Scaling High-quality Instructional Conversations (**UltraChat** dataset)](https://arxiv.org/abs/2305.14233). Ning Ding et al, 2023. 1.5 million high-quality multi-turn dialogues covering a wide range of topics and instructions.
4. [**Albel**](https://github.com/GAIR-NLP/abel/#Citation). *"We propose Parental Oversight, A Babysitting Strategy for Supervised Fine-tuning."* Albel can outperform ChatGPT on GSM8K.
5. [**WizardMath**: Empowering Mathematical Reasoning for Large Language Models via Reinforced Evol-Instruct](https://arxiv.org/abs/2308.09583). A evolutional self-instructing method for SFT. WizardMath can outperform ChatGPT on GSM8K. Noted that all open-sourced models still fall behind GPT-4 on MATH with significant margin.

### Better inference techniques 
