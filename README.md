# OpenSource-LLMs-better-than-ChatGPT


# Datasets

## Evaluation datasets

### Long-context (summarization, QA, etc)

#### General long-context benchmarks 

10. [**ZeroSCROLLS**: A Zero-Shot Benchmark for Long Text Understanding](https://arxiv.org/abs/2305.14196). Uri Shaham et al, EMNLP 2023. Long-input benchmark containing 10 datasets: 2 summarization tasks ([GovReport](https://arxiv.org/abs/2104.02112) and [SummScreenFD](https://arxiv.org/abs/2104.07091)), 2 query-based summarization tasks ([QMSum](https://arxiv.org/abs/2104.05938) and [SQuALITY](https://arxiv.org/abs/2205.11465)), 4 question-answering datasets ([Qasper](https://arxiv.org/abs/2105.03011), [NarrativeQA](https://arxiv.org/abs/1712.07040), [QuALITY](https://arxiv.org/abs/2112.08608), [MuSiQue](https://arxiv.org/abs/2108.00573)), 2 aggregation tasks ([SpaceDigest](https://arxiv.org/abs/2012.04443), [BookSumSort](https://arxiv.org/abs/2105.08209)).

#### Long-context summarization 

1. [**BookSum**: A Collection of Datasets for Long-form Narrative Summarization](https://arxiv.org/abs/2105.08209). Kryściński et al, 2021.
2. [**QMSum**: A New Benchmark for Query-based Multi-domain Meeting Summarization](https://arxiv.org/abs/2104.05938). Ming Zhong et al, NAACL 2021. 1,808 query-summary pairs over 232 meetings in multiple domains. Meetings are from 3 categories: product, academic, committee ; and are annotated by AMT workers.
3. [Efficient Attentions for Long Document Summarization (**GovReport**)](https://arxiv.org/abs/2104.02112). Huang et al, NAACL 2021.
4. [**SummScreen**: A Dataset for Abstractive Screenplay Summarization](https://arxiv.org/abs/2104.07091). Chen et al, ACL 2022.
5. [**SQuALITY**: Building a Long-Document Summarization Dataset the Hard Way](https://arxiv.org/abs/2205.11465). Wang et al, EMNLP 2022.

#### Long-context question-answering (QA)

1. [The **NarrativeQA** Reading Comprehension Challenge](https://arxiv.org/abs/1712.07040). Kočiský et al, 2017.
2. [A Dataset of Information-Seeking Questions and Answers Anchored in Research Papers (**Qasper**)](https://arxiv.org/abs/2105.03011). Dasigi et al, NAACL 2021.
3. [**QuALITY**: Question Answering with Long Input Texts, Yes!](https://arxiv.org/abs/2112.08608). Pang et al, NAACL 2022.
4. [**MuSiQue**: Multihop Questions via Single-hop Question Composition](https://arxiv.org/abs/2108.00573) Trivedi et al, TACL 2022.

### Logical reasoning (maths, coding, etc)

1. [Evaluating Large Models Trained on Code (**HumanEval** benchmark)](https://arxiv.org/abs/2107.03374). Mark Chen et al, 2021. 164 hand-written programming problems. Each problem includes a function signature, docstring, body, and several unit tests, with an average of 7.7 tests per problem.
2. [Training Verifiers to Solve Math Word Problems (**GSM8K** benchmark)](https://arxiv.org/abs/2110.14168). Karl Cobbe et al, 2021. 8.5K (7.5k training + 1k test) high quality grade school math problems created by human problem writers. Each problem takes between 2 and 8 steps to solve, and solutions primarily involve performing a sequence of elementary calculations using basic arithmetic operations.

### Specific NLP tasks

#### Question-answering (QA)

##### Reading comprehension

1. [**SQuAD**: 100,000+ Questions for Machine Comprehension of Text](https://arxiv.org/abs/1606.05250). Rajpurkar et al, EMNLP 2016
2. [Know What You Don't Know: Unanswerable Questions for SQuAD (**SQuAD 2.0**)](https://arxiv.org/abs/1806.03822). Rajpurkar et al, ACL 2018.
3. [**QuAC**: Question Answering in Context](https://arxiv.org/abs/1808.07036). Choi et al, EMNLP 2018.
4. [**BoolQ**: Exploring the Surprising Difficulty of Natural Yes/No Questions](https://aclanthology.org/N19-1300/). Christopher Clark et al, NAACL 2019. 16,000 naturally occurring yes/no questions, split in 9.4k train, 3.2k dev, 3.2k test. Each question is paired with a Wikipedia passage. 

##### Commonsense reasoning

1. [Think you have Solved Question Answering? Try **ARC**, the AI2 Reasoning Challenge](https://arxiv.org/abs/1803.05457). Clark et al, 2018.
2. [Can a Suit of Armor Conduct Electricity? A New Dataset for Open Book Question Answering (**OpenBookQA**)](https://arxiv.org/abs/1809.02789). Mihaylov et al, EMNLP 2018.
3. [**CommonsenseQA**: A Question Answering Challenge Targeting Commonsense Knowledge](https://arxiv.org/abs/1811.00937). Talmor et al, NAACL 2019.
4. [**HellaSwag**: Can a Machine Really Finish Your Sentence](https://arxiv.org/abs/1905.07830). Zellers et al, ACL 2019.
5. [**WinoGrande**: An Adversarial Winograd Schema Challenge at Scale](https://arxiv.org/abs/1907.10641). Sakaguchi et al, 2019.
6. [SocialIQA: Commonsense Reasoning about Social Interactions (**SIQA**)](https://arxiv.org/abs/1904.09728). Sap et al, EMNLP 2019.
7. [**PIQA**: Reasoning about Physical Commonsense in Natural Language](https://arxiv.org/abs/1911.11641). Bisk et al, AAAI 2020.

##### World knowledge

1. [**TriviaQA**: A Large Scale Distantly Supervised Challenge Dataset for Reading Comprehension](https://arxiv.org/abs/1705.03551). Joshi et al, 2017. 
2. [**Natural Questions**: A Benchmark for Question Answering Research](https://aclanthology.org/Q19-1026/). Tom Kwiatkowski et al, TACL 2019. 307,373 training examples with single annotations; 7,830 development examples with 5-way annotations and 7,842 test examples with 5-way annotations. Questions are real anonymized, aggregated queries issued to the Google search engine. Each question is paired with an entire Wikipedia page.
3. [**TruthfulQA**: Measuring How Models Mimic Human Falsehoods](https://arxiv.org/abs/2109.07958). Stephanie Lin et al, ACL 2022. 817 questions spanning 38 categories. Question and answers are hand-written by human annotators and designed to elicit imitative falsehoods.

## Fine-tuning / instruction-tuning datasets

1. [AgentTuning: Enabling Generalized Agent Abilities For LLMs](https://arxiv.org/pdf/2310.12823.pdf). Zeng et al., 2023. Introduce **AgentInstruct** dataset: 1,866 high quality interaction trajectories generated by GPT-4 and verified by human.
2. [Enhancing Chat Language Models by Scaling High-quality Instructional Conversations](https://arxiv.org/abs/2305.14233). Ning Ding et al, 2023. Introduce **UltraChat** dataset: 1.5 million high-quality multi-turn dialogues covering a wide range of topics and instructions.
3. [**OpenAssistant Conversations** -- Democratizing Large Language Model Alignment](https://arxiv.org/abs/2304.07327). Andreas Köpf et al, 2023. 161,443 messages (91,829 prompter and 69,614 assistant messages) distributed across 66,497 conversation trees, in 35 different languages, annotated with 461,292 quality ratings.


# High-performing open-source LLMs

In the following, we report cases where an open-source LLM (e.g., Llama-2) outperforms an OpenAI, paying LLM (e.g., ChatGPT). To maintain conciseness, we follow the following reporting guidelines:  
-Only report the highest performing version of the open-source LLM.  
-Only report the highest performing version of the OpenAI model which is outperformed by the open-source LLM.  
-Average results over all datasets where the open-source LLM is better than the OpenAI LLM. This implies excluding reported results on datasets where the proposed LLM underperforms all OpenAI LLMs.   
We refer the reader to the respective papers for more details.  

We split LLMs depending on the type of training performed:  
-**Pre-training** refers to LLMs pre-trained from scratch.  
-**Continual pre-training** refers to LLMs initialized from an already pre-trained LLM (e.g, Llama-2) and then undergoing another phase of pre-training.  
-**Instruction tuning** are LLMs trained with supervised fine-tuning on instruction tuning datasets or standard downstream tasks datasets.  
-**Inference** designates proposed techniques which drive LLM performance while not changing the model weights.   

Note that a proposed LLM may fall into several of the above 4 categories. In that case, we place it into the most computationally intensive category: for instance, a paper proposing both to continue pre-training Llama-2 and to fine-tune on a new, instruction-tuning dataset will land in the Continual pre-training category.  

## Pre-training (from scratch)

| **LLM**     | **Date released** | **LLM size** | **Task(s)** | **OpenAI model** | **OpenAI result** | **LLM result** | **Gain (%)** |
|-------------|-------------------|--------------|-------------|------------------|-------------------|----------------|--------------|
| **Phi-1** [[paper](https://arxiv.org/abs/2306.11644)] | June 20th, 2023 | 1.3B | HumanEval | GPT-3.5 | 47.0 | 50.6 | +7.7% |
| **Yi** [[github](https://github.com/01-ai/Yi)] | Nov 5th, 2023 | 34B | MMLU | GPT-3.5 | 70.0 | 76.3 | +9.0% |

## Continual pre-training

| **LLM**     | **Date released** | **Pre-training** | **LLM size** | **Task(s)** | **OpenAI model** | **OpenAI result** | **LLM result** | **Gain (%)** |
|-------------|-------------------|------------------|--------------|---------------------------|------------------|-------------------|----------------|--------------|
| **Llama-2-Long-Chat** [[paper](https://arxiv.org/abs/2309.16039)] | Sept 27th, 2023 | Llama-2 + 400B tokens | 70B | ZeroScrolls | GPT-3.5-turbo-16k | 36.7 | 37.7 | +2.7% |
| **Lemur** [[paper](https://arxiv.org/abs/2310.06830)] | Oct 10th, 2023 | Llama-2 + 90B tokens | 70B | HumanEval + GSM8K | GPT-3.5-turbo | 40.77 | 52.50 | +28.8% |
| **InstructRetro** [[paper](https://arxiv.org/abs/2310.07713)] | Oct 11th, 2023 | GPT + 100B tokens | 48B | SQuAD-2.0 | GPT-3 | 59.5 | 75.6 | +27.1% |

## Instruction tuning

| **LLM**     | **Date released** | **Backbone LLM** | **LLM size** | **Task(s)** | **OpenAI model** | **OpenAI result** | **LLM result** | **Gain (%)** |
|-------------|-------------------|------------------|--------------|------------------|------------------|-------------------|----------------|--------------|
| **UltraLlama** [[paper](https://arxiv.org/abs/2305.14233)] | May 23rd, 2023 | LLama | 13B | TruthfulQA | ChatGPT | 9.54 | 9.62 | +0.8% |
| **WizardCoder** [[paper](https://arxiv.org/abs/2306.08568)] | June 14th, 2023 | Alpaca (LLama-2) | 15B | HumanEval | GPT-3.5 | 48.1 | 57.3 | +19.1% |
| **WizardMath** [[paper](https://arxiv.org/abs/2308.09583)] | Aug 18th, 2023 | LLama-2 | 70B | GSM8K | GPT-3.5 | 57.1 | 81.6 | +42.9% |
| **Llama-2-32k-ret** [[paper](https://arxiv.org/abs/2310.03025)] | Oct 4th, 2023 | LLama-2 | 70B | QMSum + 4 QA datasets | GPT-3.5-turbo-16k | 44.58 | 46.48 | +4.3% |
| **FireAct** [[paper](https://arxiv.org/abs/2310.05915)] | Oct 9th, 2023 | LLama-2 | 13B | HotPotQA | GPT-3.5 | 31.4 | 34.4 | +9.6% |
| **AgentLM** [[paper](https://arxiv.org/abs/2310.12823)] | Oct 19th, 2023 | LLama-2 | 70B | ALFWorld + KG + Database + HotPotQA + GSM8K | GPT-4 | 46.68 | 54.40 | +16.5% |

## Inference

| **LLM**     | **Date released** | **LLM size** | **Task(s)** | **OpenAI model** | **OpenAI result** | **LLM result** | **Gain (%)** |
|-------------|-------------------|--------------|-------------|------------------|-------------------|----------------|--------------|
| **CoVe** [[paper](https://arxiv.org/abs/2309.11495)] | Sept 20th, 2023 | 1.3B | Biographies | GPT-3.5 | 58.7 | 71.4 | +21.6% |
