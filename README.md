# OpenSource-LLMs-better-than-ChatGPT


# Datasets

## Evaluation datasets

### 1/ General capabilities

1. [LLM-as-a-judge with **MT-Bench** and Chatbot Arena](https://arxiv.org/abs/2306.05685). Lianmin Zheng et al, Neurips 2023. 80 multi-turn questions from 8 categories: writing, roleplay, extraction, reasoning, math, coding, knowledge I (STEM), and knowledge II (humanities/social science).
2. [**AlpacaEval**](https://github.com/tatsu-lab/alpaca_eval). 805 instructions to follow from the AlpacaFarm evaluation set.
3. [**Open LLM Leaderboard**](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard). Live leaderboard ranking LLMs on ARC, HellaSwag, MMLU, TruthfulQA, Winogrande, GSM8K.
4. [Beyond the Imitation Game: Quantifying and extrapolating the capabilities of language models (**Big-Bench**)](https://arxiv.org/abs/2206.04615). Aarohi Srivastava et al, TLMR 2023. Large benchmark of 204 tasks, contributed by 450 authors across 132 institutions, including tasks from: linguistics, childhood development, maths, common-sense reasoning, biology, physics, etc.
5. [Large Language Models are not Fair Evaluators (**FairEval-Vicuna**)](https://arxiv.org/abs/2305.17926). Peiyi Wang et al, 2023. 80 questions from the Vicuna Benchmark, with multiple evidence calibration and balanced position calibration.

### 2/ Agent capabilities

#### Tool usage

1. [**API-Bank**: A Comprehensive Benchmark for Tool-Augmented LLMs](https://arxiv.org/abs/2304.08244). Minghao Li et al, EMNLP 2023.
2. [On the Tool Manipulation Capability of Open-source Large Language Models (**ToolBench**)](https://arxiv.org/abs/2305.16504). Qiantong Xu et al, 2023.
3. [Gorilla: Large Language Model Connected with Massive APIs (**APIBench**)](https://arxiv.org/abs/2305.15334). Shishir G. Patil et al, 2023.
4. [**ToolAlpaca**: Generalized Tool Learning for Language Models with 3000 Simulated Cases](https://arxiv.org/abs/2306.05301). Qiaoyu Tang et al, 2023.
5. [**MINT**: Evaluating LLMs in Multi-turn Interaction with Tools and Language Feedback](https://arxiv.org/abs/2309.10691). Xingyao Wang et al, 2023.

#### Self-debugging

1. [InterCode: Standardizing and Benchmarking Interactive Coding with Execution Feedback (**InterCode-Bash** and **InterCode-SQL**)](https://arxiv.org/abs/2306.14898). John Yang et al, 2023.
2. [MINT: Evaluating LLMs in Multi-turn Interaction with Tools and Language Feedback (**MINT-MBPP** and **MINT-HumanEval**)](https://arxiv.org/abs/2309.10691). Xingyao Wang et al, 2023.
3. [Code as Policies: Language Model Programs for Embodied Control (**RoboCodeGen**)](https://arxiv.org/abs/2209.07753). Jacky Liang et al, 2023.

#### Following feedback 

1. [**MINT**: Evaluating LLMs in Multi-turn Interaction with Tools and Language Feedback](https://arxiv.org/abs/2309.10691). Xingyao Wang et al, 2023.

#### Exploring environment

1. [**ALFWorld**: Aligning Text and Embodied Environments for Interactive Learning](https://arxiv.org/abs/2010.03768). Mohit Shridhar et al, ICLR 2021.
2. [InterCode: Standardizing and Benchmarking Interactive Coding with Execution Feedback (**InterCode-CTF**)](https://arxiv.org/abs/2306.14898). John Yang et al, 2023.
3. [**WebArena**: A Realistic Web Environment for Building Autonomous Agents](https://arxiv.org/abs/2307.13854). Shuyan Zhou et al, 2023.

### 3/ Logical reasoning (maths, coding, etc)

1. [Evaluating Large Models Trained on Code (**HumanEval** benchmark)](https://arxiv.org/abs/2107.03374). Mark Chen et al, 2021. 164 hand-written programming problems. Each problem includes a function signature, docstring, body, and several unit tests, with an average of 7.7 tests per problem.
2. [Training Verifiers to Solve Math Word Problems (**GSM8K** benchmark)](https://arxiv.org/abs/2110.14168). Karl Cobbe et al, 2021. 8.5K (7.5k training + 1k test) high quality grade school math problems created by human problem writers. Each problem takes between 2 and 8 steps to solve, and solutions primarily involve performing a sequence of elementary calculations using basic arithmetic operations.

### 4/ Long-context (summarization, QA, etc)

#### General long-context benchmarks 

1. [**Long Range Arena**: A Benchmark for Efficient Transformers](https://arxiv.org/abs/2011.04006). Yi Tay et al, ICLR 2021. Benchmark of 6 tasks, each between 1k and 16k input tokens. Tasks encompass several modalities: text, images, spatial reasoning.
2. [**SCROLLS**: Standardized CompaRison Over Long Language Sequences](https://arxiv.org/abs/2201.03533). Uri Shaham et al, EMLP 2022. Benchmark made of 7 existing long-input datasets: 2 summarization datasets ([GovReport](https://arxiv.org/abs/2104.02112) and [SummScreenFD](https://arxiv.org/abs/2104.07091)), 1 query-focused summarization dataset ([QMSum](https://arxiv.org/abs/2104.05938)), 3 QA datasets ([Qasper](https://arxiv.org/abs/2105.03011), [NarrativeQA](https://arxiv.org/abs/1712.07040), [QuALITY](https://arxiv.org/abs/2112.08608), ), and 1 NLI dataset ([ContractNLI](https://arxiv.org/abs/2110.01799)). 
3. [**ZeroSCROLLS**: A Zero-Shot Benchmark for Long Text Understanding](https://arxiv.org/abs/2305.14196). Uri Shaham et al, EMNLP 2023. Extension of SCROLLS focusing on zero-shot evaluation. Compared to SCROLLS, ZeroScrolls discards ContractNLI, and adds 1 query-based summarization tasks ([SQuALITY](https://arxiv.org/abs/2205.11465)), 1 QA dataset ([MuSiQue](https://arxiv.org/abs/2108.00573)) and 2 custom aggregation tasks ([SpaceDigest](https://arxiv.org/abs/2012.04443), [BookSumSort](https://arxiv.org/abs/2105.08209)).
4. [**LongBench**: A Bilingual, Multitask Benchmark for Long Context Understanding](https://arxiv.org/abs/2308.14508). Yushi Bai et al, 2023. Bilingual English/Chinese, multi-task benchmark for long context understanding. 21 datasets across 6 task categories, with an average length of 6,711 words (English) and 13,386 characters (Chinese). The tasks cover key long-context tasks such as single-doc QA, multi-doc QA, summarization, few-shot learning, synthetic tasks, and code completion.
5. [**L-Eval**: Instituting Standardized Evaluation for Long Context Language Models](https://arxiv.org/abs/2307.11088). Chenxin An et al, 2023. 20 long-input tasks covering diverse aspects. 4 are built from scratch, 4 are re-annotation of existing datasets, and 12 are manually filtered existing datasets.
6. [**BAMBOO**: A Comprehensive Benchmark for Evaluating Long Text Modeling Capacities of Large Language Models](https://arxiv.org/abs/2309.13345). Zican Dong et al, 2023. 10 datasets from 5 tasks, all designed to avoid pre-training data contamination by collecting evaluation data in recent period (2023).
7. [**M4LE**: A Multi-Ability Multi-Range Multi-Task Multi-Domain Long-Context Evaluation Benchmark for Large Language Models](https://arxiv.org/abs/2310.19240v1). Wai-Chung Kwan et al, 2023. 36 datasets covering 11 tasks and 12 domains, in English and Chinese. Datasets are split in 5 abilities of understanding: explicit single-span, semantic single-span, explicit multiple-span, semantic multiple-span, and global. 

#### Long-context (generic) summarization 

1. [**BookSum**: A Collection of Datasets for Long-form Narrative Summarization](https://arxiv.org/abs/2105.08209). Kryściński et al, 2021. Colletion of datasets resulting in 46,532 paragraph-level, 12,630 chapter-level (**BookSum-Chapter**), and 405 book-level summarization data points.
2. [Efficient Attentions for Long Document Summarization (**GovReport**)](https://arxiv.org/abs/2104.02112). Huang et al, NAACL 2021. 19,466 documents split into 17519 training, 974 validation and 973 test samples. Average length is 9409.4 words per document and 553.4 words per summary.
3. [**SummScreen**: A Dataset for Abstractive Screenplay Summarization](https://arxiv.org/abs/2104.07091). Chen et al, ACL 2022. 22,503 episodes from TVMegaSite (**SummScreen-TMS**, split into 18,915/1,795/1,793 train/dev/test) and 4,021 episodes from ForeverDreaming (**SummScreen-FD**, split into 3,673/338/337 train/dev/test). 

#### Long-context (query-focused) summarization 

1. [**QMSum**: A New Benchmark for Query-based Multi-domain Meeting Summarization](https://arxiv.org/abs/2104.05938). Ming Zhong et al, NAACL 2021. 1,808 query-summary pairs over 232 meetings in multiple domains. Meetings are from 3 categories: product, academic, committee ; and are annotated by AMT workers.
2. [**SQuALITY**: Building a Long-Document Summarization Dataset the Hard Way](https://arxiv.org/abs/2205.11465). Wang et al, EMNLP 2022. 100 stories, 500 questions, and 2000 summaries (there are 4 reference summaries per question).

#### Long-context question-answering (QA)

1. [The **NarrativeQA** Reading Comprehension Challenge](https://arxiv.org/abs/1712.07040). Kočiský et al, 2017. 46,765 question–answer pairs from 1,567 stories (1,102/115/355 train/valid/test) from books and movie scripts. 
2. [A Dataset of Information-Seeking Questions and Answers Anchored in Research Papers (**Qasper**)](https://arxiv.org/abs/2105.03011). Dasigi et al, NAACL 2021. 5,049 questions (2,593/1,005/1,451 train/valid/test) over 1,585 NLP papers. Each question is written by an NLP practitioner who read only the title and abstract of the corresponding paper, and the question seeks information present in the full text.
3. [**QuALITY**: Question Answering with Long Input Texts, Yes!](https://arxiv.org/abs/2112.08608). Pang et al, NAACL 2022. Multiple-choice QA dataset with context passages in English that have an average length of about 5,000 tokens. 6,737 questions split into 2,523/2,086/2,128 train/dev/test. 
4. [**MuSiQue**: Multihop Questions via Single-hop Question Composition](https://arxiv.org/abs/2108.00573) Trivedi et al, TACL 2022. Multihop QA dataset with 25K 2-4 hop questions, split into 19,938/2,417/2,459 train/dev/test. 

### 5/ Specific NLP tasks

#### Question-answering (QA)

##### Reading comprehension

1. [**SQuAD**: 100,000+ Questions for Machine Comprehension of Text](https://arxiv.org/abs/1606.05250). Rajpurkar et al, EMNLP 2016. 100k+ questions asked by crowdworkers on a set of Wikipedia articles, where the answer to each question is a segment of text from the corresponding reading passage.
2. [Know What You Don't Know: Unanswerable Questions for SQuAD (**SQuAD 2.0**)](https://arxiv.org/abs/1806.03822). Rajpurkar et al, ACL 2018. SQuAD 2.0 combines existing SQuAD data with over 50k unanswerable questions written adversarially by crowdworkers to look similar to answerable ones.
3. [**QuAC**: Question Answering in Context](https://arxiv.org/abs/1808.07036). Choi et al, EMNLP 2018. 100k questions (83,568/7,354/7,353 train/dev/test) from 14K information-seeking QA dialogs. 
4. [**BoolQ**: Exploring the Surprising Difficulty of Natural Yes/No Questions](https://aclanthology.org/N19-1300/). Christopher Clark et al, NAACL 2019. 16k naturally occurring yes/no questions, split in 9.4k train, 3.2k dev, 3.2k test. Each question is paired with a Wikipedia passage. 

##### Commonsense reasoning

1. [Think you have Solved Question Answering? Try **ARC**, the AI2 Reasoning Challenge](https://arxiv.org/abs/1803.05457). Clark et al, 2018. 7,787 natural, grade-school science questions (authored for human tests), split into 3,370/869/3,548 train/dev/test.
2. [**HotpotQA**: A Dataset for Diverse, Explainable Multi-hop Question Answering](https://arxiv.org/abs/1809.09600). Zhilin Yang et al, EMNLP 2018. 113k QA pairs from Wikipedia which require reasoning from multiple documents. 
3. [Can a Suit of Armor Conduct Electricity? A New Dataset for Open Book Question Answering (**OpenBookQA**)](https://arxiv.org/abs/1809.02789). Mihaylov et al, EMNLP 2018. Dataset modeled after open book exams for assessing human understanding of a subject. Around 6k questions (4957/500/500 train/dev/test) probe an understanding of 1,329 elementary level science facts. 
4. [**CommonsenseQA**: A Question Answering Challenge Targeting Commonsense Knowledge](https://arxiv.org/abs/1811.00937). Talmor et al, NAACL 2019. 12,247 multiple-choice questions that mention the source concept and discriminate in turn between each of the target concepts. 
5. [**HellaSwag**: Can a Machine Really Finish Your Sentence](https://arxiv.org/abs/1905.07830). Zellers et al, ACL 2019. Questions collected with Adversarial Filtering (AF), a data collection paradigm wherein a series of discriminators iteratively select an adversarial set of machine-generated wrong answers.
6. [**WinoGrande**: An Adversarial Winograd Schema Challenge at Scale](https://arxiv.org/abs/1907.10641). Sakaguchi et al, 2019. Large-scale dataset of 44k problems, inspired by the original Winograd Schema Challenge (WSC), a benchmark for commonsense reasoning made of 273 expert-crafted pronoun resolution problems originally designed to be unsolvable for statistical models that rely on selectional preferences or word associations. 12,282 instances split into 9,248/1,267/1,767 train/dev/test sets.
7. [SocialIQA: Commonsense Reasoning about Social Interactions (**SIQA**)](https://arxiv.org/abs/1904.09728). Sap et al, EMNLP 2019. 38k (33,410/1,954/2,224 train/dev/test) multiple-choice commonsense questions along with correct and incorrect answers about social interactions collected through crowdsourcing. 
8. [**PIQA**: Reasoning about Physical Commonsense in Natural Language](https://arxiv.org/abs/1911.11641). Bisk et al, AAAI 2020. Benchmarking progress in physical commonsense understanding, with 16k/2k/3k train/dev/test QA pairs. 

##### World knowledge

1. [**TriviaQA**: A Large Scale Distantly Supervised Challenge Dataset for Reading Comprehension](https://arxiv.org/abs/1705.03551). Joshi et al, 2017. 95K question-answer pairs authored by trivia enthusiasts and independently gathered evidence documents, six per question on average.
2. [**Natural Questions**: A Benchmark for Question Answering Research](https://aclanthology.org/Q19-1026/). Tom Kwiatkowski et al, TACL 2019. 307,373 training examples with single annotations; 7,830 development examples with 5-way annotations and 7,842 test examples with 5-way annotations. Questions are real anonymized, aggregated queries issued to the Google search engine. Each question is paired with an entire Wikipedia page.
3. [**TruthfulQA**: Measuring How Models Mimic Human Falsehoods](https://arxiv.org/abs/2109.07958). Stephanie Lin et al, ACL 2022. 817 questions spanning 38 categories. Question and answers are hand-written by human annotators and designed to elicit imitative falsehoods.

##### Specific domain

1. [**MedMCQA**: A Large-scale Multi-Subject Multi-Choice Dataset for Medical domain Question Answering](https://arxiv.org/abs/2203.14371). Ankit Pal et al, ACM CHIL 2022. 194k multiple-choice questions from real world medical entrance exams.

#### 6/ Trustworthy AI

1. [**FActScore**: Fine-grained Atomic Evaluation of Factual Precision in Long Form Text Generation](https://arxiv.org/abs/2305.14251). Sewon Min et al, EMNLP 2023. Evaluation of LLMs at generating people biographies, using precision of atomic facts.

## Fine-tuning / instruction-tuning datasets

1. [AgentTuning: Enabling Generalized Agent Abilities For LLMs](https://arxiv.org/pdf/2310.12823.pdf). Zeng et al., 2023. Introduce **AgentInstruct** dataset: 1,866 high quality interaction trajectories generated by GPT-4 and verified by human.
2. [Enhancing Chat Language Models by Scaling High-quality Instructional Conversations](https://arxiv.org/abs/2305.14233). Ning Ding et al, 2023. Introduce **UltraChat** dataset: 1.5 million high-quality multi-turn dialogues covering a wide range of topics and instructions.
3. [**OpenAssistant Conversations** -- Democratizing Large Language Model Alignment](https://arxiv.org/abs/2304.07327). Andreas Köpf et al, 2023. 161,443 messages (91,829 prompter and 69,614 assistant messages) distributed across 66,497 conversation trees, in 35 different languages, annotated with 461,292 quality ratings.


# Open-source LLMs vs ChatGPT

In the following, we report cases where an open-source LLM (e.g., Llama-2) outperforms an OpenAI, paying LLM (e.g., ChatGPT). To maintain conciseness, we only report the highest performing version of the open-source LLM (which usually is the largest model, when there are several backbone sizes being tested). 

We categorize LLMs depending on the type of training performed:  
-**Pre-training (PT)** refers to LLMs pre-trained from scratch.  
-**Continual pre-training (CPT)** refers to LLMs initialized from an already pre-trained LLM (e.g, Llama-2) and then undergoing another phase of pre-training.  
-**Fine-tuning or Instruction tuning (FT)** are LLMs trained with supervised fine-tuning on instruction tuning datasets or standard downstream tasks datasets.  
-**Inference (INF)** designates proposed techniques which drive LLM performance while not changing the model weights.   

Note that a proposed LLM may fall into several of the above 4 categories. 


## General capabilities (Table 1)

| **LLM**     | **Date released** | **LLM size** | **Training** | **MT-Bench** | **AlpacaEval** | **Open LLM LB** |
|-------------|-------------------|--------------|-------------|---------------|----------------|-----------------|
| **GPT-3.5-turbo** | Nov 2022 | ? | ? | 7.94 | 81.71 | 70.21 | 
| **GPT-4** | March 2023 | ? | ? | 8.99 | 95.28 | 85.36 | 
|-------------|-------------------|--------------|-------------|---------------|----------------|-----------------|
| **WizardLM** [[paper](https://arxiv.org/abs/2304.12244)] | April 24th 18th, 2023 | 70B | FT | 7.71 | 92.91 | _ | 
| **Llama-2-chat** [[paper](https://arxiv.org/abs/2307.09288)] | July 18th, 2023 | 70B | FT | 6.86 | 92.66 | _ | 
| **Godzilla** [[HF card](https://huggingface.co/MayaPH/GodziLLa2-70B)] | Aug 11th, 2023 | 70B | FT | _ | _ | 67.01 | 
| **Zephyr** [[paper](https://arxiv.org/abs/2310.16944)] | Oct 25th, 2023 | 70B | FT | 7.34 | 90.60 | 52.15 | 
| **Yi-chat** [[HF card](https://huggingface.co/01-ai/Yi-34B-Chat)] | Nov 23rd, 2023 | 34B | FT | - | - | 68.68 | 


## Agent capabilities (Table 2)

| **LLM**     | **Date released** | **LLM size** | **Training** | **ALFWorld** | **InterCode-CTF** | **WebArena** | **Code Generation** |
|-------------|-------------------|--------------|--------------|--------------|------------|---------------|---------------------|
| **GPT-3.5-turbo** | Nov 2022 | ? | ? | 41.79 | 11.00 | 7.38 | 9.56 |
| **GPT-4** | March 2023 | ? | ? | 84.33 | 37.00 | 10.59 | _ |
|-------------|-------------------|--------------|--------------|--------------|------------|---------------|---------------------|
| **Lemur-chat** [[paper](https://arxiv.org/abs/2310.06830)] | Oct 10th, 2023 | 70B | CPT + FT | 59.70 | 22.00 | 5.30 | 17.65


## Logical reasoning (Table 3)

| **LLM**     | **Date released** | **LLM size** | **Training** | **GSM8K** | **HumanEval** | 
|-------------|-------------------|--------------|--------------|--------------|------------|
| **GPT-3.5-turbo** | Nov 2022 | ? | ? | 57.1 | 48.1 | 
| **GPT-4** | March 2023 | ? | ? | 92.0 | 67.0 | 
|-------------|-------------------|--------------|--------------|--------------|------------|
| **WizardCoder** [[paper](https://arxiv.org/abs/2306.08568)] | June 14th, 2023 | 15B | FT | _ | 57.3 |
| **Phi-1** [[paper](https://arxiv.org/abs/2306.11644)] | June 20th, 2023 | 1.3B | PT + FT | _ | 50.6 |
| **WizardMath** [[paper](https://arxiv.org/abs/2308.09583)] | Aug 18th, 2023 | 70B | FT | 81.6 | _ |
| **Lemur-chat** [[paper](https://arxiv.org/abs/2310.06830)] | Oct 10th, 2023 | 70B | CPT + FT | 66.3 | 61.0 |


## Long-context modelling on ZeroSCROLLS (Table 4)

| **LLM**     | **Date released** | **LLM size** | **Training** | **GovReport** | **SummScreen** | **QMSum** | **SQuALITY** | **Qasper** | **NarrativeQA** | **QuALITY** | **MuSiQue** | **SpaceDigest** | **BookSumSort** |  
|-------------|-------------------|--------------|--------------|---------------|----------------|-----------|--------------|------------|-----------------|-------------|-------------|-----------------|-------------------|
| **GPT-3.5-turbo** | Nov 2022 | ? | ? | 21.3 | 16.1 | 15.6 | 20.4 | 49.3 | 25.1 | 66.6 | 27.1 | 49.1 | 49.8 | 
| **GPT-4** | March 2023 | ? | ? | 26.3 | 17.3 | 18.5 | 22.6 | 50.7 | 27.6 | 89.2 | 41.1 | 62.8 | 60.5 |
| **GPT-3.5-turbo-16k** | June 2023 | ? | ? | 24.3 | 16.2 | 17.4 | 21.4 | 50.0 | 29.5 | 72.0 | 27.0 | 54.1 | 54.6 | 
|-------------|-------------------|--------------|--------------|---------------|----------------|-----------|--------------|------------|-----------------|-------------|-------------|-----------------|-------------------|
| **Llama-2-long-chat** [[paper](https://arxiv.org/abs/2309.16039)] | Sept 27th, 2023 | 70B | CPT + FT | 26.0 | 15.0 | 20.0 | 20.9 | 52.0 | 31.7 | 82.6 | 27.3 | 55.5 | 46.2 |
| **Llama-2-chat-32k + retrieval** [[paper](https://arxiv.org/abs/2310.03025)] | Oct 4th, 2023 | 70B | FT | _ | _ | 18.3 | _ | 31.3 | 24.5 | 69.6 | 26.7 | _ | _ |


## Hallucination (Table 5)

| **LLM**     | **Date released** | **LLM size** | **Training** | **TruthfulQA** | **FactScore**| **HotpotQA** | **OpenBookQA** | **MedMC-QA** | **TriviaQA** |
|-------------|-------------------|--------------|--------------|----------------|--------------|--------------|----------------|--------------|--------------|
| **GPT-3.5-turbo** | Nov 2022 | ? | ? | 47.0 | 58.7 | 24.0 | 78.3 | 44.4 | 79.3 |
|-------------|-------------------|--------------|--------------|----------------|--------------|--------------|----------------|--------------|--------------|
| text-davinci-002 + **PKG** [[paper](https://arxiv.org/abs/2305.04757)] | May 8th, 2023 | 175B | INF | _ | _ | _ | _ | 47.4 | _ |
| GPT-3.5-turbo + **CRITIC** [[paper](https://arxiv.org/abs/2305.11738)] | May 19th, 2023 | _ | INF | _ | _ | 38.7 | _ | _ | 75.1 |
| text-davinci-002 + **LMvsLM** [[paper](https://arxiv.org/abs/2305.13281)] | May 22nd, 2023 | 175B | INF | _ | _ | _ | _ | _ | 83.1 |
| GPT-3.5-turbo + **CoK** [[paper](https://arxiv.org/abs/2305.13269)] | May 22nd, 2023 | _ | INF | _ | _ | 35.4 | _ | 73.3 | _ |
| **Platypus** [[paper](https://arxiv.org/abs/2308.07317)] | Aug 14th, 2023 | 70B | FT | 62.3 | _ | _ | _ | _ | _ |
| GPT-3.5-turbo + **KSL** [[paper](https://arxiv.org/abs/2309.03118)] | Sept 6th, 2023 | _ | FT | _ | _ | _ | 81.6 | _ | _ |
| LLama + **CoVe** [[paper](https://arxiv.org/abs/2309.11495)] | Sept 20th, 2023 | 65B | FT + INF | _ | 71.4 | _ | _ | _ | _ | 
