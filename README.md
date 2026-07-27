# ReliaBench Benchmark

ReliaBench - A multi-label benchmark for reliability-related knowledge classification


## ReliaBench Composition
This benchmark contains 36,000 samples annotated for reliability-related multi-label knowledge classification. The benchmark contains training, validation and test sets. These sets are located in the folder ReliaBench. 

The benchmark was conducted using the following industry and NLP benchmarks and datasets: 

1) ASRS -> Industry, aviation safety -> Aviation Safety Reporting System (ASRS)

2) NTSB -> Industry, aviation safety -> Aviation Accident NTSB

3) Aircraft M. -> Industry, maintenance -> Aircraft Historical Maintenance 

4) OMIn -> Industry, maintenance -> Operations and Maintenance Intelligence

5) MATRES -> NLP, procedures -> Matres_processed 

6) WikiHow -> NLP, instructions -> soyasis/wikihow_small

7) DBpedia 14 -> NLP, knowledge QA -> dbpedia_14

8) MultiNLI -> NLP, inference -> multi_nli 

9) PIQA -> NLP, commonsense -> nthngdy/piqa 

10) HotpotQA -> NLP, knowledge QA -> hotpotqa/hotpot_qa, distractor subset

11) MultiFC -> NLP, fact checking -> pszemraj/multi_fc 

<br>


The benchmark is composed with the following percentages:

1) Aviation safety (ASRS + NTSB) -> 55%

2) Maintenance (Aircraft M.+ OMIn)	-> 21.11%

3) Procedures (WikiHow + MATRES) -> 9.59%

4) Fact checking and inference (MultiFC + MultiNLI) -> 5.58%

5) Knowledge QA (DBpedia + HotpotQA) -> 7.03%

6) Commonsense and reasoning (PIQA) -> 1.71%

<br>

Within the benchmark the label distribution is as following: 

1) 51.3% of samples are single-labeled

2) 38.28% are multi-labeled

3) 10.42% are “none” labeled

## Reliability-Related Knowledge Categories
Within the benchmark, we annotaed the following six reliability-related knowledge types in a multi-label annotation with weak snorkel annotation. The used snorkel words are available in the snorkel_words.py file. Additionally, we generated a manual gold annotation of 1,5000 samples for every test, val and train split. 

1) **State Deviation**: Knowledge about departures from expected states and the associated conditions, e.g., the hydraulic pump failed due to seal degradation;

2) **Process Flow**: Knowledge about temporal or logical ordering and dependencies among events, actions, or processes, e.g., the engine ignition is initiated after the fuel valve is opened; 

3) **Maintenance**: Knowledge about actions taken to modify, restore, improve, or maintain system behavior, e.g., the recalibration in a navigation system to restore accuracy; 

4) **Operational**: Knowledge about the conditions and parameters influencing system behavior, e.g., the decrease of system performance if the temperature is over a  threshold; 

5) **Structural**: Knowledge about component associations, system hierarchies, and relationships, e.g., that  system A consists of tanks, pumps and filters; 

6) **Temporal and Probabilistic**: Knowledge about time-related characteristics and stochastic behavior, e.g., the battery loads in a normal distribution with a mean of 70 minutes. 

## Benchmark Objective and Evaluation Metrics

The benchmark objective is to evaluate the ability of fine-tuned language models to correctly classify textual expert input into defined reliability-related knowledge categories and to compare its performance against a baseline using key performance indicators (KPIs). We select the standard NLP text classification metrics 

1) micro F1 to assess overall predictive performance, 

2) macro F1 to evaluate class-balanced performance by weighing classes equally, 

3) hamming loss to measure incorrectly predicted labels, 

4) per-class F1 to identify classes that are difficult to classify, and 

5) confusion matrix to identify which classes are misclassified as others. 

Together, these KPIs can provide insights to analyse classification effectiveness, errors and class-wise robustness to select suitable approaches.

## Model Training

The code for the fine-tuning of the selected pre-trained language model on the benchmark is located in the folder model_training. The models are each trained with a seed of 21, 42 and 123. 

## Model Benchmarking
To support in-domain and out-of-domain benchmarking, the benchmark provides three test sets each with 3,000 samples located in the folder ReliaBench/test:

1) an in-domain test set containing samples from the same established industry and NLP datasets utilized for training and validation

2) an out-of-domain test set containing unseen samples from established NLP datasets.

3) an out-of-domain test set containing unseen manufacturing-specific samples. The manufacturing domain is not present in the training and validation samples, which supports unseen domain benchmarking.

The code for the model benchmarking of both the baseline and the fine-tuned pre-trained language models are located in the folder model_benchmarking.



## Getting started

### Via git

```bash
git clone https://github.com/SYDSEN-KIT/ReliaBench.git  # clone repository
```

For benchmark use 

```bash
cd ReliaBench # navigate into the benchmark folder
```

For model training

```bash
cd model_training # navigate into the model training folder
python bart_123.py # select the model you want to fine-tune
```

For model benchmarking

```bash
cd model_benchmarking # navigate into the model benchmarking folder
python fine-tuned_model_benchmarking.py # select the models you want to benchmark
```

## Usage & Attribution

If you are using the tool for a scientific project please consider citing the following paper:

Jungmann M, Lazarova-Molnar S. 2026. "Benchmarking Text Classification Approaches for Expert Knowledge Integration in Petri Net-Based Digital Twin Models." 12th. Federation of European Simulation Societies Conference (Eurosim 2026)
