# EconEvals: Benchmarks and Litmus Tests for Economic Decision-Making by LLM Agents

This repo contains the code associated with the paper *EconEvals: Benchmarks and Litmus Tests for Economic Decision-Making by LLM Agents* ([paper link](https://arxiv.org/pdf/2503.18825)).

Authors: [Sara Fish](sarafish.com), [Julia Shephard](https://www.linkedin.com/in/julia-shephard-80a521245)\*, [Minkai Li](https://www.linkedin.com/in/minkai-li-b511b2250)\*, [Ran Shorrer](https://rshorrer.weebly.com/), [Yannai A. Gonczarowski](https://yannai.gonch.name/scientific/) (\* = equal contribution)

For questions please contact Sara Fish ([sfish@g.harvard.edu](mailto:sfish@g.harvard.edu)). 

# Overview 

EconEvals consists of three benchmarks—procurement, scheduling, and pricing—and three litmus tests—efficiency vs. equality, patience vs. impatience, and collusiveness vs. competitiveness.

## EconEvals benchmarks 

Each benchmark consists of a synthetically generated environment with a scalable difficulty level. In the paper we instantiate three difficulty levels—Basic, Medium, and Hard—and run the benchmarks on eight LLMs—Claude-3.5 Sonnet, Gemini 1.5 Pro, GPT-4o, GPT-4.1, o4-mini, Gemini 2.5 Pro, GPT-5, and Gemini 3 Pro Preview. For more information see Sections 3-4 of the paper.

## EconEvals litmus tests 

Each litmus test consists of a synthetically generated environment that tests an LLM's response to a particular tradeoff. In the paper we run the litmus tests on sixteen LLMs—Claude 3.5 Sonnet, Claude 4 Sonnet, Claude 4.1 Opus, Claude 4.5 Haiku, Claude 4.5 Sonnet, Claude 4.5 Opus, Gemini 1.5 Pro, Gemini 2.5 Flash, Gemini 2.5 Pro, Gemini 3 Pro Preview, GPT-4o, GPT-4.1, o4-mini, GPT-5 Mini, GPT-5, and Grok 3 Beta. For more information see Sections 5-6 of the paper.

# Setup instructions

We use Python 3.12. 

**Step 0.** Set up your own virtual environment, if applicable. 

**Step 1.** Install repo as package
```
pip install -e .
```

**Step 2.** Install dependencies.

Option 1:
```
pipenv install
```
Option 2 (package versions may not match exactly):
```
pip install openai anthropic google-generativeai tenacity scikit-learn numpy gurobipy pandas inflect matplotlib seaborn
```

**Step 3.** Set environment variables `OPENAI_API_KEY_ECON_EVALS`, `ANTHROPIC_API_KEY_ECON_EVALS`, `GOOGLE_API_KEY_ECON_EVALS` as needed.

**Step 4.** (optional) Run unit tests (takes about 2min)
```
python3 -m unittest
```

# Overview of repo

- `experiments/`
    - `procurement/`: Code for procurement benchmark.
    - `scheduling/`: Code for scheduling benchmark.
    - `pricing/`: Code for pricing benchmark.
    - `efficiency_vs_equality/`: Code for efficiency versus equality litmus test.
    - `patience_vs_impatience/`: Code for patience versus impatience litmus test.
    - `collusiveness_vs_competitiveness/`: Code for collusiveness versus competitiveness litmus test.
- `tests/`: unit tests for environments.
- `utils/`: tools for LLM calling. Currently supports OpenAI, Anthropic, Google API.

# Instructions for running EconEvals on other LLMs or with modified enviornments

## Running EconEvals with other LLMs 

All of the LLM logic can be found in `utils/llm_tools.py`. To run EconEvals with a different LLM, it suffices to modify `call_llm` to work with that LLM.

### OpenAI, Anthropic, or Google LLMs.

The existing code already provides broad support for the OpenAI, Anthropic, and Google APIs. In many cases, to add support e.g. for a new OpenAI model, it may be sufficient to add that model name to the list `OPENAI_GPT_MODELS` at the top of the file. In some cases, OpenAI, Anthropic, and Google may slightly alter LLM API syntax, in which case you may need to add functionality to the appropriate provider-specific function (`call_openai`, `call_anthropic`, `call_google`). 

### Other LLMs.

We recommend implementing your own logic for making LLM queries (similar to `call_openai`, `call_anthropic`, and `call_google`), and then hooking up your code to `call_llm`. 

## Running EconEvals benchmarks at custom difficulty levels

For the EconEvals benchmarks, we provide three stock difficulty levels—Basic, Medium, and Hard. The parameters that specify each difficulty level are given in the environment-specific file. Brief instructions for adding a new difficulty level for each environment:
- Procurement: In `run_procurement_batch.py`, make a new param dict that resembles `basic_params` with your chosen params, and add it to `difficulty_to_params`. (Note that large instances may require a Gurobi license, see the appendix on procurement in the paper for more information.)
- Scheduling: In `run_scheduling_batch.py`, add your difficulty to the dicts `difficulty_to_num_workers` and `difficulty_to_num_blocking_pairs`. (It's also possible to change the preference generation parameters in that file, but this is not something we do to scale difficulty in the paper.)
- Pricing: In `run_pricing_batch.py`, add your difficulty to the dict `DIFFCULTY_TO_NUM_PRODUCTS`. 

# Instructions for paper replication

## Procurement 

### Data collection 

Example running a single experimental run:
```
python3 econ_evals/experiments/procurement/run_procurement_batch.py --model gpt-4o-2024-11-20 --difficulty Basic --seeds 0
```

*Note: On some systems, instance generation may throw an `AssertionError` at the start of the run. Restarting the job fixes the problem. We're working on a fix.*

### Analysis and score computation

Once data is collected, use `compute_scores.ipynb` to calculate the benchmark scores.

## Scheduling 

### Data collection 


Example running a single experimental run (to collect more data, vary `--model`, `--seed`, or `--difficulty` accordingly):
```
python3 econ_evals/experiments/scheduling/run_scheduling_batch.py  --model gpt-4o-2024-11-20 --difficulty Basic --seeds 0
```

### Analysis and score computation


After collecting the data, run `calculate_scheduling_baseline.py`, which will produce a file `<timestamp>__scheduling_baseline_data.txt` that contains empirical estimates for the uniform random baseline (the E[# blocking pairs in $\mu$] denominator in the benchmark score calculation).

Once data is collected, use `compute_scores.ipynb` to calculate the benchmark scores.


## Pricing 

### Data collection 


Example running a single experimental run (to collect more data, vary `--model`, `--seed`, or `--difficulty` accordingly):
```
python3 econ_evals/experiments/pricing/run_pricing_batch.py --model gpt-4o-2024-11-20 --difficulty Basic --seeds 0
```

### Analysis and score computation


After collecting the data, run `precompute_pricing_opt.py` to precompute the optimal prices and profits at each period (the OPT denominator in the benchmark score calculation).

Once data is collected, use `compute_scores.ipynb` to calculate the benchmark scores.


## Efficiency versus Equality 

### Data collection 


Example running a single experimental run (to collect more data, vary `--model`, `--seed`, or `--prompt_type` accordingly):
```
python3 econ_evals/experiments/efficiency_vs_equality/run_efficiency_vs_equality_batch.py --model gpt-4o-2024-11-20 --seeds 0 --prompt_type main
```

### Analysis and score computation


Once data is collected, use `compute_scores.ipynb` to calculate the reliability scores and litmus scores.

## Patience versus Impatience 

### Data collection 


Replicate results from paper (specify the models to run in the script):
```
python3 econ_evals/experiments/patience_vs_impatience/run_patience_vs_impatience.py --time_horizon one_month
python3 econ_evals/experiments/patience_vs_impatience/run_patience_vs_impatience.py --time_horizon six_months
python3 econ_evals/experiments/patience_vs_impatience/run_patience_vs_impatience.py --time_horizon one_year
python3 econ_evals/experiments/patience_vs_impatience/run_patience_vs_impatience.py --time_horizon five_years
```

### Analysis and score computation


Once data is collected, use `compute_scores.ipynb` to calculate the reliability scores and litmus scores.


## Collusiveness versus Competitiveness

### Data collection 


Example running a single experimental run (to collect more data, vary `--model`, `--seed`, or `--prompt_type` accordingly):
```
python3 econ_evals/experiments/collusiveness_vs_competitiveness/run_collusiveness_vs_competitiveness_batch.py --model gpt-4o-2024-11-20 --seeds 0 --prompt_type collusion_v1
```

### Analysis and score computation


Once data is collected, use `compute_scores.ipynb` to calculate the reliability scores and litmus scores.


# Citation

We welcome follow-up work that uses, is derived from, or is inspired by, EconEvals benchmarks and litmus tests. We ask that you provide credit and cite us (BibTeX entry below).

```
@article{fish2025econevals,
  title={EconEvals: Benchmarks and Litmus Tests for Economic Decision-Making by LLM Agents},
  author={Fish, Sara and Shephard, Julia and Li, Minkai and Shorrer, Ran and Gonczarowski, Yannai},
  journal={arXiv preprint arXiv:2503.18825},
  year={2025}
}
```
