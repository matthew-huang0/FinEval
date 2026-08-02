# FinEval

FinEval is a benchmarking tool that evaluates large language models against
classical machine learning baselines on financial reasoning tasks. It runs
any OpenAI-compatible model through a common harness, scores its answers with
a shared normalizer, and produces a combined leaderboard, cost analysis, and
charts. Swapping providers or models is a configuration change, not a code
change, so the same tool works with Cerebras, Ollama, NVIDIA, OpenAI, or any
other OpenAI-compatible endpoint.

## What it does

- Benchmarks language models on four financial datasets (sentiment
  classification and numerical reasoning)
- Compares them against classical ML baselines (TF-IDF with Logistic
  Regression and XGBoost)
- Scores numerical and classification answers with a shared normalizer that
  handles formatting differences fairly across all models
- Auto-discovers every result file and builds a combined leaderboard
- Estimates per-model cost from captured token counts at reference market
  prices
- Generates accuracy, latency, and cost charts

## Headline findings

Full results, methodology, and charts are in `docs/DESIGN.md` and the
generated leaderboard. The main findings from benchmarking three models
(1B to 120B parameters) across four datasets were:

- The medium model (Gemma 4 31B) matched or beat the large model
  (gpt-oss 120B) on three of the four datasets, while costing roughly 6.7
  times less to run at reference market prices. For these tasks it is the
  most cost-effective choice.
- The small model (Llama 3.2 1B) classified sentiment acceptably but
  collapsed on numerical reasoning, scoring near zero on FinQA. This marks
  a clear capability floor: a 1B model can classify but cannot calculate.
- On sentiment classification, a classical TF-IDF and XGBoost baseline beat
  the small language model and came within about nine points of the frontier
  models, while running instantly, offline, and at zero cost.
- The large model's cost came mainly from verbosity: as a reasoning model it
  generated roughly 35 times more output tokens than the medium model for
  almost the same accuracy, and output tokens are the main driver of cost.

## Installation

Requires Python 3.10 or newer.

```bash
git clone https://github.com/matthew-huang0/FinEval.git
cd FinEval
python3 -m venv venv
source venv/bin/activate        # on Windows: venv\Scripts\activate
pip install -e .
```

Installing with `-e` (editable mode) means changes to the code take effect
immediately without reinstalling.

## Datasets and setup

FinEval uses four datasets. Three require no setup; one needs a separate
download.

| Dataset | Task | Setup |
|---------|------|-------|
| FinQA | Numerical reasoning | Auto-downloaded from Hugging Face |
| ConvFinQA | Multi-turn numerical reasoning | Auto-downloaded from Hugging Face |
| Custom SEC 10-K | Numerical reasoning | Included in the repository |
| Financial PhraseBank | Sentiment classification | Requires a separate download (see below) |

### Financial PhraseBank download

Financial PhraseBank is licensed for non-commercial research use only and
cannot be redistributed, so it is not included in this repository. To use it:

1. Download the `FinancialPhraseBank-v1.0` release and locate the
   `Sentences_AllAgree.txt` file.
2. Point FinEval at it with the `FINPB_PATH` environment variable:

```bash
export FINPB_PATH=/path/to/FinancialPhraseBank-v1.0/Sentences_AllAgree.txt
```

If `FINPB_PATH` is not set, FinEval looks in
`~/Downloads/FinancialPhraseBank-v1.0/Sentences_AllAgree.txt` by default. If
the file is not found, FinEval prints a message explaining how to obtain it.

The other three datasets need no setup, so if you skip this step you can still
run FinEval on FinQA, ConvFinQA, and the custom dataset.

### How the Hugging Face datasets load

FinQA and ConvFinQA are downloaded automatically from Hugging Face the first
time you run them. You do not need to download anything manually. The first
run fetches the data and caches it locally, so later runs are fast.

You do not need a Hugging Face account for this. If you run many downloads you
may see a message suggesting you set an `HF_TOKEN` for higher rate limits;
this is optional and not required for normal use. To set one, create a free
token on huggingface.co and export it:

```bash
export HF_TOKEN=your_token_here
```

The custom SEC 10-K dataset is included in the repository, so it also needs no
download. Only Financial PhraseBank requires the separate download described
above, because its license does not allow redistribution.


## Setting up a provider

FinEval does not host any models itself. It sends requests to a provider you
choose. Any provider with an OpenAI-compatible API works. This section walks
through the three used during development: Cerebras (hosted, free tier),
Ollama (local, free), and NVIDIA (hosted). You only need one to get started.

For every provider, three things must match up: the `--provider` name, the
`--base-url` for that provider's API, and the API key in the environment
variable FinEval expects. The runner reads the key from either
`CEREBRAS_API_KEY` or `NVIDIA_API_KEY`; local providers such as Ollama need
no key.

### Cerebras (hosted, has a free tier)

Cerebras runs models on custom hardware and offers a free tier with a daily
token allowance, which is enough to run these benchmarks.

1. Go to cloud.cerebras.ai and sign up with an email address or GitHub
   account. No credit card is required for the free tier.
2. In the console, open the API Keys section in the left navigation.
3. Click Generate API Key, give it a name, and copy the key.
4. Set it in your terminal:

```bash
export CEREBRAS_API_KEY=your_key_here
```

The `export` command makes the key available to FinEval for the current
terminal session. If you open a new terminal, set it again, or add the line
to your shell profile (such as `~/.zshrc`) to make it permanent.

Then run a benchmark, passing the Cerebras base URL:

```bash
python3 -m fineval.cli --model gpt-oss-120b --provider cerebras \
  --base-url https://api.cerebras.ai/v1 --datasets finqa --max-items 20 \
  --out results/cerebras-run.json
```

The free tier has a daily token limit. If you hit it, requests return a 429
error and the runner retries with a wait. If they keep failing, you have
reached the daily cap, which resets the next day.

### Ollama (local, free, no key)

Ollama runs models entirely on your own machine. There is no API key, no cost,
and no internet requirement once a model is downloaded. This is how the small
model was run during development.

1. Install Ollama from ollama.com. On macOS you can also use Homebrew with
   `brew install ollama`; on Linux, `curl -fsSL https://ollama.com/install.sh | sh`.
2. Download a model. For example, the small model used here:

```bash
ollama pull llama3.2:1b
```

3. Make sure the Ollama server is running. On macOS and Windows it starts
   automatically when the app is open. You can also start it manually in a
   separate terminal:

```bash
ollama serve
```

Ollama serves an OpenAI-compatible API on `http://localhost:11434/v1`. Point
FinEval at it:

```bash
python3 -m fineval.cli --model llama3.2:1b --provider ollama \
  --base-url http://localhost:11434/v1 --datasets finqa --max-items 20 \
  --out results/ollama-run.json
```

If you see a connection error, the server is not running; start it with
`ollama serve` and try again.

### NVIDIA (hosted)

NVIDIA hosts many models through an OpenAI-compatible API.

1. Get an API key from NVIDIA's build platform (build.nvidia.com).
2. Set it in your terminal:

```bash
export NVIDIA_API_KEY=your_key_here
```

3. Run, passing NVIDIA's base URL and a model it hosts:

```bash
python3 -m fineval.cli --model meta/llama-3.3-70b-instruct --provider nvidia \
  --base-url https://integrate.api.nvidia.com/v1 --datasets finqa --max-items 20 \
  --out results/nvidia-run.json
```

### Any other provider

Because FinEval uses the OpenAI SDK, any OpenAI-compatible endpoint works. Set
the provider's key (as either `CEREBRAS_API_KEY` or `NVIDIA_API_KEY`, whichever
your setup reads), pass its `--base-url`, and give a `--model` it hosts. No
code changes are required.


## Quickstart

Run a model on all four datasets, 20 items each, and save the results:

```bash
export CEREBRAS_API_KEY=your_key_here
python3 -m fineval.cli \
  --model gpt-oss-120b \
  --provider cerebras \
  --base-url https://api.cerebras.ai/v1 \
  --datasets finpb,finqa,convfinqa,custom \
  --max-items 20 \
  --out results/my-run.json
```

Then build the leaderboard from every result file:

```bash
python3 analysis/leaderboard.py
```

## Running benchmarks

A benchmark run sends every item in the chosen datasets to the model, scores
each answer, and writes the results to a JSON file. The general shape is:

```bash
python3 -m fineval.cli \
  --model <model id> \
  --provider <provider> \
  --base-url <provider base url> \
  --datasets <comma separated datasets> \
  --max-items <number> \
  --out results/<name>.json
```

A few practical notes:

- `--max-items` limits how many items per dataset are run. Start with a small
  number such as 20 to confirm everything works, then raise it. The full
  benchmark results were produced at 500 items per dataset.
- `--datasets` accepts any combination of `finpb`, `finqa`, `convfinqa`, and
  `custom`. To run just one, pass a single name; to run several, separate them
  with commas and no spaces.
- Each run writes one JSON file. Give each model its own output file, because
  the leaderboard reads every file in `results/` and builds the comparison
  from all of them.
- The run prints a live summary as it goes, including accuracy, F1, latency,
  and cost, and a final summary when each dataset finishes.

To benchmark several models, run the command once per model with a different
`--model`, `--provider`, and `--out`. For example, run the large model on
Cerebras, the small model on Ollama, then build the leaderboard to compare
them.


## CLI reference

```
python3 -m fineval.cli [options]
```

| Option | Description |
|--------|-------------|
| `--model` | Model ID to benchmark (e.g. `gpt-oss-120b`) |
| `--provider` | Provider name: `cerebras`, `ollama`, `openai`, `nvidia` |
| `--base-url` | API base URL for the provider |
| `--datasets` | Comma-separated list: `finpb`, `finqa`, `convfinqa`, `custom` |
| `--max-items` | Maximum number of items per dataset |
| `--out` | Path to save the results JSON file |
| `--cost-per-million-input-tokens` | Input token cost in cents. Auto-detected from provider and model if not set. |
| `--cost-per-million-output-tokens` | Output token cost in cents. Auto-detected if not set. |

## Classical ML baselines

The baselines train TF-IDF with Logistic Regression and XGBoost on Financial
PhraseBank and evaluate them on its test split. They require Financial
PhraseBank to be available (see the setup section above).

```bash
python3 fineval/baselines/finpb_baseline.py
```

This saves baseline result files to `results/`, which the leaderboard then
picks up automatically.

## Leaderboard and charts

Build the leaderboard (Markdown and CSV) from every result file in `results/`:

```bash
python3 analysis/leaderboard.py
```

Generate the accuracy, latency, and comparison charts:

```bash
python3 analysis/analysis.py
```

Estimate per-model cost from captured token counts, and generate the cost
charts:

```bash
python3 analysis/cost_analysis.py
python3 analysis/cost_charts.py
```

## Project structure

```
FinEval/
├── fineval/                 the core package
│   ├── cli.py               command-line entry point
│   ├── runner.py            the evaluation loop
│   ├── data_structures.py   shared data models
│   ├── datasets/            dataset loaders (finpb, finqa, convfinqa, custom)
│   ├── scorers/             normalizer and scorers
│   └── baselines/           classical ML baselines
├── analysis/                leaderboard, charts, and cost analysis scripts
├── custom_data/             scripts that build the custom SEC 10-K dataset
├── tests/                   test suite
├── notebooks/               exploratory analysis notebook
├── docs/                    design document and methodology
└── results/                 benchmark results, leaderboard, and charts
```

## Extending FinEval

### Adding a model

No code changes are needed. Pass any OpenAI-compatible model through the CLI
with the matching `--provider`, `--base-url`, and API key.

### Regenerating the custom dataset

The custom SEC 10-K dataset is built from verified figures in
`custom_data/custom_dataset_source.py`. To regenerate it:

```bash
python3 custom_data/generate_custom_dataset.py
```

This writes the dataset to `fineval/datasets/custom/custom_dataset.json`. The
methodology, including how the figures were verified against SEC filings, is
documented in `docs/custom-dataset-methodology.md`.

## License

FinEval is released under the MIT License. Note that Financial PhraseBank,
which FinEval can use but does not include, is licensed separately for
non-commercial research use only.
