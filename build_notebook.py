"""
build_notebook.py
Generates the FinEval exploratory analysis notebook (analysis_notebook.ipynb).
Run once: python3 build_notebook.py
Then open analysis_notebook.ipynb in VS Code or Jupyter and run the cells.
"""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

def md(text):
    cells.append(nbf.v4.new_markdown_cell(text))

def code(text):
    cells.append(nbf.v4.new_code_cell(text))


# ---- Title ----
md("""# FinEval — Exploratory Analysis

This notebook explores the results of benchmarking three language models and two
classical machine learning baselines on financial reasoning tasks.

The three language models span a wide size range across three companies:
- **gpt-oss-120b** (OpenAI, 120B) — large, a reasoning model
- **gemma-4-31b** (Google, 31B) — medium
- **llama3.2:1b** (Meta, 1B) — small, run locally

They are evaluated on three datasets: Financial PhraseBank (sentiment
classification), FinQA (single-shot numerical reasoning), and ConvFinQA
(multi-turn numerical reasoning). Two classical baselines (TF-IDF with Logistic
Regression and with XGBoost) run on Financial PhraseBank for comparison.

All language model results use a 500-item sample per dataset.""")


# ---- Setup ----
md("## 1. Setup and Data Loading")
code("""import json
import glob
import os
import pandas as pd

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 120)

def load_results(path):
    with open(path) as f:
        return json.load(f)

# load the three model result files
model_files = {
    'gpt-oss-120b': 'results/large-gptoss120b.json',
    'gemma-4-31b': 'results/medium-gemma31b.json',
    'llama3.2:1b': 'results/small-llama1b.json',
}

all_rows = []
for model, path in model_files.items():
    for r in load_results(path):
        all_rows.append(r)

df = pd.DataFrame(all_rows)
print(f'Loaded {len(df)} total result rows')
df.head()""")


# ---- Accuracy summary ----
md("""## 2. Accuracy by Model and Dataset

The first question is simply how accurate each model is on each task. The table
below shows accuracy for every model and dataset combination.""")
code("""summary = (
    df.groupby(['model_id', 'dataset_name'])
    .agg(items=('is_correct', 'size'), correct=('is_correct', 'sum'))
    .reset_index()
)
summary['accuracy_%'] = (summary['correct'] / summary['items'] * 100).round(1)
summary_pivot = summary.pivot(index='model_id', columns='dataset_name', values='accuracy_%')
summary_pivot""")

md("""**What this shows.** On Financial PhraseBank (classification), all three models
do well, and the medium model actually edges out the large one. On the two
numerical datasets, the picture splits sharply: the large and medium models stay
strong while the small model collapses, scoring near zero on FinQA. This is the
central finding of the project. A 1-billion-parameter model can classify sentiment
but cannot perform financial calculation.""")


# ---- Classical ML comparison ----
md("""## 3. Language Models vs Classical Machine Learning

Financial PhraseBank is the one task where classical machine learning can compete,
because it is a classification problem rather than a reasoning problem. Here we
compare the language models against the two classical baselines on that dataset.""")
code("""baseline_files = {
    'TF-IDF + XGBoost': 'results/baseline-xgboost.json',
    'TF-IDF + Logistic Regression': 'results/baseline-logreg.json',
}

rows = []
# LLM accuracy on finpb
for model, path in model_files.items():
    data = load_results(path)
    finpb = [r for r in data if r['dataset_name'] == 'finpb']
    acc = sum(r['is_correct'] for r in finpb) / len(finpb) * 100
    rows.append({'model': model, 'type': 'LLM', 'accuracy_%': round(acc, 1)})

# baseline accuracy
for model, path in baseline_files.items():
    data = load_results(path)
    acc = sum(r['is_correct'] for r in data) / len(data) * 100
    rows.append({'model': model, 'type': 'Classical ML', 'accuracy_%': round(acc, 1)})

finpb_compare = pd.DataFrame(rows).sort_values('accuracy_%', ascending=False)
finpb_compare""")

md("""**What this shows.** XGBoost, a traditional algorithm with no language
understanding, beats the small language model and lands within about nine points
of the frontier models, while running instantly, offline, and at zero cost. For
sentiment classification specifically, a classical approach is genuinely
competitive. This is the kind of result that matters to a team deciding whether a
task actually needs a large language model.""")


# ---- Cost analysis ----
md("""## 4. Cost and Token Efficiency

The models were run for free, but in production they would cost money. Using
reference market prices, we estimate what each model would cost and, more
importantly, how efficiently it uses tokens. Token counts were captured during the
benchmark run.""")
code("""# reference market prices per million tokens (input, output) in USD
PRICING = {
    'gpt-oss-120b': (0.35, 0.75),
    'gemma-4-31b': (0.08, 0.16),
    'llama3.2:1b': (0.02, 0.04),
}

rows = []
for model, path in model_files.items():
    data = load_results(path)
    tin = sum(r.get('tokens_in', 0) for r in data)
    tout = sum(r.get('tokens_out', 0) for r in data)
    pin, pout = PRICING[model]
    cost = tin / 1e6 * pin + tout / 1e6 * pout
    correct = sum(r['is_correct'] for r in data)
    acc = correct / len(data) * 100
    rows.append({
        'model': model,
        'avg_accuracy_%': round(acc, 1),
        'tokens_in': tin,
        'tokens_out': tout,
        'total_cost_$': round(cost, 4),
        'cost_per_correct_$': round(cost / correct, 6),
    })

cost_df = pd.DataFrame(rows)
cost_df""")

md("""**What this shows.** The large model costs several times more than the medium
model for almost no accuracy gain. The reason is visible in the token columns: the
large model is a reasoning model and generates far more output tokens than the
medium model, because it works through calculations verbosely. That verbosity has a
direct dollar cost. For a team choosing a model, the medium model delivers nearly
the same accuracy at a fraction of the cost.""")


# ---- Failure mode analysis ----
md("""## 5. Failure Mode Analysis

Accuracy numbers tell us how often each model is right, but not why it is wrong.
This section digs into the actual errors to understand how each model fails.""")

md("### 5a. What does the small model get wrong on classification?")
code("""small = load_results('results/small-llama1b.json')
small_finpb_wrong = [r for r in small if r['dataset_name'] == 'finpb' and not r['is_correct']]
print(f'Small model got {len(small_finpb_wrong)} of 500 finpb items wrong')
print()
for r in small_finpb_wrong[:8]:
    print(f\"expected: {r['expected']:<10} got: {r['ai_answer'][:30]}\")""")

md("""On classification the small model's errors are ordinary misclassifications:
it confuses positive, negative, and neutral. It understands the task and simply
gets some judgments wrong, which is why it still scores in the 80s.""")

md("### 5b. What does the small model get wrong on numerical reasoning?")
code("""small_finqa_wrong = [r for r in small if r['dataset_name'] == 'finqa' and not r['is_correct']]
print(f'Small model got {len(small_finqa_wrong)} of 500 finqa items wrong')
print()
for r in small_finqa_wrong[:8]:
    print(f\"expected: {r['expected']:<14} got: {r['ai_answer'][:45]}\")""")

md("""Here the failure is different in kind. The small model does not misjudge, it
is simply unable to perform the calculation. It often returns raw numbers copied
from the context, or malformed output, rather than a computed answer. This is why
its numerical accuracy collapses to near zero rather than merely declining. It is a
capability limit, not a tuning problem.""")

md("### 5c. Which questions did every model get wrong?")
code("""# find finqa items that all three models missed
def wrong_ids(path, dataset):
    data = load_results(path)
    return {r['item_id'] for r in data if r['dataset_name'] == dataset and not r['is_correct']}

large_wrong = wrong_ids('results/large-gptoss120b.json', 'finqa')
medium_wrong = wrong_ids('results/medium-gemma31b.json', 'finqa')
small_wrong = wrong_ids('results/small-llama1b.json', 'finqa')

all_missed = large_wrong & medium_wrong & small_wrong
print(f'FinQA items missed by ALL THREE models: {len(all_missed)}')
print(f'Items missed by the large model but not the medium: {len(large_wrong - medium_wrong)}')
print(f'Items missed by the medium model but not the large: {len(medium_wrong - large_wrong)}')""")

md("""The items every model missed are the genuinely hard questions in the dataset,
where even the frontier model fails. The items one model gets and another misses
show where the models differ in capability. This kind of breakdown is more useful
than a single accuracy number, because it separates task difficulty from model
capability.""")


# ---- Conclusion ----
md("""## 6. Summary of Findings

1. **Model size matters enormously for numerical reasoning but little for
classification.** All three models classify sentiment well, but only the larger two
can do financial calculation. The small model collapses to near zero on FinQA.

2. **Classical machine learning is competitive on classification.** XGBoost beats
the small language model and nearly matches the frontier models on Financial
PhraseBank, at zero cost and instant speed.

3. **The medium model is the practical winner.** It matches or beats the large
model on two of three datasets while costing several times less to run.

4. **Reasoning models carry a verbosity cost.** The large model's habit of working
through calculations in long detail generates many more output tokens, which
translates directly into higher cost for almost no accuracy benefit on these tasks.""")


nb['cells'] = cells

with open('analysis_notebook.ipynb', 'w') as f:
    nbf.write(nb, f)

print('Notebook written to analysis_notebook.ipynb')
print('Open it in VS Code or run: jupyter notebook analysis_notebook.ipynb')