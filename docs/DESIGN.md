# FinEval Design Document

This document explains the key decisions made while building FinEval,
what I chose, why I chose it, and what I discovered along the way.

**How to read this document.** This is a chronological design log. Each
week records the decisions made at that point in time, along with the
reasoning behind them. Where a later week supersedes an earlier decision,
the later week governs. For example, the model set described in Week 2
was replaced in Week 6 after a provider migration, and the scoring
tolerance described in Week 3 was refined in Week 6. Reading the document
in order shows how the design evolved. Reading only the most recent week
shows the current state of the system.

---

# Week 2 — Research & Decisions

## The Datasets

FinEval tests AI models against three real finance datasets. Each one
tests a different skill.

### FinQA
FinQA gives an AI model a real financial report, the kind a Wall Street
analyst would read, and asks it a math question. For example: "Revenue
grew from $4.2 billion to $4.8 billion. What was the growth rate?" The
AI has to find the right numbers buried in a long document and calculate
the correct answer.

Scoring works by checking whether the AI's final number matches the
correct answer. We only check the final number, not how the AI got there.

The best AI model ever published on this dataset scores 61%. A human
expert scores 91%. That 30 point gap shows this is genuinely difficult
even for the most advanced AI.

How we prompt the AI: we ask for a single number only, with no units, commas, or explanation. This keeps the answer unambiguous for the scorer. That way our scoring system can reliably find the answer without getting confused by the reasoning.

### ConvFinQA
ConvFinQA is like FinQA but in conversation form. Instead of one
question, the AI is asked a series of follow-up questions about the same
financial report, where each question builds on the previous answer.

For example: Q1 asks for the 2009 cash flow value. Q2 asks for the 2008
value. Q3 asks for the difference. Q4 asks for the percentage change. The
AI has to remember its earlier answers to get the later ones right.

Scoring works the same as FinQA. We check the final number for each
turn.

The best published AI model scores 69%. Humans score 89%.

How we prompt the AI: ConvFinQA ships each turn with a query field that already contains the context and the prior conversation, so we pass that through directly and ask for a single number.This gives the AI the context it needs to answer follow-up questions correctly.

### Financial PhraseBank
Financial PhraseBank gives the AI one finance news sentence and asks it
to classify the sentiment as positive, neutral, or negative from an
investor's perspective.

For example: "The company moved from a loss to breaking even" sounds
neutral in everyday language, but is positive from an investor's view
because the company stopped losing money. That finance-specific context
is what makes this task tricky.

We use the highest quality version of this dataset, only sentences where
all 16 human annotators agreed on the label. This gives us the cleanest
possible ground truth to score against.

License note: this dataset is free for research use only and cannot be
used commercially.

How we prompt the AI: we ask for a single word answer only, positive,
neutral, or negative. This makes scoring simple and unambiguous.

---

## The AI Models

We benchmark three AI models of different sizes. The goal is to find out
whether paying for a larger, more expensive model actually gets you
meaningfully better results on finance tasks.

**Large model: Meta's Llama 3.3 (70 billion parameters)**
This is the most capable model we test. Think of "parameters" as the
number of connections in the AI's brain. More connections generally
means smarter but slower and more expensive. This model is the
performance ceiling everything else gets compared against.

**Medium model: Mistral's Mixtral**
Instead of one single large model, Mixtral is actually 8 smaller models
working together. Each question gets routed to whichever sub-model is
best suited for it. This represents a fundamentally different approach
from Llama, which makes the comparison more interesting.

**Small model: Meta's Llama 3.2 (1 billion parameters)**
This model is 70 times smaller than the large model. It is much faster
and much cheaper to run, but less capable. It answers the question: what
do you lose by going with the cheapest option?

All three models are available for free via NVIDIA's API.

Note: this model set was replaced in Week 6. NVIDIA's API became unusable, and the models below are no longer the ones benchmarked. See the Week 6 model set section for the current three models.

---

## Open Questions Heading into Week 3
- How do we handle "$1.5M" vs "1500000" in numerical scoring?
- Should we use zero-shot or few-shot prompts?
- How do we store and pass conversation history for ConvFinQA?

---

# Week 3: Dataset Loaders & Numerical Scorer

## Scoring Decisions

### How we score numerical answers
When an AI answers a math question, it might write the same number in
many different ways. For example, one and a half million dollars could
appear as "$1.5 million", "1,500,000", "1.5e6", or "1500000.0", all
meaning exactly the same thing. Our scoring system normalizes all of
these into the same number before comparing.

Specifically it handles:
- Currency symbols like $, £, and € are removed before comparing
- Commas used as thousand separators are removed before comparing
- Words like "million" and "billion" are converted to the actual number
- Scientific notation like 1.5e6 is converted automatically

We also allow a small tolerance of 0.01, meaning an answer that is
slightly off due to rounding still counts as correct. Note: this flat tolerance was refined in Week 6. Large numbers now use a relative tolerance, and the percentage check uses a tolerance scaled to the size of the expected value. See the Week 6 normalizer section.

### Prompting approach
We use zero-shot prompts, meaning we ask the AI the question directly
without showing it any examples first. This ensures the comparison
between models is fair, since pre-written examples could advantage one
model over another.

## What We Discovered While Building

### ConvFinQA data structure
When I went to build the ConvFinQA loader, the field I expected to
contain the question was named differently than anticipated. The dataset
uses a field called "query" which contains the full prompt including
context and conversation history combined. I discovered this by
inspecting the raw data and fixed the loader accordingly.

### Yes/No answers in FinQA
I discovered that some FinQA questions have yes or no answers rather
than numbers. Our numerical scorer cannot handle these. It correctly
returns a score of zero for them. I will build a separate handler for
yes/no questions in Week 4.

## Open Questions Heading into Week 4
- How do we score yes/no answers in FinQA?
- How do we build the ClassificationScorer for Financial PhraseBank?
- How do we structure the command line tool to run full benchmarks?

# Week 4 — Runner, Scorers, CLI, and First Benchmark

## What I Built
This week I built four new pieces of code that together form the
complete evaluation pipeline for FinEval.

The ClassificationScorer checks whether the AI correctly labeled a
finance news sentence as positive, neutral, or negative. It handles
cases where the model adds extra words like "The sentiment is
positive" instead of just saying "positive" by checking whether the
correct label appears anywhere in the response.

The Yes/No handler extends the NumericalScorer to correctly handle
FinQA questions that have yes or no answers instead of numbers. This
fixed the issue discovered in Week 3 where those questions
automatically scored zero regardless of what the AI said.

The Runner sends questions to the AI model in a continuous loop
without stopping or exiting between items, collects the answers,
routes each one to the right scorer, and stores the results. For
ConvFinQA it passes the full conversation history with each turn so
the model can genuinely answer follow up questions by referencing
its previous answers. After every run it prints a full summary
including accuracy, F1 score, average latency, and total cost.
This directly addresses the manager's request to run the eval in a
loop without exiting and keep talking back to the model.

The CLI lets anyone run a full benchmark with a single typed command
without touching any code. It accepts the model name, datasets,
item limit, and output file as inputs and runs the entire pipeline
automatically.

## Manager Feedback Addressed This Week

Concern 1 — Run the eval in a loop, not exiting, being able to
keep talking back to the model.
This was addressed by building the Runner with a continuous loop
that processes every item without stopping. For ConvFinQA
specifically the runner passes the full conversation history to the
model with each turn, meaning the model genuinely receives all
prior context and can reference previous answers when answering
follow up questions. The loop does not exit between items or between
datasets.

Concern 2 — The test script should output additional information
such as evaluation metrics and F1 score.
This was addressed by adding a print_metrics function to the runner
that prints a full summary after every dataset finishes. The summary
includes total items evaluated, number correct, accuracy as a
percentage, F1 score, average latency per API call, and total cost.
This output appears automatically after every benchmark run.

## First Benchmark Results (20 items per dataset, llama-3.3-70b)

Financial PhraseBank: 100% accuracy. The large Llama model performs
perfectly on sentiment classification on this sample. All 20 items
were labeled correctly.

FinQA: 10% accuracy. Main failure mode identified: the AI expresses
ratios and percentages as whole numbers instead of decimals. For
example it answered 53 when the correct answer was 0.53232. The AI
is calculating the right value but formatting it as a percentage
instead of a decimal. This is a prompting issue not an intelligence
issue. Will update the prompt in Week 5 to explicitly request
decimal format.

ConvFinQA: 55% accuracy. Same percentage vs decimal issue as FinQA.
Additional failure mode: the AI occasionally drops negative signs,
answering 2620 when the correct answer is -2620.

## Prompt Improvement Planned for Week 5
Add this instruction to numerical prompts: "Express all ratios and
percentages as decimals. For example 14.5% should be written as
0.145 not 14.5 or 14."

## Open Questions for Week 5
- Will the prompt fix significantly improve FinQA accuracy?
- How will the medium and small models compare to the large model?
- Will XGBoost beat the large model on Financial PhraseBank?

Here is the full Week 5 section for DESIGN.md:

```markdown
# Week 5 — Normalization, Baselines, LLM Judge, and CLI Updates

## What I Built This Week

### Standalone Normalization Module (normalizer.py)
Pulled the number normalization logic out of numerical_scorer.py
into its own standalone file. The normalizer now handles percentage
vs decimal conversion, currency symbols, commas, scientific
notation, extra words in model answers, negative numbers, and
yes/no answers. The key design decision is that a bare integer
like 53 is never automatically treated as a percentage. Only a
value explicitly marked with a percent sign like 53% gets converted
to a decimal. This prevents false positives.

### Updated NumericalScorer
Now delegates all normalization to normalizer.py instead of doing
it internally. Behavior is identical from the outside but the
architecture is cleaner and the normalization logic can be improved
independently without touching the scorer.

### LLM Judge Scorer (llm_judge_scorer.py)
Built a scorer that uses a second AI model to grade the first AI
model's answer. Useful for open ended questions where programmatic
scoring falls short. The judge receives the question, the expected
answer, and the predicted answer and returns a score between 0 and
1. Already defined in DataStructures.py by my manager as a planned
component.

### Provider, URL, and Cost Per Token in CLI
The CLI now accepts provider, base URL, cost per million input
tokens, and cost per million output tokens as inputs. The runner
calculates real cost per API call using the token counts returned
by the API and populates the cost field in results. When using
NVIDIA's free tier cost shows as zero. When pointed at a paid API
the cost field will show real numbers.

### Yes/No Prompt Fix
Updated the runner to detect yes/no questions before building the
prompt. Items where the expected answer is yes or no now receive
a prompt that says "Answer with yes or no only" instead of "Answer
with a single number only." This fixed the issue where the model
was saying "Yes but the question asks for a number so: 0."

### Classical ML Baselines (finpb_baseline.py)
Built two classical ML baselines for Financial PhraseBank:
TF-IDF plus Logistic Regression and TF-IDF plus XGBoost. Both
trained on 80% of the full 2,264 item dataset and tested on the
remaining 20%.

Results:
TF-IDF plus Logistic Regression: 87.86% accuracy, F1 0.8731
TF-IDF plus XGBoost: 90.95% accuracy, F1 0.9058

### First Leaderboard (leaderboard.py)
Built the leaderboard generator that combines LLM benchmark results
and classical ML baseline results into one comparison table exported
as both Markdown and CSV.

### Retry Logic Update
The retry wait time between API call attempts was increased from
5 seconds to 15 seconds. This is more appropriate for 504 Gateway
Timeout errors caused by server overload, giving the server more
time to recover before the next attempt.

## Key Finding This Week
XGBoost scored 90.95% accuracy on the full 2,264 item Financial
PhraseBank dataset with no API calls, no internet connection, and
no cost. Llama 3.3 70B scored 100% on a 20 item sample. A fair
comparison requires running the LLM on a larger sample which is
planned for Week 6.

## Fairness Note
The LLM vs baseline comparison is not perfectly fair yet. The LLM
was tested on only 20 items while the baselines were tested on 453
items. The baselines were trained on 1,811 labeled examples while
the LLM is zero-shot. Both are documented as methodology
limitations.

## Manager Feedback Addressed This Week
Provider, URL, and cost per token configuration added to the CLI
as requested. The cost metric now calculates real costs based on
token counts when a paid API is used.

## Yes/No Handler Investigation
A targeted test was run to verify the yes/no handler. Results
confirmed two things. First the scorer is working correctly — it
correctly marked "No" as wrong when the expected answer was "yes."
Second the model genuinely got the question wrong — it answered
"No" when the correct answer was "yes." This is a model accuracy
issue not a code bug. Getting yes/no questions wrong is expected
given FinQA is a hard dataset where the best published model scores
only 61%.

## API Issue Confirmed
A 504 Gateway Timeout error was returned during a targeted test,
confirming the server issue is on NVIDIA's end. The retry logic
in runner.py catches these errors and retries automatically up to
3 times with 15 second waits between attempts. NVIDIA's status
page showed all systems operational, suggesting their monitoring
did not reflect the actual performance degradation. This is
believed to be related to the July 4th holiday weekend.

## Open Questions for Week 6
- How do the medium and small models compare to the large model
  once servers recover?
- Will the normalizer significantly improve FinQA accuracy on
  clean short model answers?
- How does the LLM compare to XGBoost on a larger Financial
  PhraseBank sample for a fair comparison?
- Should the LLM judge scorer be integrated into the main runner
  for generation task types?

# Week 6 Design Decisions

---

## Provider Migration

NVIDIA's free tier was unusable for a second consecutive week due
to server slowdown, with latency reaching over 120 seconds per call
against a normal 2 to 5 seconds. Rather than wait indefinitely, I
migrated to two new providers.

Cerebras serves the large and medium models through an OpenAI
compatible API at roughly 500 milliseconds per call. Ollama runs
the small model entirely on my local machine with no API calls, no
rate limits, and no cost.

The migration required no changes to the core benchmarking code.
The runner, scorers, CLI, and leaderboard all worked with the new
providers by changing only the model name, base URL, and provider
passed in at the command line. This confirms the tool is genuinely
provider agnostic, which was one of the design goals.

The tradeoff is that Cerebras occasionally returns 429 rate limit
errors under load. The retry logic handles these, but they do add
latency to some calls.

---

## Model Set

The original plan used NVIDIA hosted Llama and Mixtral models, but
Cerebras does not host them. The scope requires at least three models
spanning sizes rather than naming specific required models, so
substituting compliant alternatives is allowed.

The final model set is gpt-oss-120b from OpenAI as the large model
at 120 billion parameters, gemma-4-31b from Google as the medium
model at 31 billion parameters, and llama3.2:1b from Meta as the
small model at 1 billion parameters run locally through Ollama.

This spans roughly 1 billion to 120 billion parameters across three
different companies, which produces a more interesting cross company
comparison than three variants of the same model family would have.

A limitation worth noting is that the size tiers are not perfectly
clean. Two of the models use mixture of experts architectures, so
the number of parameters actually active per token differs from the
total parameter count. The comparison is best understood as three
leading models at different scales rather than a strict size tier
experiment.

---

## Reasoning Model Handling

Two of the models are reasoning models, meaning they think before
answering. These models return their final answer in the message
content field and their thinking process in a separate reasoning
field. The runner was updated to read the content field first and
fall back to the reasoning field if content is empty.

The max token limit was also raised from 512 to 1024, because
reasoning models need room to think before producing an answer. At
the original limit the models were being cut off mid thought and
returning nothing usable.

---

## Normalizer Improvements

Four changes were made to the normalizer this week. All of them
apply equally to every model, which preserves the fairness of the
benchmark. A change that helped only one model would invalidate the
comparison.

The implicit percentage detection handles the case where a model
answers 53.2 when the expected answer is 0.53232. Analysis of the
benchmark errors showed this single pattern accounted for 14 of the
17 FinQA errors. The model was calculating correctly but expressing
the result as a percentage without the percent sign. The normalizer
now detects this case and divides by 100 before comparing.

To avoid crediting a coincidental match, the implicit percentage
detection uses a relative tolerance scaled to the size of the
expected value rather than a flat tolerance. The divided value must
be within 2 percent of the expected value, with a minimum floor.
This means a genuine match such as 5 to 0.05 passes, while a
coincidental miss such as 7 to 0.05 correctly fails.

The large number tolerance allows a model that answers 41932 to be
marked correct for an expected answer of 41932.20339. For values
above 1000 the comparison uses a relative tolerance of 0.1 percent,
because a difference of 0.2 out of 41932 is negligible.

The million scale detection handles the case where a model answers
in full units when the expected answer is expressed in millions.
For example a model answering 3800000 for an expected answer of 3.8
is now marked correct.

Finally the soft score was fixed. Previously an answer could be
marked correct while receiving a score of 0.00, because the score
formula did not know about the new tolerance rules. The soft score
now returns 1.0 whenever the answer is marked correct, removing the
contradiction.

The tradeoff of these changes is increased leniency. It is
theoretically possible for a coincidental match to be credited. The
relative tolerance reduces this risk substantially, and the pattern
being corrected accounted for the large majority of real errors, so
the change is catching genuine model behavior rather than inventing
false positives.

---

## Cost Configuration

The CLI now auto detects cost based on the provider and model rather
than requiring the user to pass it in every time. A pricing table
maps each provider and model to its real cost per million tokens.
Free tier models such as those on Cerebras and Ollama return zero.
Paid models such as OpenAI's GPT-4o return their real published
pricing.

An important clarification is that the cost field is a local
calculation, not a report of actual charges. The tool multiplies the
token counts returned by the API by the rate in the pricing table.
It does not connect to any billing system. This is how most
benchmarking tools work, because APIs do not return their own
pricing programmatically.

---

## Auto Discovering Leaderboard

The leaderboard was rewritten to scan the results folder rather than
reading from a hardcoded list of files. It reads the model name from
inside each result file, detects which datasets are present, and
builds the comparison table automatically.

The classical ML baselines were brought into the same system. The
baseline script now saves its results as JSON in the same format as
the LLM results, with the real F1 score stored in a dedicated field
so the leaderboard can display it correctly.

The practical effect is that any future user can benchmark any model,
run the leaderboard, and see their results appear with no code
changes. This directly supports the reusability goal in the scope.

---

## Analysis Charts

Three charts were built. The first shows accuracy by model and
dataset as grouped bars. The second compares LLMs against classical
ML on Financial PhraseBank, with LLMs in blue and classical models
in red. The third is a cost and quality scatter plot showing average
accuracy against latency.

The scatter plot uses median latency rather than mean. The mean was
badly distorted by a small number of slow retry calls, which made
the Cerebras models appear to take 9 to 11 seconds per call when
their typical response time is a few hundred milliseconds. The median
reflects typical performance honestly.

---

## Benchmark Results

All three models were benchmarked on all three datasets at 20 items
per dataset after the normalizer improvements.

On Financial PhraseBank the large and medium models both scored 100
percent, the small model scored 85 percent, XGBoost scored 90.95
percent, and Logistic Regression scored 87.86 percent.

On FinQA the large model scored 80 percent, the medium model scored
70 percent, and the small model scored 10 percent.

On ConvFinQA the large model scored 95 percent, the medium model
scored 95 percent, and the small model scored 25 percent.

---

## Findings

The normalizer improvements raised the large model's FinQA accuracy from 15 percent to 80 percent and its ConvFinQA accuracy from 80 percent to 95 percent. This was not a change to the models but a
correction to how percentage and decimal answers are compared. This
confirms the Week 5 hypothesis that the low FinQA scores were largely
a formatting issue rather than a reasoning failure, though genuine
reasoning errors remain in the answers that are still wrong.

The medium model nearly matches the large model on every dataset
while running faster. Gemma averages 88 percent accuracy at roughly
185 milliseconds median latency, while gpt-oss averages 92 percent
at roughly 513 milliseconds. For a team choosing a model, the medium
option gives up very little accuracy for a substantial gain in speed.

XGBoost at 90.95 percent beats the small language model at 85 percent
on sentiment classification and comes close to the frontier models
at 100 percent, while running instantly at zero cost with no internet
connection. For this task a traditional algorithm is genuinely
competitive.

The small model collapses on numerical reasoning, scoring 10 percent
on FinQA and 25 percent on ConvFinQA while performing acceptably at
85 percent on classification. This defines a clear capability floor.
A 1 billion parameter model is usable for simple classification but
not for financial calculation.