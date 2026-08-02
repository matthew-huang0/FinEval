"""
test_cost_simulation.py
Simulates cost calculation to demonstrate how the cost metric works
when pointed at a paid API like OpenAI or Anthropic.
Shows the manager exactly what the cost field will show in production.
"""

def calculate_cost(
    tokens_in: int,
    tokens_out: int,
    cost_per_million_input: float,
    cost_per_million_output: float
) -> float:
    """
    Calculate the cost of one API call in cents.
    Same formula used in runner.py.
    """
    input_cost = (tokens_in / 1_000_000) * cost_per_million_input
    output_cost = (tokens_out / 1_000_000) * cost_per_million_output
    return input_cost + output_cost


# simulate a typical FinQA call
# a FinQA prompt is long — about 500 tokens in, 10 tokens out
typical_tokens_in = 500
typical_tokens_out = 10

print("=" * 60)
print("Cost Simulation — How the cost metric works with paid APIs")
print("=" * 60)
print()
print(f"Typical FinQA call:")
print(f"  Input tokens:  {typical_tokens_in}")
print(f"  Output tokens: {typical_tokens_out}")
print()

# OpenAI GPT-4o pricing 
# $5 per million input tokens = 500 cents per million
# $15 per million output tokens = 1500 cents per million
gpt4o_input_cost = 500.0
gpt4o_output_cost = 1500.0
gpt4o_cost = calculate_cost(
    typical_tokens_in,
    typical_tokens_out,
    gpt4o_input_cost,
    gpt4o_output_cost
)

# Anthropic Claude Sonnet pricing 
# $3 per million input tokens = 300 cents per million
# $15 per million output tokens = 1500 cents per million
claude_input_cost = 300.0
claude_output_cost = 1500.0
claude_cost = calculate_cost(
    typical_tokens_in,
    typical_tokens_out,
    claude_input_cost,
    claude_output_cost
)

# NVIDIA free tier
nvidia_cost = calculate_cost(typical_tokens_in, typical_tokens_out, 0.0, 0.0)

print("Cost per API call:")
print(f"  NVIDIA free tier:       {nvidia_cost:.6f} cents  (free)")
print(f"  OpenAI GPT-4o:          {gpt4o_cost:.6f} cents")
print(f"  Anthropic Claude:       {claude_cost:.6f} cents")
print()

# simulate a full 20 item benchmark run
n_items = 20
print(f"Cost for full 20 item benchmark run:")
print(f"  NVIDIA free tier:       {nvidia_cost * n_items:.4f} cents  (free)")
print(f"  OpenAI GPT-4o:          {gpt4o_cost * n_items:.4f} cents")
print(f"  Anthropic Claude:       {claude_cost * n_items:.4f} cents")
print()

# simulate a full dataset run (6251 items for FinQA)
n_full = 6251
print(f"Cost for full FinQA dataset ({n_full} items):")
print(f"  NVIDIA free tier:       {nvidia_cost * n_full:.4f} cents  (free)")
print(f"  OpenAI GPT-4o:          {gpt4o_cost * n_full:.2f} cents  (${gpt4o_cost * n_full / 100:.2f})")
print(f"  Anthropic Claude:       {claude_cost * n_full:.2f} cents  (${claude_cost * n_full / 100:.2f})")
print()

print("=" * 60)
print("How to use with a real paid API:")
print("=" * 60)
print()
print("python3 -m fineval.cli \\")
print("  --model gpt-4o \\")
print("  --provider openai \\")
print("  --base-url https://api.openai.com/v1 \\")
print("  --cost-per-million-input-tokens 500 \\")
print("  --cost-per-million-output-tokens 1500 \\")
print("  --datasets finqa \\")
print("  --max-items 20 \\")
print("  --out results/gpt4o-benchmark.json")
print()
print("The cost field in the results summary will then show")
print("the real dollar cost of every API call automatically.")