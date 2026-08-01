# FinEval Leaderboard
Generated: 2026-07-31 17:58

| Model | Dataset | Task | Items | Accuracy | F1/Mean Score | P50 Latency | Cost (cents) |
|-------|---------|------|-------|----------|---------------|-------------|--------------|
| gemma-4-31b | convfinqa | multiturn_qa | 500 | 81.80% | 0.8207 | 433ms | 0.0000 |
| gpt-oss-120b | convfinqa | multiturn_qa | 500 | 80.40% | 0.8068 | 472ms | 0.0000 |
| llama3.2:1b | convfinqa | multiturn_qa | 500 | 19.60% | 0.2005 | 259ms | 0.0000 |
| gemma-4-31b | custom | numerical_qa | 91 | 98.90% | 0.9890 | 489ms | 0.0000 |
| gpt-oss-120b | custom | numerical_qa | 91 | 90.11% | 0.9011 | 520ms | 0.0000 |
| llama3.2:1b | custom | numerical_qa | 91 | 45.05% | 0.5249 | 237ms | 0.0000 |
| gemma-4-31b | finpb | classification | 500 | 99.20% | 0.9920 | 374ms | 0.0000 |
| gpt-oss-120b | finpb | classification | 500 | 98.40% | 0.9840 | 270ms | 0.0000 |
| TF-IDF + XGBoost | finpb | classification | 453 | 90.95% | 0.9058 | 0ms | 0.0000 |
| TF-IDF + Logistic Regression | finpb | classification | 453 | 87.86% | 0.8731 | 0ms | 0.0000 |
| llama3.2:1b | finpb | classification | 500 | 82.60% | 0.8260 | 201ms | 0.0000 |
| gpt-oss-120b | finqa | numerical_qa | 500 | 69.80% | 0.7129 | 463ms | 0.0000 |
| gemma-4-31b | finqa | numerical_qa | 500 | 63.20% | 0.6387 | 410ms | 0.0000 |
| llama3.2:1b | finqa | numerical_qa | 500 | 2.80% | 0.0337 | 914ms | 0.0000 |