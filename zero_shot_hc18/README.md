# Zero Shot HC18

This repository contains code for evaluating zero-shot classification on HC18 data.

## Running Test Evaluations

**Note:** Ensure that you alter the paths in the scripts according to your local setup before running the evaluation.

The following scripts must be run sequentially:

1. First, run the following command to generate `FetalCLIP_dot_prods_map.pt`, which contains the dot products between image embeddings and text prompts across gestational age:
```bash
python test_ga_dot_prods_map.py
```

2. Then, run the following command to postprocess the results:
```bash
python test_ga_postprocess_dot_prods.py
```
