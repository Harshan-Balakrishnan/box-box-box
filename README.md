# Box Box Box — F1 Pit Strategy Optimization Challenge

This repository contains my solution work for the **Box Box Box** challenge:

> Reverse-engineer a hidden race simulation algorithm from 30,000 historical Formula 1 races and predict final finishing positions from pit strategy, tire choices, and track conditions.

## Project goal

The target problem is to build a simulator that:

- reads race input from `stdin` as JSON
- simulates all 20 drivers
- models lap-by-lap race behavior
- applies tire performance and degradation
- handles pit stop timing and penalties
- outputs the final finishing order as JSON to `stdout`

## Current best confirmed result

My best confirmed **real simulator baseline** currently scores:

- **14/100 exact matches** on the visible official scorer (`tools/score_official_100.py`)

This is the best stable non-cheating version I found during reverse-engineering.

## What is considered the main solution

The current main solution is:

- `solution/race_simulator.py`

This file is the best stable simulator baseline I found and is the version pointed to by:

- `solution/run_command.txt`

I also keep a protected copy here:

- `solution/race_simulator_best_14of100.py`

## Solution approach

I explored several approaches while working on the challenge:

### 1. Hand-built lap-by-lap simulator
This is the current main baseline.

It models:
- base lap time
- tire compound performance differences
- fresh tire bonuses
- tire degradation over time
- temperature interaction
- pit lane time penalties
- basic tie-breaking using grid position when predicted times are equal

This is the most challenge-faithful version in the repository.

### 2. Learned ranking model
I also explored a machine-learning approach using XGBoost ranking trained on historical races.

Relevant files:
- `solution/train_model.py`
- `solution/trained_model/`

This approach learned patterns from historical race strategies and final orders, but in my visible evaluation it did not beat the best simulator baseline consistently enough to replace it as the main solution.

### 3. Hybrid experiments
I also tested hybrid approaches that combined:
- simulator output
- learned ranker output

Relevant file:
- `solution/race_simulator_hybrid.py`

These experiments were useful for research, but they were not selected as the final primary solution because they did not outperform the best stable baseline on the visible official scorer.

## Repository layout

- `solution/race_simulator.py` — current best stable simulator baseline
- `solution/race_simulator_best_14of100.py` — backup copy of the best baseline
- `solution/run_command.txt` — command used by the test runner
- `solution/train_model.py` — XGBoost ranker training script
- `solution/race_simulator_hybrid.py` — hybrid experiment
- `tools/score_official_100.py` — visible official scorer
- `tools/debug_test001.py` — debugging helper for `TEST_001`
- `tools/tune_historical_params.py` — parameter search script
- `tools/train_driver_offsets.py` — driver offset experiment
- `data/historical_races/` — historical training races
- `data/test_cases/` — provided visible tests

## How to run the current best solution

### 1. Install dependencies

```bash
pip install numpy xgboost