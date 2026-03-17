
"""
OFFLINE EXPERIMENT: Constraint-inversion via ordinal regression.
Uses historical data to recover exact lap time formula parameters.
This script does NOT run at competition time. It only trains parameters offline.

Usage:
  cd box-box-box
  python tools/inverse_fit_experiment.py
"""

import json
import sys
import os
import numpy as np
from scipy.optimize import minimize, differential_evolution
from itertools import product
import time

# ── Load historical races ──────────────────────────────────────────────────────

def load_all_races(n_files=10):
    """Load first n_files * 1000 historical races."""
    races = []
    base = "data/historical_races"
    files = sorted(os.listdir(base))[:n_files]
    for fn in files:
        with open(os.path.join(base, fn)) as f:
            races.extend(json.load(f))
    print(f"Loaded {len(races)} races from {n_files} files.")
    return races

# ── Build constraint pairs ─────────────────────────────────────────────────────

def build_isolated_pairs(races):
    """
    Extract pairs of drivers in same race with identical compound sequences
    but different pit laps. The finishing order directly constrains the
    degradation function.
    
    Returns: list of dicts with keys:
      compound_in, compound_out, pit_a, pit_b, total_laps, track_temp,
      a_beats_b (bool: driver A finishes before driver B)
    """
    pairs = []
    
    for race in races:
        cfg = race['race_config']
        fp = race['finishing_positions']
        rank = {d: i for i, d in enumerate(fp)}
        strats = list(race['strategies'].items())
        
        # 1-stop drivers
        for i in range(len(strats)):
            for j in range(i + 1, len(strats)):
                ak, av = strats[i]
                bk, bv = strats[j]
                
                if (len(av['pit_stops']) != 1 or len(bv['pit_stops']) != 1):
                    continue
                
                if (av['starting_tire'] != bv['starting_tire'] or
                        av['pit_stops'][0]['to_tire'] != bv['pit_stops'][0]['to_tire']):
                    continue
                
                pa = av['pit_stops'][0]['lap']
                pb = bv['pit_stops'][0]['lap']
                
                if pa == pb:
                    continue
                
                da, db = av['driver_id'], bv['driver_id']
                
                pairs.append({
                    'compound_in':  av['starting_tire'],
                    'compound_out': av['pit_stops'][0]['to_tire'],
                    'pit_a':        pa,
                    'pit_b':        pb,
                    'total_laps':   cfg['total_laps'],
                    'track_temp':   cfg['track_temp'],
                    'a_beats_b':    rank[da] < rank[db],
                })
    
    print(f"Built {len(pairs)} isolated comparison pairs.")
    return pairs


# ── Formula families ───────────────────────────────────────────────────────────

def make_deg_table(params, compound, max_age=80):
    """
    Return deg[age] = degradation at tire_age for ages 1..max_age.
    Supports multiple formula families controlled by params dict.
    
    Formula family: PIECEWISE-LINEAR + OPTIONAL QUADRATIC TERM + TEMP INTERACTION
    
    params per compound: [delta, fresh_bonus, break_age, linear1, linear2, quad, temp_coeff]
    
    deg(age, temp_offset) = 
        compound_delta                                  (constant offset)
      + fresh_bonus * I(age==1)                        (first lap bonus)
      + linear1 * min(age, break_age)                  (pre-break linear)
      + linear2 * max(0, age - break_age)              (post-break linear, steeper cliff)
      + quad * (age**2)                                (optional quadratic)
      + temp_coeff * temp_offset * age                 (temperature interaction)
    """
    pass  # see compute_time_delta below

def compute_time_delta_vectorized(pairs_batch, params_flat):
    """
    Compute predicted time_B - time_A for all pairs given flat parameter vector.
    
    params_flat layout (21 values per compound = 63 total):
    SOFT:   [delta, fresh_bonus, break_age, lin1, lin2, quad, temp_coeff]
    MEDIUM: [delta, fresh_bonus, break_age, lin1, lin2, quad, temp_coeff]
    HARD:   [delta, fresh_bonus, break_age, lin1, lin2, quad, temp_coeff]
    """
    compounds = ['SOFT', 'MEDIUM', 'HARD']
    n = 7  # params per compound
    
    p = {}
    for ci, c in enumerate(compounds):
        base = ci * n
        p[c] = {
            'delta':      params_flat[base + 0],
            'fresh_bonus': params_flat[base + 1],
            'break_age':  max(1.0, params_flat[base + 2]),
            'lin1':       params_flat[base + 3],
            'lin2':       params_flat[base + 4],
            'quad':       params_flat[base + 5],
            'temp_coeff': params_flat[base + 6],
        }
    
    def lap_t(c, age, temp_offset):
        cp = p[c]
        age_eff = max(0, age - 1)  # grace of 1 lap
        t = (cp['delta']
             + (cp['fresh_bonus'] if age == 1 else 0.0)
             + cp['lin1'] * min(age_eff, cp['break_age'])
             + cp['lin2'] * max(0.0, age_eff - cp['break_age'])
             + cp['quad'] * (age_eff ** 2)
             + cp['temp_coeff'] * temp_offset * age_eff)
        return t
    
    def total_delta_for_pair(pair):
        """time_B - time_A"""
        ci = pair['compound_in']
        co = pair['compound_out']
        pa = pair['pit_a']
        pb = pair['pit_b']
        N  = pair['total_laps']
        temp_offset = pair['track_temp'] - 30.0
        
        if pa > pb:
            # Swap so pa < pb, flip sign at end
            pa, pb = pb, pa
            flip = True
        else:
            flip = False
        
        delta = 0.0
        # Laps pa+1 .. pb:
        #   A is on co (ages 1..pb-pa)
        #   B is still on ci (ages pa+1..pb)
        for lap in range(pa + 1, pb + 1):
            age_b_ci = lap            # B's ci tire age
            age_a_co = lap - pa       # A's fresh co tire age
            delta += lap_t(ci, age_b_ci, temp_offset)
            delta -= lap_t(co, age_a_co, temp_offset)
        
        # Laps pb+1 .. N:
        #   A is on co (ages pb-pa+1 .. N-pa)
        #   B is on co (ages 1 .. N-pb)
        d = pb - pa
        for lap in range(pb + 1, N + 1):
            age_a_co = lap - pa      # A's co tire age
            age_b_co = lap - pb      # B's co tire age
            delta += lap_t(co, age_a_co, temp_offset)
            delta -= lap_t(co, age_b_co, temp_offset)
        
        return -delta if flip else delta
    
    return [total_delta_for_pair(pair) for pair in pairs_batch]


def pairwise_accuracy(pairs, params_flat):
    """Fraction of pairs where predicted sign == observed sign."""
    deltas = compute_time_delta_vectorized(pairs, params_flat)
    correct = 0
    for i, pair in enumerate(pairs):
        pred_b_slower = deltas[i] > 0
        actual_b_slower = not pair['a_beats_b']
        if pred_b_slower == actual_b_slower:
            correct += 1
    return correct / len(pairs)


def loss_fn(params_flat, pairs):
    """Hinge loss on pairwise constraints. Minimized = more correct."""
    deltas = compute_time_delta_vectorized(pairs, params_flat)
    loss = 0.0
    for i, pair in enumerate(pairs):
        # target: if a_beats_b, delta should be > 0 (b is slower)
        # if not a_beats_b, delta should be < 0 (b is faster)
        target_sign = 1.0 if not pair['a_beats_b'] else -1.0
        margin = target_sign * deltas[i]
        loss += max(0.0, 1.0 - margin)  # hinge loss
    return loss / len(pairs)


# ── Full race pairwise accuracy (uses complete simulator) ─────────────────────

def build_full_simulator(params_flat):
    """Build a simulate_driver function from params_flat."""
    compounds = ['SOFT', 'MEDIUM', 'HARD']
    n = 7
    p = {}
    for ci, c in enumerate(compounds):
        base = ci * n
        p[c] = {
            'delta':      params_flat[base + 0],
            'fresh_bonus': params_flat[base + 1],
            'break_age':  max(1.0, params_flat[base + 2]),
            'lin1':       params_flat[base + 3],
            'lin2':       params_flat[base + 4],
            'quad':       params_flat[base + 5],
            'temp_coeff': params_flat[base + 6],
        }
    
    def lap_time_fn(base_lap_time, compound, tire_age, track_temp):
        cp = p[compound]
        age_eff = max(0, tire_age - 1)
        temp_offset = float(track_temp) - 30.0
        return (float(base_lap_time)
                + cp['delta']
                + (cp['fresh_bonus'] if tire_age == 1 else 0.0)
                + cp['lin1'] * min(age_eff, cp['break_age'])
                + cp['lin2'] * max(0.0, age_eff - cp['break_age'])
                + cp['quad'] * (age_eff ** 2)
                + cp['temp_coeff'] * temp_offset * age_eff)
    
    def simulate_driver_fn(race_config, strategy):
        total_laps = int(race_config['total_laps'])
        base = float(race_config['base_lap_time'])
        pit_penalty = float(race_config['pit_lane_time'])
        temp = float(race_config['track_temp'])
        
        compound = str(strategy['starting_tire']).upper()
        tire_age = 0
        total_time = 0.0
        
        pit_map = {int(s['lap']): str(s['to_tire']).upper()
                   for s in strategy.get('pit_stops', [])}
        
        for lap in range(1, total_laps + 1):
            tire_age += 1
            total_time += lap_time_fn(base, compound, tire_age, temp)
            if lap in pit_map:
                total_time += pit_penalty
                compound = pit_map[lap]
                tire_age = 0
        
        return total_time
    
    return simulate_driver_fn


def full_pairwise_accuracy_on_races(races, params_flat, max_races=300):
    """Measure pairwise accuracy on full races (not just isolated pairs)."""
    sim_fn = build_full_simulator(params_flat)
    correct = 0
    total = 0
    
    for race in races[:max_races]:
        cfg = race['race_config']
        fp = race['finishing_positions']
        rank = {d: i for i, d in enumerate(fp)}
        
        times = {}
        for pk, strat in race['strategies'].items():
            times[strat['driver_id']] = sim_fn(cfg, strat)
        
        drivers = list(times.keys())
        for i in range(len(drivers)):
            for j in range(i + 1, len(drivers)):
                da, db = drivers[i], drivers[j]
                pred_a_faster = times[da] < times[db]
                actual_a_faster = rank[da] < rank[db]
                if pred_a_faster == actual_a_faster:
                    correct += 1
                total += 1
    
    return correct / total if total else 0.0


def score_on_official_100(params_flat):
    """Score on the 100 official test cases."""
    sim_fn = build_full_simulator(params_flat)
    correct = 0
    
    for i in range(1, 101):
        with open(f'data/test_cases/inputs/test_{i:03d}.json') as f:
            inp = json.load(f)
        with open(f'data/test_cases/expected_outputs/test_{i:03d}.json') as f:
            exp = json.load(f)
        
        cfg = inp['race_config']
        times = {}
        for pk, strat in inp['strategies'].items():
            pos = int(pk[3:])
            times[strat['driver_id']] = (sim_fn(cfg, strat), pos)
        
        sorted_drivers = sorted(times.items(), key=lambda x: (x[1][0], x[1][1]))
        predicted = [d for d, _ in sorted_drivers]
        
        if predicted == exp['finishing_positions']:
            correct += 1
    
    return correct


# ── Initial params from current solver (as starting point) ────────────────────

# Current params mapped to new 7-param format:
# [delta, fresh_bonus, break_age, lin1, lin2, quad, temp_coeff]
INITIAL_PARAMS = np.array([
    # SOFT
    -0.96,   # delta
    -0.31,   # fresh_bonus
    6.0,     # break_age (fade_start)
    0.03,    # lin1 (linear_deg pre-fade)
    0.23,    # lin2 (fade_per_lap post-fade)
    0.0022,  # quad
    0.0024,  # temp_coeff
    # MEDIUM
    -0.15,   # delta
    -0.03,   # fresh_bonus
    12.0,    # break_age
    0.016,   # lin1
    0.015,   # lin2 (late stint)
    0.001,   # quad
    0.0014,  # temp_coeff
    # HARD
    0.68,    # delta
    -0.08,   # fresh_bonus
    13.0,    # break_age
    0.01,    # lin1
    0.0003,  # lin2 (late stint, very flat)
    0.0005,  # quad
    0.0008,  # temp_coeff
])

# Bounds for optimization
BOUNDS = [
    # SOFT
    (-2.0, 0.0),    # delta
    (-1.0, 0.0),    # fresh_bonus
    (2.0, 15.0),    # break_age
    (0.0, 0.3),     # lin1
    (0.0, 1.0),     # lin2
    (0.0, 0.02),    # quad
    (-0.01, 0.02),  # temp_coeff
    # MEDIUM
    (-0.5, 0.5),    # delta
    (-0.2, 0.1),    # fresh_bonus
    (5.0, 25.0),    # break_age
    (0.0, 0.1),     # lin1
    (0.0, 0.2),     # lin2
    (0.0, 0.01),    # quad
    (-0.01, 0.01),  # temp_coeff
    # HARD
    (0.2, 1.5),     # delta
    (-0.3, 0.1),    # fresh_bonus
    (5.0, 30.0),    # break_age
    (0.0, 0.05),    # lin1
    (0.0, 0.05),    # lin2
    (0.0, 0.005),   # quad
    (-0.005, 0.01), # temp_coeff
]


# ── Main experiment ────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("=" * 60)
    print("CONSTRAINT-INVERSION EXPERIMENT")
    print("Formula family: piecewise-linear + quad + temp interaction")
    print("=" * 60)
    
    t0 = time.time()
    
    # Load data
    races = load_all_races(n_files=2)  # 2,000 races
    pairs = build_isolated_pairs(races)
    
    # Subsample for speed (use all if you have time)
    import random
    random.seed(42)
    sample = random.sample(pairs, min(2500, len(pairs)))
    
    # Baseline with current params
    baseline_acc = pairwise_accuracy(sample, INITIAL_PARAMS)
    baseline_score = score_on_official_100(INITIAL_PARAMS)
    print(f"\nBaseline pairwise acc (isolated pairs): {baseline_acc:.4f}")
    print(f"Baseline official score: {baseline_score}/100")
    
    # Phase 1: Fast local optimization (Nelder-Mead from current params)
    print("\nPhase 1: Local optimization from current params...")
    
    def neg_acc(params):
        return -pairwise_accuracy(sample, params)
    
    res1 = minimize(
    neg_acc,
    INITIAL_PARAMS,
    method='Nelder-Mead',
    options={'maxiter': 800, 'xatol': 1e-4, 'fatol': 1e-4, 'disp': True}
)
    
    acc1 = pairwise_accuracy(sample, res1.x)
    score1 = score_on_official_100(res1.x)
    print(f"Phase 1 pairwise acc: {acc1:.4f}")
    print(f"Phase 1 official score: {score1}/100")
    print(f"Phase 1 params: {res1.x}")
    
    # Phase 2: Global search (differential evolution)
    print("\nPhase 2: Global search via differential evolution (may take 10-30 min)...")
    
    res2 = differential_evolution(
    lambda p: loss_fn(p, sample),
    bounds=BOUNDS,
    seed=42,
    maxiter=40,
    popsize=8,
    tol=1e-3,
    mutation=(0.5, 1.0),
    recombination=0.7,
    workers=1,
    disp=True,
)
    
    acc2 = pairwise_accuracy(sample, res2.x)
    score2 = score_on_official_100(res2.x)
    print(f"\nPhase 2 pairwise acc: {acc2:.4f}")
    print(f"Phase 2 official score: {score2}/100")
    print(f"Phase 2 params: {res2.x}")
    
    # Pick best
    best_params = res2.x if score2 >= score1 else res1.x
    best_score = max(score1, score2)
    
    # Full race pairwise check
    full_acc = full_pairwise_accuracy_on_races(races, best_params, max_races=100)
    print(f"\nFull race pairwise accuracy: {full_acc:.4f}")
    
    print(f"\n{'='*60}")
    print(f"RESULT: Best official score = {best_score}/100")
    print(f"Total time: {time.time()-t0:.1f}s")
    print(f"Best params: {best_params}")
    print(f"{'='*60}")
    
    # Save best params to file for use in solver
    with open('tools/inverse_fit_best_params.json', 'w') as f:
        json.dump({
            'pairwise_acc_isolated': float(acc2 if score2 >= score1 else acc1),
            'pairwise_acc_full': float(full_acc),
            'official_score': best_score,
            'params': best_params.tolist()
        }, f, indent=2)
    
    print("\nSaved to tools/inverse_fit_best_params.json")