"""
update_current.py -- Stage 2: the FREQUENT, cheap job.

Run this daily (or whatever cadence you choose) via a scheduled GitHub Action.
Loads already-trained models from Stage 1 (train_models.py) -- does NOT
retrain anything. Pulls only the current season's data (cheap: ~3 API calls),
computes current ratings/profiles, and precomputes predictions for every
possible team matchup, since the static site can't run Python/XGBoost live
in a visitor's browser.

Output: JSON files in site/data/, which the static site reads directly.
"""

import os
import json
import time
import requests
import numpy as np
import pandas as pd
import joblib

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
API_KEY = os.environ.get("CFBD_API_KEY")
if not API_KEY:
    raise RuntimeError("CFBD_API_KEY environment variable not set.")

HEADERS = {"Authorization": f"Bearer {API_KEY}", "Accept": "application/json"}
BASE_URL = "https://api.collegefootballdata.com"

CURRENT_SEASON = 2026  # update this once per year when the season changes
PRIOR_SEASON = CURRENT_SEASON - 1

MODELS_DIR = "models"
OUTPUT_DIR = "site/data"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def fetch(endpoint, params):
    resp = requests.get(f"{BASE_URL}{endpoint}", headers=HEADERS, params=params)
    resp.raise_for_status()
    return resp.json()


def filter_fbs_games(games_list):
    return [g for g in games_list if (
        g.get("seasonType") == "regular" and g.get("completed")
        and g.get("homePoints") is not None and g.get("awayPoints") is not None
        and g.get("week") is not None
        and g.get("homeClassification") == "fbs" and g.get("awayClassification") == "fbs")]


def compute_srs(game_list):
    team_games = {}
    for g in game_list:
        home, away = g["homeTeam"], g["awayTeam"]
        hp, ap = g["homePoints"], g["awayPoints"]
        team_games.setdefault(home, []).append((away, hp - ap))
        team_games.setdefault(away, []).append((home, ap - hp))
    team_list = list(team_games.keys())
    if not team_list:
        return {}, {}
    ratings = {t: 0.0 for t in team_list}
    for _ in range(50):
        next_ratings = {}
        for team in team_list:
            s = sum(margin + ratings.get(opp, 0) for opp, margin in team_games[team])
            next_ratings[team] = s / len(team_games[team])
        avg = sum(next_ratings.values()) / len(team_list)
        for t in team_list:
            next_ratings[t] -= avg
        ratings = next_ratings
    return ratings, team_games


def main():
    print(f"=== Stage 2 update run for {CURRENT_SEASON} season ===\n")

    # -----------------------------------------------------------------
    # Load Stage 1 artifacts (trained models, weights, config)
    # -----------------------------------------------------------------
    print("Loading trained models and metadata...")
    with open(f"{MODELS_DIR}/metadata.json") as f:
        meta = json.load(f)
    with open(f"{MODELS_DIR}/prior_advanced_lookup.json") as f:
        prior_advanced_lookup = json.load(f)

    gen2_margin_model = joblib.load(f"{MODELS_DIR}/gen2_margin_model.joblib")
    gen2_total_model = joblib.load(f"{MODELS_DIR}/gen2_total_model.joblib")
    off_model = joblib.load(f"{MODELS_DIR}/gen3_offense_model.joblib")
    def_model = joblib.load(f"{MODELS_DIR}/gen3_defense_model.joblib")

    feature_cols_gen2 = meta["feature_cols_gen2"]
    adv_feature_names = meta["adv_feature_names"]
    off_features = meta["off_features"]
    def_features = meta["def_features"]
    HFA = meta["hfa"]
    GAMES_TO_FULL_TRUST_MARGIN = meta["games_to_full_trust_margin"]

    spread_weights = dict(zip(meta["spread_metamodel"]["selected"], meta["spread_metamodel"]["weights"]))
    total_weights = dict(zip(meta["total_metamodel"]["selected"], meta["total_metamodel"]["weights"]))
    print(f"  Spread weights: {spread_weights}")
    print(f"  Total weights: {total_weights}")

    # Prior-season advanced stats: check if Stage 1 already has it saved (it will,
    # as long as train_models.py has been re-run at least once since PRIOR_SEASON
    # completed). Fallback: fetch fresh if missing (rare, one extra cheap call).
    prior_adv = prior_advanced_lookup.get(str(PRIOR_SEASON))
    if prior_adv is None:
        print(f"  Prior-season ({PRIOR_SEASON}) advanced stats not found in saved training "
              f"data -- fetching fresh as a fallback. Consider re-running train_models.py "
              f"to bake this in properly.")
        raw = fetch("/stats/season/advanced", {"year": PRIOR_SEASON})
        adv_fields = meta["adv_fields"]
        lookup = {}
        for rec in raw:
            team = rec.get("team")
            if not team:
                continue
            row = {}
            for side, field in adv_fields:
                row[f"{side}_{field}"] = rec.get(side, {}).get(field)
            row["offense_havoc"] = rec.get("offense", {}).get("havoc", {}).get("total")
            row["defense_havoc"] = rec.get("defense", {}).get("havoc", {}).get("total")
            lookup[team] = row
        prior_adv = lookup

    def get_prior_advanced(team):
        return prior_adv.get(team, {})

    # -----------------------------------------------------------------
    # Pull current-season data (cheap: 3 calls total)
    # -----------------------------------------------------------------
    print(f"\nPulling {CURRENT_SEASON} data...")
    games_current_raw = fetch("/games", {"year": CURRENT_SEASON})
    talent_current = fetch("/talent", {"year": CURRENT_SEASON})
    ppa_current = fetch("/ppa/games", {"year": CURRENT_SEASON})

    games_current = filter_fbs_games(games_current_raw)
    print(f"  {len(games_current)} completed FBS games so far this season.")

    talent_raw = {r["team"]: r["talent"] for r in talent_current if r.get("talent") is not None}

    # -----------------------------------------------------------------
    # Gen 1 current ratings (SRS + Talent blend)
    # -----------------------------------------------------------------
    print("\nComputing Gen 1 current ratings...")
    full_ratings, team_games = compute_srs(games_current)
    all_teams = sorted(team_games.keys()) if team_games else sorted(talent_raw.keys())
    # Include teams with talent data even if they haven't played yet (e.g. week 0/1)
    all_teams = sorted(set(all_teams) | set(talent_raw.keys()))

    if len(full_ratings) > 0:
        talent_vals = np.array([talent_raw[t] for t in all_teams if t in talent_raw])
        talent_mean, talent_std = talent_vals.mean(), talent_vals.std()
        srs_vals = np.array(list(full_ratings.values()))
        srs_std = srs_vals.std() if srs_vals.std() > 1e-6 else 1.0
    else:
        talent_vals = np.array([talent_raw[t] for t in all_teams if t in talent_raw])
        talent_mean, talent_std = talent_vals.mean(), talent_vals.std()
        srs_std = 15.0  # reasonable fallback scale if literally zero games played yet anywhere

    def talent_prior(team):
        if team not in talent_raw:
            return 0.0
        return ((talent_raw[team] - talent_mean) / talent_std) * srs_std

    gen1_current = {}
    for team in all_teams:
        gp = len(team_games.get(team, []))
        w = min(gp / GAMES_TO_FULL_TRUST_MARGIN, 1)
        gen1_current[team] = w * full_ratings.get(team, 0.0) + (1 - w) * talent_prior(team)

    # -----------------------------------------------------------------
    # Gen 3 current team profiles (rolling PPG/PPA)
    # -----------------------------------------------------------------
    print("Computing Gen 3 current team profiles...")
    game_ppa = {}
    for r in ppa_current:
        gid = r.get("gameId")
        if gid is None or not r.get("offense") or not r.get("defense"):
            continue
        if r["offense"].get("overall") is None or r["defense"].get("overall") is None:
            continue
        game_ppa.setdefault(gid, {})[r["team"]] = {"off": r["offense"]["overall"], "def": r["defense"]["overall"]}

    scoring, ppa_agg = {}, {}
    for g in games_current:
        home, away = g["homeTeam"], g["awayTeam"]
        hp, ap = g["homePoints"], g["awayPoints"]
        for t, pf, pa in [(home, hp, ap), (away, ap, hp)]:
            s = scoring.setdefault(t, {"pf": [], "pa": []})
            s["pf"].append(pf); s["pa"].append(pa)
        gp = game_ppa.get(g["id"])
        if gp:
            for t in [home, away]:
                if t in gp:
                    p = ppa_agg.setdefault(t, {"off": [], "def": []})
                    p["off"].append(gp[t]["off"]); p["def"].append(gp[t]["def"])

    team_profiles = {}
    for t in all_teams:
        s = scoring.get(t)
        p = ppa_agg.get(t)
        team_profiles[t] = {
            "own_ppg": float(np.mean(s["pf"])) if s and s["pf"] else 24.0,
            "opp_ppg_allowed": float(np.mean(s["pa"])) if s and s["pa"] else 24.0,
            "off_ppa": float(np.mean(p["off"])) if p and p["off"] else 0.0,
            "def_ppa": float(np.mean(p["def"])) if p and p["def"] else 0.0,
            "games_played": len(s["pf"]) if s else 0
        }

    # -----------------------------------------------------------------
    # Precompute every ordered matchup (home, away), batched for speed
    # -----------------------------------------------------------------
    print(f"\nPrecomputing all matchups among {len(all_teams)} teams...")
    pairs = [(h, a) for h in all_teams for a in all_teams if h != a]
    print(f"  {len(pairs)} ordered pairs to compute.")

    # Build batch input dataframes for all four models at once (vectorized -- much
    # faster than looping and calling .predict() on single rows 4x per pair)
    off_rows, def_rows, gen2_rows, gen1_margins = [], [], [], []
    for home, away in pairs:
        hp, ap = team_profiles[home], team_profiles[away]
        gen1_margin = gen1_current[home] - gen1_current[away] + HFA
        gen1_margins.append(gen1_margin)

        off_rows.append({
            "own_ppg": hp["own_ppg"], "own_off_ppa": hp["off_ppa"],
            "opp_ppg_allowed": ap["opp_ppg_allowed"], "opp_def_ppa": ap["def_ppa"],
            "games_played": hp["games_played"]
        })
        def_rows.append({
            "own_opp_ppg": hp["opp_ppg_allowed"], "own_def_ppa": hp["def_ppa"],
            "opp_own_ppg": ap["own_ppg"], "opp_off_ppa": ap["off_ppa"],
            "games_played": hp["games_played"]
        })
        # Mirror rows for the away team's own offense/defense (needed for gen3_away_score)
        off_rows.append({
            "own_ppg": ap["own_ppg"], "own_off_ppa": ap["off_ppa"],
            "opp_ppg_allowed": hp["opp_ppg_allowed"], "opp_def_ppa": hp["def_ppa"],
            "games_played": ap["games_played"]
        })
        def_rows.append({
            "own_opp_ppg": ap["opp_ppg_allowed"], "own_def_ppa": ap["def_ppa"],
            "opp_own_ppg": hp["own_ppg"], "opp_off_ppa": hp["off_ppa"],
            "games_played": ap["games_played"]
        })

        home_adv, away_adv = get_prior_advanced(home), get_prior_advanced(away)
        gen2_feat = {"gen1_pred": gen1_margin}
        for name in adv_feature_names:
            gen2_feat[f"home_{name}"] = home_adv.get(name)
            gen2_feat[f"away_{name}"] = away_adv.get(name)
        gen2_rows.append(gen2_feat)

    off_batch = pd.DataFrame(off_rows)[off_features]
    def_batch = pd.DataFrame(def_rows)[def_features]
    gen2_batch = pd.DataFrame(gen2_rows)[feature_cols_gen2]
    gen2_batch = gen2_batch.fillna(gen2_batch.median())

    print("  Running batched predictions (4 vectorized model calls)...")
    off_preds = off_model.predict(off_batch)
    def_preds = def_model.predict(def_batch)
    gen2_margin_preds = gen2_margin_model.predict(gen2_batch)
    gen2_total_preds = gen2_total_model.predict(gen2_batch)

    # off_preds/def_preds are interleaved (home, away, home, away, ...) since we
    # appended two rows per pair -- split back out
    off_home = off_preds[0::2]
    off_away = off_preds[1::2]
    def_home = def_preds[0::2]  # home's defense prediction (points home would allow)
    def_away = def_preds[1::2]  # away's defense prediction (points away would allow)

    gen3_home_score = (off_home + def_away) / 2
    gen3_away_score = (off_away + def_home) / 2
    gen3_margins = gen3_home_score - gen3_away_score
    gen3_totals = gen3_home_score + gen3_away_score

    # -----------------------------------------------------------------
    # Combine via the metamodel weights, assemble final predictions
    # -----------------------------------------------------------------
    print("Combining via metamodel weights...")
    gen1_margins_arr = np.array(gen1_margins)

    final_margins = (
        spread_weights.get("gen1", 0) * gen1_margins_arr +
        spread_weights.get("gen2", 0) * gen2_margin_preds +
        spread_weights.get("gen3", 0) * gen3_margins
    )
    # Our final validated total metamodel only includes gen2_total and gen3_total
    # (gen1_total was tested and excluded during Colab validation) -- warn loudly if
    # metadata ever changes and includes gen1_total, since this script doesn't compute it.
    if "gen1_total" in total_weights and total_weights["gen1_total"] != 0:
        raise RuntimeError(
            "metadata.json's total_metamodel includes 'gen1_total' with nonzero weight, "
            "but this script never computes a gen1_total prediction. Update this script "
            "to add that computation before proceeding, rather than silently producing "
            "wrong totals."
        )

    final_totals = (
        total_weights.get("gen2_total", 0) * gen2_total_preds +
        total_weights.get("gen3_total", 0) * gen3_totals
    )

    final_home_scores = (final_totals + final_margins) / 2
    final_away_scores = (final_totals - final_margins) / 2

    predictions = {}
    for i, (home, away) in enumerate(pairs):
        predictions[f"{home}|{away}"] = {
            "home": home, "away": away,
            "predicted_home_score": round(float(final_home_scores[i]), 1),
            "predicted_away_score": round(float(final_away_scores[i]), 1),
            "predicted_spread": round(float(final_margins[i]), 1),
            "predicted_total": round(float(final_totals[i]), 1),
            "favorite": home if final_margins[i] > 0 else away
        }

    # -----------------------------------------------------------------
    # Power rankings (vs. league-average opponent, same as Colab Part 5)
    # -----------------------------------------------------------------
    print("Computing power rankings...")
    league_avg_ppg_allowed = float(np.mean([p["opp_ppg_allowed"] for p in team_profiles.values()]))
    league_avg_def_ppa = float(np.mean([p["def_ppa"] for p in team_profiles.values()]))
    league_avg_ppg = float(np.mean([p["own_ppg"] for p in team_profiles.values()]))
    league_avg_off_ppa = float(np.mean([p["off_ppa"] for p in team_profiles.values()]))

    rank_off_rows, rank_def_rows = [], []
    for t in all_teams:
        p = team_profiles[t]
        rank_off_rows.append({"own_ppg": p["own_ppg"], "own_off_ppa": p["off_ppa"],
                               "opp_ppg_allowed": league_avg_ppg_allowed, "opp_def_ppa": league_avg_def_ppa,
                               "games_played": p["games_played"]})
        rank_def_rows.append({"own_opp_ppg": p["opp_ppg_allowed"], "own_def_ppa": p["def_ppa"],
                               "opp_own_ppg": league_avg_ppg, "opp_off_ppa": league_avg_off_ppa,
                               "games_played": p["games_played"]})

    rank_off_batch = pd.DataFrame(rank_off_rows)[off_features]
    rank_def_batch = pd.DataFrame(rank_def_rows)[def_features]
    rank_off_preds = off_model.predict(rank_off_batch)
    rank_def_preds = def_model.predict(rank_def_batch)
    gen3_vs_avg = rank_off_preds - rank_def_preds

    gen1_mean = float(np.mean(list(gen1_current.values())))
    rankings = []
    for i, t in enumerate(all_teams):
        composite = (spread_weights.get("gen1", 0) * (gen1_current[t] - gen1_mean) +
                     spread_weights.get("gen3", 0) * float(gen3_vs_avg[i]))
        # NOTE: gen2's ranking contribution is intentionally omitted here for simplicity/
        # robustness -- it requires a league-average proxy matchup (see Colab Part 5 Step 3)
        # which adds complexity for a relatively small effect on ranking order. The live
        # matchup predictor (predictions dict above) DOES use the full 3-system model.
        rankings.append({"team": t, "composite_rating": round(composite, 2),
                          "gen1": round(gen1_current[t], 2), "games_played": team_profiles[t]["games_played"]})

    rankings.sort(key=lambda r: -r["composite_rating"])
    for i, r in enumerate(rankings):
        r["rank"] = i + 1

    # -----------------------------------------------------------------
    # Save output
    # -----------------------------------------------------------------
    print(f"\nSaving output to {OUTPUT_DIR}/...")
    with open(f"{OUTPUT_DIR}/predictions.json", "w") as f:
        json.dump(predictions, f)
    with open(f"{OUTPUT_DIR}/rankings.json", "w") as f:
        json.dump(rankings, f, indent=2)
    with open(f"{OUTPUT_DIR}/teams.json", "w") as f:
        json.dump(sorted(all_teams), f)
    with open(f"{OUTPUT_DIR}/last_updated.json", "w") as f:
        json.dump({
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
            "season": CURRENT_SEASON,
            "games_processed": len(games_current)
        }, f, indent=2)

    print(f"Done. {len(predictions)} matchups precomputed, {len(rankings)} teams ranked.")


if __name__ == "__main__":
    main()
