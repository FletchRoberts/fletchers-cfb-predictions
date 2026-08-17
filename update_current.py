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
import glob
import re
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
HISTORY_DIR = "site/data/history"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(HISTORY_DIR, exist_ok=True)


def fetch(endpoint, params, max_retries=3, timeout=30):
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.get(f"{BASE_URL}{endpoint}", headers=HEADERS, params=params, timeout=timeout)
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.RequestException as e:
            print(f"  Request to {endpoint} failed (attempt {attempt}/{max_retries}): {e}")
            if attempt == max_retries:
                raise
            time.sleep(5 * attempt)


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


def manage_weekly_snapshots(games_current_raw, predictions):
    """Snapshots the next fully-upcoming week's predictions (once, immutably),
    grades any existing snapshots whose games have since completed, and
    recomputes the season-wide accuracy summary. Never overwrites a prediction
    once saved -- only ever fills in actual results after the fact."""
    all_fbs_games = [g for g in games_current_raw if (
        g.get("seasonType") == "regular" and g.get("week") is not None
        and g.get("homeClassification") == "fbs" and g.get("awayClassification") == "fbs")]

    existing_snapshot_weeks = set()
    for p in glob.glob(f"{HISTORY_DIR}/week_*.json"):
        m = re.search(r"week_(\d+)\.json", p)
        if m:
            existing_snapshot_weeks.add(int(m.group(1)))

    # ---- 1. Snapshot the next fully-upcoming week, if there is one and it's not done yet ----
    weeks_present = sorted(set(g["week"] for g in all_fbs_games))
    for w in weeks_present:
        week_games = [g for g in all_fbs_games if g["week"] == w]
        any_completed = any(g.get("completed") for g in week_games)
        if any_completed or w in existing_snapshot_weeks:
            continue  # either already happened (can't honestly snapshot in hindsight) or already saved

        snapshot_games = []
        for g in week_games:
            home, away = g["homeTeam"], g["awayTeam"]
            pred = predictions.get(f"{home}|{away}")
            if not pred:
                continue
            snapshot_games.append({
                "game_id": g["id"], "home": home, "away": away,
                "start_date": g.get("startDate"),
                "predicted_home_score": pred["predicted_home_score"],
                "predicted_away_score": pred["predicted_away_score"],
                "predicted_spread": pred["predicted_spread"],
                "predicted_total": pred["predicted_total"],
                "favorite": pred["favorite"],
                "actual_home_score": None, "actual_away_score": None,
                "graded": False
            })
        with open(f"{HISTORY_DIR}/week_{w}.json", "w") as f:
            json.dump({
                "week": w, "season": CURRENT_SEASON,
                "snapshotted_at": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
                "games": snapshot_games
            }, f, indent=2)
        print(f"  Snapshotted {len(snapshot_games)} games for week {w}.")
        break  # only the nearest upcoming week -- keeps snapshots close to kickoff, not stale

    # ---- 2. Grade any snapshots whose games have since completed ----
    completed_by_id = {g["id"]: g for g in all_fbs_games if g.get("completed")}
    for path in sorted(glob.glob(f"{HISTORY_DIR}/week_*.json")):
        with open(path) as f:
            snap = json.load(f)
        changed = False
        for g in snap["games"]:
            if g["graded"]:
                continue
            actual = completed_by_id.get(g["game_id"])
            if actual:
                g["actual_home_score"] = actual["homePoints"]
                g["actual_away_score"] = actual["awayPoints"]
                g["graded"] = True
                changed = True
        if changed:
            with open(path, "w") as f:
                json.dump(snap, f, indent=2)
            print(f"  Graded newly-completed games in {path}.")

    # ---- 3. Manifest of available weeks (for the site's dropdown) ----
    available_weeks = sorted(int(re.search(r"week_(\d+)\.json", p).group(1))
                              for p in glob.glob(f"{HISTORY_DIR}/week_*.json"))
    with open(f"{HISTORY_DIR}/weeks_available.json", "w") as f:
        json.dump(available_weeks, f)

    # ---- 4. Season-wide accuracy summary across every graded game so far ----
    total_games, correct_winner = 0, 0
    margin_err_sum, total_err_sum = 0.0, 0.0
    for path in glob.glob(f"{HISTORY_DIR}/week_*.json"):
        with open(path) as f:
            snap = json.load(f)
        for g in snap["games"]:
            if not g["graded"]:
                continue
            actual_margin = g["actual_home_score"] - g["actual_away_score"]
            actual_total = g["actual_home_score"] + g["actual_away_score"]
            total_games += 1
            if (g["predicted_spread"] > 0) == (actual_margin > 0):
                correct_winner += 1
            margin_err_sum += abs(g["predicted_spread"] - actual_margin)
            total_err_sum += abs(g["predicted_total"] - actual_total)

    summary = {
        "games_graded": total_games,
        "win_accuracy_pct": round(100 * correct_winner / total_games, 1) if total_games else None,
        "avg_margin_error": round(margin_err_sum / total_games, 2) if total_games else None,
        "avg_total_error": round(total_err_sum / total_games, 2) if total_games else None,
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
    }
    with open(f"{HISTORY_DIR}/summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Season summary: {summary}")


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
    with open(f"{MODELS_DIR}/prior_team_profiles.json") as f:
        prior_team_profiles_data = json.load(f)
    prior_team_profiles = prior_team_profiles_data["profiles"]
    saved_profile_season = prior_team_profiles_data["season"]
    print(f"  Loaded saved team profiles for season {saved_profile_season} "
          f"({len(prior_team_profiles)} teams).")

    if saved_profile_season != PRIOR_SEASON:
        print(f"  WARNING: saved profiles are from {saved_profile_season}, but we need "
              f"{PRIOR_SEASON} (train_models.py hasn't been re-run since the season "
              f"changed). Fetching {PRIOR_SEASON} data fresh instead of using the stale "
              f"file, so predictions stay correct even if the annual retrain was missed.")
        prior_games_fresh = filter_fbs_games(fetch("/games", {"year": PRIOR_SEASON}))
        prior_ppa_fresh = fetch("/ppa/games", {"year": PRIOR_SEASON})

        game_ppa_fresh = {}
        for r in prior_ppa_fresh:
            gid = r.get("gameId")
            if gid is None or not r.get("offense") or not r.get("defense"):
                continue
            if r["offense"].get("overall") is None or r["defense"].get("overall") is None:
                continue
            game_ppa_fresh.setdefault(gid, {})[r["team"]] = {"off": r["offense"]["overall"], "def": r["defense"]["overall"]}

        fresh_scoring, fresh_ppa_agg = {}, {}
        for g in prior_games_fresh:
            home, away = g["homeTeam"], g["awayTeam"]
            hp, ap = g["homePoints"], g["awayPoints"]
            for t, pf, pa in [(home, hp, ap), (away, ap, hp)]:
                s = fresh_scoring.setdefault(t, {"pf": [], "pa": []})
                s["pf"].append(pf); s["pa"].append(pa)
            gp = game_ppa_fresh.get(g["id"])
            if gp:
                for t in [home, away]:
                    if t in gp:
                        p = fresh_ppa_agg.setdefault(t, {"off": [], "def": []})
                        p["off"].append(gp[t]["off"]); p["def"].append(gp[t]["def"])

        prior_team_profiles = {}
        for t in fresh_scoring:
            s = fresh_scoring[t]
            p = fresh_ppa_agg.get(t, {})
            prior_team_profiles[t] = {
                "own_ppg": float(np.mean(s["pf"])) if s["pf"] else 24.0,
                "opp_ppg_allowed": float(np.mean(s["pa"])) if s["pa"] else 24.0,
                "off_ppa": float(np.mean(p.get("off", [0]))) if p.get("off") else 0.0,
                "def_ppa": float(np.mean(p.get("def", [0]))) if p.get("def") else 0.0,
            }
        print(f"  Freshly computed {PRIOR_SEASON} profiles for {len(prior_team_profiles)} teams "
              f"(self-healed instead of using stale {saved_profile_season} data).")

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

    if len(talent_raw) < 20:  # sanity threshold -- a real season has ~130 teams with talent data
        print(f"  Only {len(talent_raw)} teams found in {CURRENT_SEASON} talent data "
              f"(expected ~130) -- likely means this season's Talent Composite isn't "
              f"published yet. Falling back to {PRIOR_SEASON}'s talent data as a cold-start "
              f"proxy (same principle as our prior-season PPG/advanced-stats fallbacks).")
        talent_prior_season = fetch("/talent", {"year": PRIOR_SEASON})
        talent_raw = {r["team"]: r["talent"] for r in talent_prior_season if r.get("talent") is not None}

        if len(talent_raw) < 20:
            raise RuntimeError(
                f"No usable talent data found for either {CURRENT_SEASON} or {PRIOR_SEASON} "
                f"({len(talent_raw)} teams). Cannot compute any team ratings without this. "
                f"Check the CFBD /talent endpoint directly for both years before re-running."
            )

    # -----------------------------------------------------------------
    # Gen 1 current ratings (SRS + Talent blend)
    # -----------------------------------------------------------------
    print("\nComputing Gen 1 current ratings...")
    full_ratings, team_games = compute_srs(games_current)
    all_teams = sorted(team_games.keys()) if team_games else sorted(talent_raw.keys())
    # Include teams with talent data even if they haven't played yet (e.g. week 0/1)
    all_teams = sorted(set(all_teams) | set(talent_raw.keys()))

    talent_vals = np.array([talent_raw[t] for t in all_teams if t in talent_raw])
    talent_mean, talent_std = talent_vals.mean(), talent_vals.std()
    if len(full_ratings) > 0:
        srs_vals = np.array(list(full_ratings.values()))
        srs_std = srs_vals.std() if srs_vals.std() > 1e-6 else 1.0
    else:
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

    # League-average fallback ONLY for teams with neither current-season games NOR
    # prior-season data at all (e.g. a team brand new to FBS this year) -- everyone
    # else gets a real, differentiated blend instead of a generic flat number.
    league_avg_prior_ppg = float(np.mean([p["own_ppg"] for p in prior_team_profiles.values()])) if prior_team_profiles else 24.0
    league_avg_prior_opp_ppg = float(np.mean([p["opp_ppg_allowed"] for p in prior_team_profiles.values()])) if prior_team_profiles else 24.0
    league_avg_prior_off_ppa = float(np.mean([p["off_ppa"] for p in prior_team_profiles.values()])) if prior_team_profiles else 0.0
    league_avg_prior_def_ppa = float(np.mean([p["def_ppa"] for p in prior_team_profiles.values()])) if prior_team_profiles else 0.0

    team_profiles = {}
    for t in all_teams:
        s = scoring.get(t)
        p = ppa_agg.get(t)
        games_played_this_season = len(s["pf"]) if s else 0

        current = {
            "own_ppg": float(np.mean(s["pf"])) if s and s["pf"] else None,
            "opp_ppg_allowed": float(np.mean(s["pa"])) if s and s["pa"] else None,
            "off_ppa": float(np.mean(p["off"])) if p and p["off"] else None,
            "def_ppa": float(np.mean(p["def"])) if p and p["def"] else None,
        }
        prior = prior_team_profiles.get(t, {
            "own_ppg": league_avg_prior_ppg, "opp_ppg_allowed": league_avg_prior_opp_ppg,
            "off_ppa": league_avg_prior_off_ppa, "def_ppa": league_avg_prior_def_ppa,
        })

        # Same games-played trust ramp used throughout this project for margin/total blending
        w = min(games_played_this_season / GAMES_TO_FULL_TRUST_MARGIN, 1)
        blended = {}
        for key in ["own_ppg", "opp_ppg_allowed", "off_ppa", "def_ppa"]:
            cur_val = current[key]
            prior_val = prior.get(key, 0.0)
            blended[key] = (w * cur_val + (1 - w) * prior_val) if cur_val is not None else prior_val

        team_profiles[t] = {
            "own_ppg": blended["own_ppg"],
            "opp_ppg_allowed": blended["opp_ppg_allowed"],
            "off_ppa": blended["off_ppa"],
            "def_ppa": blended["def_ppa"],
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
    # Weekly prediction snapshots (for grading against real results later)
    # -----------------------------------------------------------------
    print("\nManaging weekly prediction snapshots...")
    manage_weekly_snapshots(games_current_raw, predictions)

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
