"""
train_models.py -- Stage 1: the RARE, expensive job.

Run this: once now to bootstrap, then once per year when a season completes
and should be folded into training. NOT part of the daily refresh -- that's
update_current.py, a separate, much cheaper script.

What this does:
1. Pulls the full historical dataset (2021-2025 by default).
2. Trains Gen 2 (kitchen-sink XGBoost) and Gen 3 (Offense/Defense models) for
   both spread (margin) and total.
3. Fits both metamodels (spread: Gen1+Gen2+Gen3; total: Gen1+Gen2+Gen3) via
   the same Coleman-style forward selection validated in Colab.
4. Saves every trained model + metamodel weights + supporting metadata to
   the models/ directory, so update_current.py can load them without ever
   retraining.

API key: read from the CFBD_API_KEY environment variable (GitHub Actions
injects repository secrets this way -- different from Colab's userdata
secrets manager, but same underlying principle: never hardcoded in code).
"""

import os
import json
import time
import requests
import numpy as np
import pandas as pd
import xgboost as xgb
import joblib
from scipy.optimize import nnls
from sklearn.model_selection import KFold

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
API_KEY = os.environ.get("CFBD_API_KEY")
if not API_KEY:
    raise RuntimeError("CFBD_API_KEY environment variable not set. "
                        "In GitHub Actions, this comes from a repository secret.")

HEADERS = {"Authorization": f"Bearer {API_KEY}", "Accept": "application/json"}
BASE_URL = "https://api.collegefootballdata.com"

TRAIN_SEASONS = [2021, 2022, 2023, 2024, 2025]  # update this list each year at retrain time
HFA = 2.5
GAMES_TO_FULL_TRUST_MARGIN = 8
ASSUMED_PLAYS_PER_GAME = 70

MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

ADV_FIELDS = [
    ("offense", "successRate"), ("defense", "successRate"),
    ("offense", "explosiveness"), ("defense", "explosiveness"),
    ("offense", "powerSuccess"), ("defense", "powerSuccess"),
    ("offense", "stuffRate"), ("defense", "stuffRate"),
    ("offense", "lineYards"), ("defense", "lineYards"),
    ("offense", "secondLevelYards"), ("defense", "secondLevelYards"),
    ("offense", "openFieldYards"), ("defense", "openFieldYards"),
    ("offense", "pointsPerOpportunity"), ("defense", "pointsPerOpportunity"),
]
ADV_FEATURE_NAMES = [f"{side}_{field}" for side, field in ADV_FIELDS] + ["offense_havoc", "defense_havoc"]


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


def forward_selection_nnls(systems_dict, y, n_folds=6):
    remaining = list(systems_dict.keys())
    selected = []
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    def cv_error(selected_systems):
        X = np.column_stack([systems_dict[s] for s in selected_systems])
        total_sse = 0
        for train_idx, test_idx in kf.split(X):
            coefs, _ = nnls(X[train_idx], y.values[train_idx])
            pred = X[test_idx] @ coefs
            total_sse += np.sum((y.values[test_idx] - pred) ** 2)
        return total_sse

    best_overall = None
    while remaining:
        best_system, best_error = None, None
        for candidate in remaining:
            err = cv_error(selected + [candidate])
            if best_error is None or err < best_error:
                best_error, best_system = err, candidate
        if best_overall is not None and best_error >= best_overall:
            break
        selected.append(best_system)
        remaining.remove(best_system)
        best_overall = best_error
        print(f"    Added '{best_system}' -- holdout SSE: {best_error:,.0f}")

    X_final = np.column_stack([systems_dict[s] for s in selected])
    final_coefs, _ = nnls(X_final, y.values)
    return selected, final_coefs


def main():
    print(f"=== Stage 1 training run, seasons: {TRAIN_SEASONS} ===\n")

    # -----------------------------------------------------------------
    # Step 1: Pull historical data
    # -----------------------------------------------------------------
    print("Pulling historical data...")
    all_games, all_talent, all_ppa, all_advanced = {}, {}, {}, {}
    for year in TRAIN_SEASONS:
        print(f"  {year}...")
        all_games[year] = fetch("/games", {"year": year})
        all_talent[year] = fetch("/talent", {"year": year})
        all_ppa[year] = fetch("/ppa/games", {"year": year})
        all_advanced[year] = fetch("/stats/season/advanced", {"year": year})
        time.sleep(0.5)

    fbs_games = {year: filter_fbs_games(all_games[year]) for year in TRAIN_SEASONS}
    TEST_SEASON = TRAIN_SEASONS[-1]  # most recent season = held-out validation, matches Colab methodology

    all_games_combined = [g for year in TRAIN_SEASONS for g in fbs_games[year]]

    # -----------------------------------------------------------------
    # Step 2: Gen 1 predictions (SRS + Talent) -- needed as a Gen 2 input feature
    # -----------------------------------------------------------------
    print("\nBuilding Gen 1 (SRS + Talent) predictions...")

    def build_gen1_predictions(games, talent_data, year):
        talent_raw = {r["team"]: r["talent"] for r in talent_data if r.get("talent") is not None}
        talent_vals = np.array(list(talent_raw.values()))
        talent_mean, talent_std = talent_vals.mean(), talent_vals.std()

        full_ratings, _ = compute_srs(games)
        full_vals = np.array(list(full_ratings.values()))
        srs_std = full_vals.std() if len(full_vals) else 1.0

        def talent_prior(team):
            if team not in talent_raw:
                return 0.0
            return ((talent_raw[team] - talent_mean) / talent_std) * srs_std

        weeks = sorted(set(g["week"] for g in games))
        rows = []
        for w in weeks:
            games_before = [g for g in games if g["week"] < w]
            ratings, team_games = compute_srs(games_before)

            def games_played(team):
                return len(team_games.get(team, []))

            week_games = [g for g in games if g["week"] == w]
            for g in week_games:
                home, away = g["homeTeam"], g["awayTeam"]
                home_gp, away_gp = games_played(home), games_played(away)
                w_home = min(home_gp / GAMES_TO_FULL_TRUST_MARGIN, 1)
                w_away = min(away_gp / GAMES_TO_FULL_TRUST_MARGIN, 1)
                blended_home = w_home * ratings.get(home, 0.0) + (1 - w_home) * talent_prior(home)
                blended_away = w_away * ratings.get(away, 0.0) + (1 - w_away) * talent_prior(away)
                gen1_margin = blended_home - blended_away + HFA

                # rolling PPG, for gen1_total
                pf_home = pf_away = pa_home = pa_away = None
                scoring = {}
                for gb in games_before:
                    h, a = gb["homeTeam"], gb["awayTeam"]
                    hp, ap = gb["homePoints"], gb["awayPoints"]
                    for t, pf, pa in [(h, hp, ap), (a, ap, hp)]:
                        s = scoring.setdefault(t, {"pf": [], "pa": []})
                        s["pf"].append(pf); s["pa"].append(pa)

                def ppg(t):
                    s = scoring.get(t)
                    if not s or not s["pf"]:
                        return 24.0, 24.0
                    return np.mean(s["pf"]), np.mean(s["pa"])

                hppg, hoppg = ppg(home)
                appg, aoppg = ppg(away)
                gen1_total = ((hppg + aoppg) + (appg + hoppg)) / 2

                rows.append({
                    "season": year, "week": w, "game_id": g["id"], "home": home, "away": away,
                    "actual_margin": g["homePoints"] - g["awayPoints"],
                    "actual_total": g["homePoints"] + g["awayPoints"],
                    "gen1_pred": gen1_margin, "gen1_total_pred": gen1_total
                })
        return pd.DataFrame(rows)

    gen1_frames = [build_gen1_predictions(fbs_games[y], all_talent[y], y) for y in TRAIN_SEASONS]
    gen1_df = pd.concat(gen1_frames, ignore_index=True)

    # -----------------------------------------------------------------
    # Step 3: Gen 2 (kitchen-sink XGBoost) -- margin and total
    # -----------------------------------------------------------------
    print("\nBuilding Gen 2 kitchen-sink features...")

    def build_advanced_lookup(advanced_data):
        lookup = {}
        for rec in advanced_data:
            team = rec.get("team")
            if not team:
                continue
            row = {}
            for side, field in ADV_FIELDS:
                row[f"{side}_{field}"] = rec.get(side, {}).get(field)
            row["offense_havoc"] = rec.get("offense", {}).get("havoc", {}).get("total")
            row["defense_havoc"] = rec.get("defense", {}).get("havoc", {}).get("total")
            lookup[team] = row
        return lookup

    prior_advanced_lookup = {year: build_advanced_lookup(all_advanced[year]) for year in TRAIN_SEASONS}

    def get_prior_advanced(team, season):
        return prior_advanced_lookup.get(season - 1, {}).get(team, {})

    kitchen_sink_rows = []
    for _, row in gen1_df.iterrows():
        home_adv = get_prior_advanced(row["home"], row["season"])
        away_adv = get_prior_advanced(row["away"], row["season"])
        feat = {"gen1_pred": row["gen1_pred"]}
        for name in ADV_FEATURE_NAMES:
            feat[f"home_{name}"] = home_adv.get(name)
            feat[f"away_{name}"] = away_adv.get(name)
        feat.update({
            "season": row["season"], "week": row["week"], "game_id": row["game_id"],
            "home": row["home"], "away": row["away"],
            "actual_margin": row["actual_margin"], "actual_total": row["actual_total"]
        })
        kitchen_sink_rows.append(feat)

    kitchen_sink_df = pd.DataFrame(kitchen_sink_rows)
    feature_cols_gen2 = ["gen1_pred"] + [f"home_{n}" for n in ADV_FEATURE_NAMES] + [f"away_{n}" for n in ADV_FEATURE_NAMES]
    kitchen_sink_df = kitchen_sink_df.dropna(subset=feature_cols_gen2, thresh=int(len(feature_cols_gen2) * 0.7))
    kitchen_sink_df[feature_cols_gen2] = kitchen_sink_df[feature_cols_gen2].fillna(kitchen_sink_df[feature_cols_gen2].median())

    train_mask = kitchen_sink_df["season"] < TEST_SEASON
    test_mask = kitchen_sink_df["season"] == TEST_SEASON
    train_df = kitchen_sink_df[train_mask].reset_index(drop=True)
    test_df = kitchen_sink_df[test_mask].reset_index(drop=True)

    print("Training Gen 2 margin model (out-of-fold, 6-fold)...")
    X_train, y_train = train_df[feature_cols_gen2], train_df["actual_margin"]
    kf = KFold(n_splits=6, shuffle=True, random_state=42)
    oof_preds = np.zeros(len(train_df))
    for fit_idx, hold_idx in kf.split(X_train):
        m = xgb.XGBRegressor(n_estimators=200, max_depth=4, learning_rate=0.05,
                              subsample=0.8, colsample_bytree=0.8, random_state=42)
        m.fit(X_train.iloc[fit_idx], y_train.iloc[fit_idx])
        oof_preds[hold_idx] = m.predict(X_train.iloc[hold_idx])
    train_df = train_df.copy()
    train_df["gen2_pred"] = oof_preds

    gen2_final_model = xgb.XGBRegressor(n_estimators=200, max_depth=4, learning_rate=0.05,
                                          subsample=0.8, colsample_bytree=0.8, random_state=42)
    gen2_final_model.fit(X_train, y_train)
    test_df = test_df.copy()
    test_df["gen2_pred"] = gen2_final_model.predict(test_df[feature_cols_gen2])

    print("Training Gen 2 total model (out-of-fold, 6-fold)...")
    y_train_t = train_df["actual_total"]
    oof_preds_t = np.zeros(len(train_df))
    for fit_idx, hold_idx in kf.split(X_train):
        m = xgb.XGBRegressor(n_estimators=200, max_depth=4, learning_rate=0.05,
                              subsample=0.8, colsample_bytree=0.8, random_state=42)
        m.fit(X_train.iloc[fit_idx], y_train_t.iloc[fit_idx])
        oof_preds_t[hold_idx] = m.predict(X_train.iloc[hold_idx])
    train_df["gen2_total_pred"] = oof_preds_t

    gen2_total_final_model = xgb.XGBRegressor(n_estimators=200, max_depth=4, learning_rate=0.05,
                                                subsample=0.8, colsample_bytree=0.8, random_state=42)
    gen2_total_final_model.fit(X_train, y_train_t)
    test_df["gen2_total_pred"] = gen2_total_final_model.predict(test_df[feature_cols_gen2])

    # -----------------------------------------------------------------
    # Step 4: Gen 3 (Offense/Defense architecture) -- margin and total
    # -----------------------------------------------------------------
    print("\nBuilding Gen 3 (Offense/Defense) rolling features...")

    def build_rolling_scoring_and_ppa(games, ppa_data, year):
        game_ppa = {}
        for r in ppa_data:
            gid = r.get("gameId")
            if gid is None or not r.get("offense") or not r.get("defense"):
                continue
            if r["offense"].get("overall") is None or r["defense"].get("overall") is None:
                continue
            game_ppa.setdefault(gid, {})[r["team"]] = {"off": r["offense"]["overall"], "def": r["defense"]["overall"]}

        weeks = sorted(set(g["week"] for g in games))
        rows = []
        for w in weeks:
            games_before = [g for g in games if g["week"] < w]
            scoring, ppa_agg = {}, {}
            for g in games_before:
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

            def own_ppg(t):
                s = scoring.get(t)
                return np.mean(s["pf"]) if s and s["pf"] else 24.0
            def own_opp_ppg(t):
                s = scoring.get(t)
                return np.mean(s["pa"]) if s and s["pa"] else 24.0
            def own_off_ppa(t):
                p = ppa_agg.get(t)
                return np.mean(p["off"]) if p and p["off"] else 0.0
            def own_def_ppa(t):
                p = ppa_agg.get(t)
                return np.mean(p["def"]) if p and p["def"] else 0.0
            def games_played(t):
                return len(scoring.get(t, {}).get("pf", []))

            week_games = [g for g in games if g["week"] == w]
            for g in week_games:
                home, away = g["homeTeam"], g["awayTeam"]
                rows.append({
                    "season": year, "week": w, "game_id": g["id"], "home": home, "away": away,
                    "home_score": g["homePoints"], "away_score": g["awayPoints"],
                    "home_own_ppg": own_ppg(home), "home_opp_ppg_allowed": own_opp_ppg(home),
                    "home_off_ppa": own_off_ppa(home), "home_def_ppa": own_def_ppa(home),
                    "home_games_played": games_played(home),
                    "away_own_ppg": own_ppg(away), "away_opp_ppg_allowed": own_opp_ppg(away),
                    "away_off_ppa": own_off_ppa(away), "away_def_ppa": own_def_ppa(away),
                    "away_games_played": games_played(away),
                })
        return pd.DataFrame(rows)

    rolling_frames = [build_rolling_scoring_and_ppa(fbs_games[y], all_ppa[y], y) for y in TRAIN_SEASONS]
    rolling_df = pd.concat(rolling_frames, ignore_index=True)

    offense_rows, defense_rows = [], []
    for _, g in rolling_df.iterrows():
        offense_rows.append({"season": g["season"], "game_id": g["game_id"], "side": "home",
            "own_ppg": g["home_own_ppg"], "own_off_ppa": g["home_off_ppa"],
            "opp_ppg_allowed": g["away_opp_ppg_allowed"], "opp_def_ppa": g["away_def_ppa"],
            "games_played": g["home_games_played"], "target": g["home_score"]})
        offense_rows.append({"season": g["season"], "game_id": g["game_id"], "side": "away",
            "own_ppg": g["away_own_ppg"], "own_off_ppa": g["away_off_ppa"],
            "opp_ppg_allowed": g["home_opp_ppg_allowed"], "opp_def_ppa": g["home_def_ppa"],
            "games_played": g["away_games_played"], "target": g["away_score"]})
        defense_rows.append({"season": g["season"], "game_id": g["game_id"], "side": "home",
            "own_opp_ppg": g["home_opp_ppg_allowed"], "own_def_ppa": g["home_def_ppa"],
            "opp_own_ppg": g["away_own_ppg"], "opp_off_ppa": g["away_off_ppa"],
            "games_played": g["home_games_played"], "target": g["away_score"]})
        defense_rows.append({"season": g["season"], "game_id": g["game_id"], "side": "away",
            "own_opp_ppg": g["away_opp_ppg_allowed"], "own_def_ppa": g["away_def_ppa"],
            "opp_own_ppg": g["home_own_ppg"], "opp_off_ppa": g["home_off_ppa"],
            "games_played": g["away_games_played"], "target": g["home_score"]})

    offense_df = pd.DataFrame(offense_rows)
    defense_df = pd.DataFrame(defense_rows)
    off_features = ["own_ppg", "own_off_ppa", "opp_ppg_allowed", "opp_def_ppa", "games_played"]
    def_features = ["own_opp_ppg", "own_def_ppa", "opp_own_ppg", "opp_off_ppa", "games_played"]

    def train_with_oof(df, feature_cols, target_col, test_season, n_folds=6):
        train_mask = df["season"] < test_season
        test_mask_ = df["season"] == test_season
        train_d = df[train_mask].reset_index(drop=True)
        test_d = df[test_mask_].reset_index(drop=True)
        X_tr, y_tr = train_d[feature_cols], train_d[target_col]
        X_te = test_d[feature_cols]
        kf_ = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        oof = np.zeros(len(train_d))
        for fit_idx, hold_idx in kf_.split(X_tr):
            mm = xgb.XGBRegressor(n_estimators=150, max_depth=3, learning_rate=0.05,
                                   subsample=0.8, colsample_bytree=0.8, random_state=42)
            mm.fit(X_tr.iloc[fit_idx], y_tr.iloc[fit_idx])
            oof[hold_idx] = mm.predict(X_tr.iloc[hold_idx])
        train_d = train_d.copy()
        train_d["oof_pred"] = oof
        # train_d/test_d retain "game_id" and "side" columns (not used as model
        # features, but needed downstream to pivot predictions back to one row per game)
        final_m = xgb.XGBRegressor(n_estimators=150, max_depth=3, learning_rate=0.05,
                                    subsample=0.8, colsample_bytree=0.8, random_state=42)
        final_m.fit(X_tr, y_tr)
        test_d = test_d.copy()
        test_d["oof_pred"] = final_m.predict(X_te)
        return train_d, test_d, final_m

    print("Training Gen 3 Offense/Defense models (out-of-fold, 6-fold)...")
    off_train, off_test, off_model = train_with_oof(offense_df, off_features, "target", TEST_SEASON)
    def_train, def_test, def_model = train_with_oof(defense_df, def_features, "target", TEST_SEASON)

    def assemble_gen3(off_df, def_df, source_df):
        off_pivot = off_df.pivot_table(index="game_id", columns="side", values="oof_pred", aggfunc="first")
        off_pivot.columns = [f"off_pred_{c}" for c in off_pivot.columns]
        def_pivot = def_df.pivot_table(index="game_id", columns="side", values="oof_pred", aggfunc="first")
        def_pivot.columns = [f"def_pred_{c}" for c in def_pivot.columns]

        merged = source_df[["game_id", "season", "week", "home", "away", "home_score", "away_score"]].copy()
        merged = merged.merge(off_pivot, on="game_id", how="inner").merge(def_pivot, on="game_id", how="inner")
        merged["gen3_home_score"] = (merged["off_pred_home"] + merged["def_pred_away"]) / 2
        merged["gen3_away_score"] = (merged["off_pred_away"] + merged["def_pred_home"]) / 2
        merged["gen3_pred"] = merged["gen3_home_score"] - merged["gen3_away_score"]
        merged["gen3_total_pred"] = merged["gen3_home_score"] + merged["gen3_away_score"]
        merged["actual_margin"] = merged["home_score"] - merged["away_score"]
        merged["actual_total"] = merged["home_score"] + merged["away_score"]
        return merged

    gen3_train = assemble_gen3(off_train, def_train, rolling_df)
    gen3_test = assemble_gen3(off_test, def_test, rolling_df)

    # -----------------------------------------------------------------
    # Step 5: Fit both metamodels (spread and total)
    # -----------------------------------------------------------------
    print("\nFitting spread metamodel...")
    gen1_slim_tr = gen1_df[gen1_df["season"] < TEST_SEASON][["game_id", "gen1_pred", "actual_margin"]]
    merged_tr = gen1_slim_tr.merge(train_df[["game_id", "gen2_pred"]], on="game_id") \
                             .merge(gen3_train[["game_id", "gen3_pred"]], on="game_id")

    spread_systems = {
        "gen1": merged_tr["gen1_pred"].values,
        "gen2": merged_tr["gen2_pred"].values,
        "gen3": merged_tr["gen3_pred"].values,
    }
    selected_spread, coefs_spread = forward_selection_nnls(spread_systems, merged_tr["actual_margin"])
    print(f"  Selected: {selected_spread}, weights: {coefs_spread.round(3).tolist()}")

    print("\nFitting total metamodel...")
    gen1_total_slim_tr = gen1_df[gen1_df["season"] < TEST_SEASON][["game_id", "gen1_total_pred", "actual_total"]]
    merged_total_tr = gen1_total_slim_tr.merge(train_df[["game_id", "gen2_total_pred"]], on="game_id") \
                                          .merge(gen3_train[["game_id", "gen3_total_pred"]], on="game_id")

    total_systems = {
        "gen1_total": merged_total_tr["gen1_total_pred"].values,
        "gen2_total": merged_total_tr["gen2_total_pred"].values,
        "gen3_total": merged_total_tr["gen3_total_pred"].values,
    }
    selected_total, coefs_total = forward_selection_nnls(total_systems, merged_total_tr["actual_total"])
    print(f"  Selected: {selected_total}, weights: {coefs_total.round(3).tolist()}")

    # -----------------------------------------------------------------
    # Step 6: Save everything
    # -----------------------------------------------------------------
    print("\nSaving models and metadata...")
    joblib.dump(gen2_final_model, f"{MODELS_DIR}/gen2_margin_model.joblib")
    joblib.dump(gen2_total_final_model, f"{MODELS_DIR}/gen2_total_model.joblib")
    joblib.dump(off_model, f"{MODELS_DIR}/gen3_offense_model.joblib")
    joblib.dump(def_model, f"{MODELS_DIR}/gen3_defense_model.joblib")

    metadata = {
        "trained_on_seasons": TRAIN_SEASONS,
        "test_season_used_for_validation": TEST_SEASON,
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "feature_cols_gen2": feature_cols_gen2,
        "adv_feature_names": ADV_FEATURE_NAMES,
        "adv_fields": ADV_FIELDS,
        "off_features": off_features,
        "def_features": def_features,
        "spread_metamodel": {"selected": selected_spread, "weights": coefs_spread.tolist()},
        "total_metamodel": {"selected": selected_total, "weights": coefs_total.tolist()},
        "hfa": HFA,
        "games_to_full_trust_margin": GAMES_TO_FULL_TRUST_MARGIN,
        "assumed_plays_per_game": ASSUMED_PLAYS_PER_GAME,
    }
    with open(f"{MODELS_DIR}/metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # Save prior-season advanced stats too (Stage 2 needs the season BEFORE whatever
    # it's predicting -- if predicting 2026, it needs 2025's advanced stats, which
    # this training run already pulled as part of TRAIN_SEASONS)
    with open(f"{MODELS_DIR}/prior_advanced_lookup.json", "w") as f:
        json.dump(prior_advanced_lookup, f)

    print(f"\nDone. Models and metadata saved to {MODELS_DIR}/")
    print(f"Spread metamodel: {dict(zip(selected_spread, coefs_spread.round(3)))}")
    print(f"Total metamodel: {dict(zip(selected_total, coefs_total.round(3)))}")


if __name__ == "__main__":
    main()
