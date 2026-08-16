# Guards against the swapped-orientation merge bug: when the results file and
# the eloratings file disagree on home/away, the elo/rank data columns must be
# flipped along with the team names (00_merge_datasets.py). Before the fix,
# Panama carried England's Elo (2038) and vice versa.

KNOWN_ROWS = [
    # (home, away, date, stronger_side)  — stronger side must carry the higher elo AND better (lower) rank
    ("Panama", "England", "2026-06-27", "away"),
    ("Jordan", "Argentina", "2026-06-27", "away"),
]


def test_known_matches_have_elo_on_the_right_team(merged):
    for home, away, date, stronger in KNOWN_ROWS:
        row = merged[
            (merged["home_team"] == home)
            & (merged["away_team"] == away)
            & (merged["date"] == date)
        ]
        assert len(row) == 1, f"expected exactly one row for {home} vs {away} {date}"
        r = row.iloc[0]
        if stronger == "away":
            assert r["away_elo"] > r["home_elo"], f"{away} should out-rate {home} on {date}"
            assert r["away_rank"] < r["home_rank"], f"{away} should out-rank {home} on {date}"
        else:
            assert r["home_elo"] > r["away_elo"]
            assert r["home_rank"] < r["away_rank"]


def test_england_is_always_strong_in_2026(merged):
    # Orientation-agnostic aggregate: whichever side England appears on in 2026,
    # the Elo attached to England's side must be elite-level.
    recent = merged[merged["date"] >= "2026-01-01"]
    as_home = recent[recent["home_team"] == "England"]["home_elo"]
    as_away = recent[recent["away_team"] == "England"]["away_elo"]
    assert len(as_home) + len(as_away) > 0
    assert (as_home > 1900).all() and (as_away > 1900).all()
