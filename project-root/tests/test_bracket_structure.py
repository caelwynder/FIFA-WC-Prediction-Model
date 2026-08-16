# The website cascades knockout rounds from the r32 array by consecutive
# pairs (slots 2i and 2i+1 feed R16 match i, R16 matches 2j and 2j+1 feed
# QF j, ...), with the lower slot's winner shown as home. That cascade must
# reproduce the real 2026 bracket: the scheduled R16 fixtures in the scraped
# results file, plus the played rounds the file predates (the July 7 R16
# games, quarterfinals and semifinals).

import json

import pandas as pd
import pytest

R16_NOT_IN_RESULTS_FILE = [("Argentina", "Egypt"), ("Switzerland", "Colombia")]
QUARTERFINALS = [("France", "Morocco"), ("Spain", "Belgium"),
                 ("Norway", "England"), ("Argentina", "Switzerland")]
SEMIFINALS = [("France", "Spain"), ("England", "Argentina")]


@pytest.fixture(scope="module")
def r32(base_dir):
    with open(base_dir.parent / "website" / "data.json") as f:
        return json.load(f)["r32"]


@pytest.fixture(scope="module")
def slot_of(r32):
    slots = {}
    for i, fx in enumerate(r32):
        slots[fx["home"]] = i
        slots[fx["away"]] = i
    return slots


def scheduled_r16(base_dir):
    df = pd.read_csv(base_dir / "data" / "raw" / "00_results_final.csv", parse_dates=["date"])
    ko = df[(df["tournament"] == "FIFA World Cup")
            & (df["date"] > pd.Timestamp("2026-07-03"))
            & (df["date"] <= pd.Timestamp("2026-07-08"))]
    return list(ko[["home_team", "away_team"]].itertuples(index=False, name=None))


def test_r16_pairings_and_orientation(base_dir, r32, slot_of):
    fixtures = scheduled_r16(base_dir) + R16_NOT_IN_RESULTS_FILE
    assert len(fixtures) == 8
    for home, away in fixtures:
        h, a = slot_of[home], slot_of[away]
        assert h // 2 == a // 2, f"R16 {home} v {away}: slots {h} and {a} don't meet"
        assert h % 2 == 0, f"R16 {home} v {away}: {home} should be the home side"


def test_quarterfinal_pairings_and_orientation(slot_of):
    for home, away in QUARTERFINALS:
        h, a = slot_of[home], slot_of[away]
        assert h // 4 == a // 4, f"QF {home} v {away}: slots {h} and {a} don't meet"
        assert (h // 2) % 2 == 0, f"QF {home} v {away}: {home} should be the home side"


def test_semifinal_pairings_and_orientation(slot_of):
    for home, away in SEMIFINALS:
        h, a = slot_of[home], slot_of[away]
        assert h // 8 == a // 8, f"SF {home} v {away}: slots {h} and {a} don't meet"
        assert (h // 4) % 2 == 0, f"SF {home} v {away}: {home} should be the home side"
