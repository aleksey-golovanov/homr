"""Meter changes must survive symbol cleanup and MusicXML export."""

# ruff: noqa: S101

import xml.etree.ElementTree as ET
from fractions import Fraction

import pytest

from homr.music_xml_generator import XmlGeneratorArguments, generate_xml
from homr.transformer.vocabulary import EncodedSymbol, remove_duplicated_symbols


def _bar(
    rhythms: list[str], denominator: int | None, barline: str = "barline"
) -> list[EncodedSymbol]:
    symbols = [EncodedSymbol(f"timeSignature/{denominator}")] if denominator else []
    symbols.extend(
        EncodedSymbol(rhythm, "_" if rhythm.startswith("rest") else "C4", "_", "_", "_", "upper")
        for rhythm in rhythms
    )
    return [*symbols, EncodedSymbol(barline)]


def _export(bars: list[list[EncodedSymbol]]) -> ET.Element:
    symbols = [EncodedSymbol("clef_G2", position="upper")]
    symbols.extend(symbol for bar in bars for symbol in bar)
    return generate_xml(XmlGeneratorArguments(), [remove_duplicated_symbols(symbols)], "")


def _signatures(xml: ET.Element) -> list[tuple[str, str, str]]:
    return [
        (measure.get("number", ""), time.findtext("beats", ""), time.findtext("beat-type", ""))
        for measure in xml.findall("./part/measure")
        for time in measure.findall("./attributes/time")
    ]


def _durations(xml: ET.Element) -> list[Fraction]:
    divisions = 1
    result = []
    for measure in xml.findall("./part/measure"):
        divisions = int(measure.findtext("./attributes/divisions", str(divisions)))
        result.append(
            sum(
                (
                    Fraction(int(note.findtext("duration", "0")), divisions)
                    for note in measure.findall("note")
                    if note.find("chord") is None
                ),
                Fraction(0),
            )
        )
    return result


@pytest.mark.parametrize("quarters", [(4, 2), (2, 4), (4, 2, 4), (3, 5, 3)])
def test_same_denominator_changes(quarters: tuple[int, ...]) -> None:
    xml = _export([_bar(["note_4"] * count, 4) for count in quarters])
    assert _signatures(xml) == [(str(i + 1), str(count), "4") for i, count in enumerate(quarters)]
    assert _durations(xml) == list(quarters)


def test_inference_stops_at_change_and_covers_following_measures() -> None:
    xml = _export([_bar(["note_4"] * 2, 4), _bar(["note_4"] * 4, 4), _bar(["note_4"] * 4, None)])
    assert _signatures(xml) == [("1", "2", "4"), ("2", "4", "4")]
    assert _durations(xml) == [2, 4, 4]


def test_changed_denominator_keeps_equal_measure_length() -> None:
    xml = _export([_bar(["note_4"] * 3, 4), _bar(["note_8"] * 6, 8)])
    assert _signatures(xml) == [("1", "3", "4"), ("2", "6", "8")]
    assert _durations(xml) == [3, 3]


@pytest.mark.parametrize(
    "quarters, beats", [((1, 4, 4), 4), ((4, 4, 1), 4), ((3, 3), 3), ((4, 4), 4)]
)
def test_fixed_meter_with_partial_measures(quarters: tuple[int, ...], beats: int) -> None:
    xml = _export(
        [_bar(["note_4"] * count, 4 if i == 0 else None) for i, count in enumerate(quarters)]
    )
    assert _signatures(xml) == [("1", str(beats), "4")]
    assert _durations(xml) == list(quarters)


def test_single_meter_keeps_existing_estimator_for_disagreeing_durations() -> None:
    xml = _export([_bar(["note_4"] * 4, 4), _bar(["note_4"] * 5, None)])
    assert _signatures(xml) == [("1", "4", "4")]
    assert _durations(xml) == [4, 5]


def test_inferred_opening_does_not_use_last_signature() -> None:
    xml = _export([_bar(["note_4"] * 4, None), _bar(["note_4"] * 2, 4)])
    assert _signatures(xml) == [("1", "4", "4"), ("2", "2", "4")]
    assert _durations(xml) == [4, 2]


@pytest.mark.parametrize(
    "barline", ["barline", "doublebarline", "repeatEnd", "repeatEndStart", "repeatStart"]
)
def test_signature_changes_at_bar_and_repeat_boundaries(barline: str) -> None:
    xml = _export([_bar(["note_4"] * 4, 4, barline), _bar(["note_4"] * 2, 4)])
    assert _signatures(xml) == [("1", "4", "4"), ("2", "2", "4")]
    assert _durations(xml) == [4, 2]


def test_divisions_cover_tuplets_after_signature_change() -> None:
    # Exercise export directly: the independent tuplet-cleanup heuristic can
    # lengthen a short tuplet measure before the exporter receives it.
    symbols = _bar(["note_4"] * 4, 4) + _bar(["note_12"] * 3 + ["note_4"], 4)
    xml = generate_xml(XmlGeneratorArguments(), [symbols], "")
    assert _signatures(xml) == [("1", "4", "4"), ("2", "2", "4")]
    assert _durations(xml) == [4, 2]
    divisions = int(xml.findtext("./part/measure/attributes/divisions", "0"))
    assert divisions % 3 == 0


def test_rests_and_grace_notes_keep_their_duration() -> None:
    xml = _export([_bar(["rest_2", "note_2"], 4), _bar(["note_8G", "rest_4", "note_4"], 4)])
    assert _signatures(xml) == [("1", "4", "4"), ("2", "2", "4")]
    assert _durations(xml) == [4, 2]
    grace = xml.find("./part/measure/note/grace")
    assert grace is not None


def test_opening_meter_is_not_duplicated_after_clef() -> None:
    xml = _export([_bar(["note_8"] * 6, 8)])
    assert _signatures(xml) == [("1", "6", "8")]


def test_empty_signature_sequence_uses_fallback() -> None:
    xml = _export([[EncodedSymbol("timeSignature/4"), EncodedSymbol("timeSignature/4")]])
    assert _signatures(xml) == [("1", "4", "4")]


def test_trailing_marker_does_not_change_preceding_measure() -> None:
    xml = _export([_bar(["note_4"] * 2, 4), [EncodedSymbol("timeSignature/4")]])
    assert _signatures(xml) == [("1", "2", "4"), ("2", "2", "4")]


def test_courtesy_signature_after_notes_does_not_replace_opening() -> None:
    first = _bar(["note_4"] * 4, None)
    first.insert(-1, EncodedSymbol("timeSignature/4"))
    xml = _export([first, _bar(["note_4"] * 2, None)])
    assert _signatures(xml) == [("1", "4", "4"), ("1", "2", "4")]
    assert _durations(xml) == [4, 2]


def test_grand_staff_has_one_opening_signature() -> None:
    symbols = [
        EncodedSymbol("clef_G2", position="upper"),
        EncodedSymbol("chord"),
        EncodedSymbol("clef_F4", position="lower"),
        EncodedSymbol("timeSignature/4"),
    ]
    for _ in range(2):
        symbols.extend(
            [
                EncodedSymbol("note_4", "C4", "_", "_", "_", "upper"),
                EncodedSymbol("chord"),
                EncodedSymbol("note_4", "C3", "_", "_", "_", "lower"),
            ]
        )
    symbols.append(EncodedSymbol("barline"))
    xml = generate_xml(XmlGeneratorArguments(), [remove_duplicated_symbols(symbols)], "")
    assert _signatures(xml) == [("1", "2", "4")]
    assert {note.findtext("staff") for note in xml.findall("./part/measure/note")} == {"1", "2"}
