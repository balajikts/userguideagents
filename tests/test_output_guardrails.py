from app.guardrails.output import BLOCKED_MESSAGE, apply_output_guardrails, check_grounding, safety_warnings
from app.schemas import DeviceQuery, GuideAnswer, GuideStep, Source, SourceOrigin

Q = DeviceQuery(brand="Sony", model="WH-1000XM5", device_type="headphones", question="factory reset", confidence=0.9)
S1 = Source(id="S1", title="Initializing the headset", url="https://helpguide.sony.net/r", origin=SourceOrigin.MANUAL_STORE,
            snippet="Connect the headset to a USB AC adaptor. Press and hold the power button and the custom button "
                    "simultaneously for 7 seconds or more. The indicator flashes blue 4 times and settings are reset.")
S2 = Source(id="S2", title="Charging", url="https://sony.com/c", origin=SourceOrigin.WEB,
            snippet="Charge the headset with the supplied USB Type-C cable. Charging takes about 3.5 hours.")


def answer(*steps, warnings=None, question="factory reset"):
    return GuideAnswer(
        status="answered", summary="Resetting restores defaults.",
        steps=[GuideStep(number=i, instruction=t, citation=c) for i, (t, c) in enumerate(steps, 1)],
        warnings=warnings or [], sources=[S1, S2], query=Q.model_copy(update={"question": question}),
    )


def test_grounded_answer_passes_and_gets_reset_warning():
    a = answer(("Connect the headset to a USB AC adaptor.", "S1"),
               ("Press and hold the power and custom buttons for 7 seconds.", "S1"))
    out, report = apply_output_guardrails(a)
    assert report.passed and out.status == "answered"
    assert all(s.verified for s in out.steps)
    assert any("Back up" in w for w in out.warnings)


def test_unknown_citation_fails():
    out, report = apply_output_guardrails(answer(("Hold the power button for 7 seconds.", "S9")))
    assert not report.passed and out.status == "not_found" and not out.steps
    assert out.sources  # still offer the links


def test_number_not_in_source_fails():
    # Source says 7 seconds; a hallucinated 15 seconds must not get through.
    out, report = apply_output_guardrails(answer(("Press and hold the power and custom buttons for 15 seconds.", "S1")))
    assert not report.passed and out.status == "not_found"


def test_citing_wrong_source_for_number_fails():
    r = check_grounding(answer(("Hold the power button for 7 seconds.", "S2")))
    assert not r.passed


def test_paraphrase_kept_but_flagged():
    a = answer(("Connect the headset to a USB AC adaptor.", "S1"),
               ("Press and hold power and custom buttons for 7 seconds.", "S1"),
               ("The indicator flashes blue 4 times.", "S1"),
               ("Wait patiently, then celebrate.", "S1"))
    out, report = apply_output_guardrails(a)
    assert out.status == "answered" and report.verified_ratio == 0.75
    assert [s.verified for s in out.steps] == [True, True, True, False]


def test_too_many_unverified_steps_fails():
    a = answer(("Wiggle the left earcup.", "S1"), ("Sing loudly to the device.", "S1"),
               ("Press the power button.", "S1"))
    out, _ = apply_output_guardrails(a)
    assert out.status == "not_found"


def test_existing_warning_not_duplicated():
    a = answer(("Press and hold the power and custom buttons for 7 seconds.", "S1"),
               warnings=["This erases pairing data; back up first."])
    assert safety_warnings(a) == []


def test_battery_and_case_warnings():
    a = answer(("Remove the back cover and replace the battery.", "S1"), question="replace battery")
    w = " ".join(safety_warnings(a))
    assert "Unplug" in w and "catch fire" in w


def test_firmware_update_warning():
    a = answer(("Open the Sound Connect app and start the update.", "S1"), question="update firmware")
    assert any("update is installing" in w for w in safety_warnings(a))


def test_dangerous_procedure_blocked():
    a = answer(("Discharge the high-voltage capacitor with a screwdriver.", "S1"), question="repair microwave magnetron")
    out, _ = apply_output_guardrails(a)
    assert out.status == "rejected" and out.summary == BLOCKED_MESSAGE and not out.steps


def test_non_answered_passthrough():
    a = GuideAnswer(status="needs_clarification", clarification_question="Which model?")
    out, report = apply_output_guardrails(a)
    assert out is a and report.passed


def test_number_match_is_whole_number_not_substring():
    # "1" must not be satisfied by the "4" or "7"... nor "15" by "1"; compare whole numbers.
    r = check_grounding(answer(("Hold the power button for 1 second.", "S1")))
    assert not r.passed
