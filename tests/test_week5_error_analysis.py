import pandas as pd

from scripts.evaluation.analyze_week5_errors import summarize


def test_confusion_and_rejected_windows_have_separate_denominators():
    frame = pd.DataFrame({"accepted": [True, True, True, True, False],
                          "label": [0, 0, 1, 1, 1], "prediction": [0, 1, 0, 1, 0]})
    result = summarize(frame)
    assert result["coverage"] == .8 and result["accuracy"] == .5
    assert [result[key] for key in ("tn", "fp", "fn", "tp")] == [1, 1, 1, 1]
    assert result["drowsy_recall"] == result["specificity"] == .5


def test_no_accepted_windows_does_not_report_zero_accuracy():
    result = summarize(pd.DataFrame({"accepted": [False], "label": [1], "prediction": [0]}))
    assert result["accuracy"] is None and result["macro_f1"] is None
    assert result["coverage"] == 0
