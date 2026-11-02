from __future__ import annotations

import unittest
from types import SimpleNamespace

from toolkit.external_behavior.mathematics import canonical_polynomial
from toolkit.external_behavior_v2.mathematics import (
    RECOVERABLE_LINE,
    RECOVERABLE_RELATIVE_PATH,
    assertion_record,
    sample_retained,
)


REGISTERED_CODE = compile(
    "\n" * (RECOVERABLE_LINE - 1) + "assert False",
    f"/tmp/{RECOVERABLE_RELATIVE_PATH}",
    "exec",
)


def registered_assertion() -> AssertionError:
    try:
        exec(REGISTERED_CODE, {})
    except AssertionError as error:
        return error
    raise AssertionError("Synthetic registered assertion did not fire.")


class RecoverThenReturn:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self) -> SimpleNamespace:
        self.calls += 1
        if self.calls == 1:
            exec(REGISTERED_CODE, {})
        return SimpleNamespace(
            question="Expand (x + 1)*(x - 1).",
            answer="x**2 - 1",
        )


class ExternalBehaviorV2Test(unittest.TestCase):
    def test_registered_assertion_is_logged(self) -> None:
        error = registered_assertion()
        record = assertion_record(
            error,
            regime="train",
            module_name="polynomials__expand",
            next_retained_row_index=67937,
            anomaly_index=0,
        )
        self.assertEqual(
            record["terminal_frame_relative_path"],
            RECOVERABLE_RELATIVE_PATH,
        )
        self.assertEqual(record["terminal_frame_line"], RECOVERABLE_LINE)
        self.assertEqual(record["next_retained_row_index"], 67937)
        self.assertEqual(len(record["traceback_sha256"]), 64)

    def test_unregistered_assertion_remains_fatal(self) -> None:
        try:
            assert False
        except AssertionError as error:
            with self.assertRaises(AssertionError):
                assertion_record(
                    error,
                    regime="train",
                    module_name="polynomials__expand",
                    next_retained_row_index=0,
                    anomaly_index=0,
                )

    def test_sampler_continues_without_reseeding(self) -> None:
        module = RecoverThenReturn()
        problem, length_drops, anomalies = sample_retained(
            module,
            regime="train",
            module_name="polynomials__expand",
            next_retained_row_index=67937,
            anomaly_start_index=0,
        )
        self.assertEqual(module.calls, 2)
        self.assertEqual(length_drops, 0)
        self.assertEqual(len(anomalies), 1)
        self.assertEqual(
            str(problem.question),
            "Expand (x + 1)*(x - 1).",
        )

    def test_semantic_identity_is_unchanged_from_v1(self) -> None:
        first = canonical_polynomial(
            "polynomials__expand",
            "Expand (x + 1)*(x - 1).",
        )
        second = canonical_polynomial(
            "polynomials__expand",
            "Expand (z - 1)*(z + 1).",
        )
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
