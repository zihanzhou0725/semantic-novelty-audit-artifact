# External Behavioral Validation v2

This is a Mathematics-only recovery amendment to `external-behavior-v1`.
It does not reopen CLUTRR, alter either prior No-Go decision, replace a
candidate, or permit a behavioral claim.

V1 stopped deterministically while generating the next retained
`train/polynomials__expand` row at index 67,937. The pinned upstream code
raised `AssertionError` at
`mathematics_dataset/sample/polynomials.py:107`. No Mathematics train/test
overlap join or aggregate was computed before this amendment.

V2 changes one implementation rule. An `AssertionError` is recoverable only
when its terminal traceback frame exactly matches that pinned file and line
and its message is empty. Each event is logged as a generator assertion drop,
generation continues from the advanced PRNG state, and a stream fails after
1,000 such events. Every other exception remains fatal.

All four streams restart from row zero. The three V1-complete streams must be
byte-identical, and the first 67,937 retained train/expand rows must match the
V1 partial file byte-for-byte. Modules, budgets, seeds, length filters,
parsers, and semantic signatures are unchanged.

This stage materializes and validates rows while keeping overlap aggregates
sealed. A future collision census requires separately frozen analysis inputs.
