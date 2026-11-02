# Semantic Audit Toolkit v1

The toolkit maps benchmark-specific identity and witness evidence into one
claim-specific schema. It deliberately separates:

- exact identity or executable-equivalence decisions;
- sound overlap lower bounds that do not exhaust all aliases;
- unresolved rows, which cannot be described as semantically novel.

Run both public adapters in an isolated Python process:

```bash
python3 -I scripts/semantic_audit_v1.py reproduce \
  --output-dir analysis/quality-enhancement-v1/p4
```

Validate existing adapter outputs:

```bash
python3 -I scripts/semantic_audit_v1.py validate \
  analysis/quality-enhancement-v1/p4/tensor_logic.audit.json \
  analysis/quality-enhancement-v1/p4/occam_markov.audit.json
```

The implementation uses only the Python standard library. The Tensor Logic
adapter consumes graph-path identities and executable witnesses. The Occam
adapter consumes exact transition-tensor identities and derives a
pigeonhole-based exposure bound when order assignments are unavailable.

Every output conforms to
`toolkit/schemas/audit-output-v1.schema.json`. The schema requires source
revisions and hashes, train/evaluation identity records, semantic IDs,
witness types, completeness status, bounded overlap counts, behavioral
estimand metadata, and claim wording gates. Every current adapter also emits
an observation contract covering admissible contexts, observable behavior,
termination and errors, effects, nondeterminism, resource observations, and
a prospective must-match/must-differ semantic-unit adequacy rule. When more
than one semantic unit remains defensible, the contract requires separate
sensitivity reporting rather than silent substitution.

`semantic_audit.bounds` implements the P5 certified-exposure interval for
pairwise-disjoint witness groups. It rejects overlapping or incoherent
certificate groups and exposes an assignment checker for executable
falsification tests.
