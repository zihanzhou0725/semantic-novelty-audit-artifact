# Directional External Behavioral Validation V4

V4 is a prospective directional confirmation that treats the completed V3
study as an explicitly declared pilot. V4 target semantic patterns must be
fully disjoint from V3 targets, and every model is trained from scratch.

The power stage considers one through four V3-sized blocks. Each block contains
20 same-label pattern pairs, eight model seeds, two complementary arms, and 16
runs. This preserves the V3 exposure dose, common-base size, and 24,000-row
training budget instead of creating one high-exposure training mixture.

The design effect is the smaller of half the V3 point estimate and the V3
registered lower confidence endpoint. The frozen gate chooses the smallest
block count whose simulated power and 95% Wilson lower endpoint are both at
least 90 percent.

Passing this design gate freezes pair and block manifests only. It does not
release training; a separate materialization gate must validate all rows,
source-group separation, arm balance, tensor lengths, and run manifests.
