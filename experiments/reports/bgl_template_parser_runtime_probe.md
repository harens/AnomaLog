# BGL Template Parser Runtime Probe

This note records a bounded runtime comparison between `Drain3Parser` and
`SpellTemplateParser` on slices of `data/BGL/BGL.log`.

The goal is not to re-run the full benchmark locally. The goal is to see
whether the expensive parser stage is an intrinsic cost of the BGL template
frequency path or whether the parser choice changes the scaling behaviour.

## Method

- Training was run directly against raw BGL lines.
- Prefect asset materialisation was bypassed so the parser cost was measured
  rather than cache behaviour.
- Each parser was trained on the same slice of raw BGL lines.
- Spell artefacts were isolated in a temporary working directory.

## Results

The small 20,000-line slices were close to the noise floor and did not show a
stable separation, so the more useful signal comes from larger head slices.

| Slice | Lines | Drain3 train time | Drain3 templates | Spell train time | Spell templates | Spell / Drain3 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| head 100k | 100,000 | 4.622 s | 24 | 32.820 s | 2,019 | 7.1x |
| head 200k | 200,000 | 9.299 s | 44 | 127.315 s | 3,646 | 13.7x |

The corresponding 20,000-line probes were:

| Slice | Lines | Drain3 train time | Spell train time |
| --- | ---: | ---: | ---: |
| head 20k | 20,000 | 1.070 s | 1.013 s |
| mid 20k | 20,000 | 0.903 s | 1.302 s |
| tail 20k | 20,000 | 1.176 s | 0.853 s |

## Interpretation

- Drain3 stays in the single-digit-second range on the 100k and 200k head
  slices.
- Spell grows much faster than Drain3 once the slice reaches 100k lines.
- The template inventory mined by Spell also expands rapidly on the larger
  slices, which is consistent with the long full-corpus training run seen in
  the cluster log.
- The observed 15-hour wall-time failure is therefore consistent with a
  parser-stage scaling problem, not with the `template_frequency` detector
  itself.

## Practical conclusion

For the BGL 2022 template-frequency benchmark path, the expensive Spell stage
is the wrong place to spend runtime. A Drain3-backed template stage is the
bounded, reproducible choice in this repository’s current BGL contract.
