# Submission 01 Record

Status: **prepared; awaiting owner-controlled Git publication and form filing**  
Prepared: 16 August 2026  
Branch: `abdullah/Q-map-nav`  
Base revision before Day 12: `e217a69211d706387ef6ad5ac1cfa13e964290e4`  
Submitted revision: **to be filled after the owner's manual commit**  
Frozen config: `configs/submission_v1.yaml`

## Verified Before Publication

- Full final-image test suite: 887 passed in 21.48 seconds with networking
  disabled (final source-mounted run: 23.76 seconds).
- Numerical benchmark: 15/15 terminal responses, 13/15 exact, 8/15 strict
  stable; every incorrect or unstable case has a failure category and trace.
- Object-reference quick: 6/6 target selections, markers, and valid protocols.
- Instruction replay: occluded target recovered; oracle and perceived routes
  completed in order; stage order enforced.
- Oracle benchmark: 6/6 quick and 75/75 full structural success.
- Clean Docker build: completed with `--no-cache`.
- Final local image:
  `sha256:571a1e11c6adb7b182f4caaed1a1cf852d5ba30180f979ca7bb56f6d5a990636`.
- Final image size: 7,194,837,842 bytes.
- Network-disabled run: parser, eight-crop YOLOE/MobileCLIP inference, and one
  numerical `Int32` response passed; no outbound-attempt log entries.
- Installed launch: loaded `submission_v1.yaml`, exposed the official numerical
  topic, latched a question, and shut down cleanly.
- Asset inventory: both model assets, colour prototypes, and config present with
  checksums and offline-load evidence.

## Known Weak Areas

- The annotation-backed numerical control is incorrect on two colour cases
  because the released annotations use `maroon` for requested red and `grey` for
  requested black. Production runtime uses canonical pixel-colour probabilities
  and was not weakened to fit those proxy labels.
- Five otherwise exact annotated counts remain conservatively unstable because
  alternative anchor hypotheses produce different qualifying sets. The bounded
  deadline path still publishes the best-supported count instead of silence.
- Detector recall and 2D-to-3D lifting remain the primary real-scene numerical
  risks; the 13/15 result is a reasoning control, not a perception benchmark.
- Hidden GPU size and the official marker scoring implementation remain organiser
  unknowns already isolated behind frozen configuration/output adapters.

## Owner Publication Fields

Fill these only from real evidence:

```text
manual commit SHA:
public repository URL:
anonymous HTTP status (must be 200):
optional Docker Hub image and immutable digest:
submission form filed at (timezone):
form confirmation text or screenshot reference:
```

## Publication State Checked During Preparation

The configured remote was:

```text
git@github.com:abdullah-ukaea/CMU-VLN-Challenge-2026-private.git
```

An anonymous request to its HTTPS URL returned HTTP 404 on 16 August 2026, so it
was not publicly submit-ready at that time. Codex did not stage, commit, push,
change repository visibility, or file the form. Those actions are reserved for
the repository owner.
