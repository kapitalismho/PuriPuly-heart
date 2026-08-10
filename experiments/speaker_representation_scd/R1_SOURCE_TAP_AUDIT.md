# R1 ERes2NetV2 Source and Tap Audit

## Verdict

The frozen ModelScope checkpoint can be evaluated with the official 3D-Speaker
ERes2NetV2 graph, but the relationship is an exact-artifact compatibility chain rather than a
same-repository source pin. R1 therefore treats extraction validity as unproven until the captured
`fuse_out34` tensor reconstructs the official embedding on deterministic fixtures.

## Checkpoint identity

- ModelScope repository: `iic/speech_eres2netv2_sv_zh-cn_16k-common`
- Frozen R0 commit: `1cf80d41fb3435bd3d8df185b5c423333b2db42a`
- Frozen tag: `v1.0.0`
- File: `pretrained_eres2netv2.ckpt`
- LFS object and file SHA-256: `0eb4057106b2573dd7b132cf0c36273ab29afd192c1610f80baa9c556dbb963c`
- Size: `71,768,231` bytes

The current official 3D-Speaker inference example names ModelScope tag `v1.0.1`, whereas R0 froze
`v1.0.0`. This does not justify silently changing the checkpoint: the same LFS object occurs at the
peeled commits for `v1.0.0`, `v1.0.1`, and `v1.0.2`:

```text
1cf80d41fb3435bd3d8df185b5c423333b2db42a
c0df10ae7e0dec76f922b2cd2dcef25f92225f09
cdcc197880394b3e1955a0e3ed42702be961d249
```

The checkpoint bytes therefore remain the R0-frozen bytes. R1 does not inherit the later tag's
threshold or configuration values.

## Official source identity

- Repository: `https://github.com/modelscope/3D-Speaker.git`
- Revision: `707eef4eb9b95fd4a9886776df0022390049a5a6`
- Commit purpose: the official inference path that added ERes2NetV2 support
- License: Apache-2.0, source license SHA-256
  `c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4`

At this revision, the ERes2NetV2 model file is byte-identical to its initial
`5667fe634162e43056d03eeea9c34966c4dca88f` introduction. The later generalized architecture
arguments are not needed to load the standard checkpoint. Every imported source file is pinned in
`models/source_registry.json` and is verified before import.

## Official frontend

The official inference path uses:

```text
FBank(80, sample_rate=16000, mean_nor=True)
dither=0
```

The source delegates framing to `torchaudio.compliance.kaldi.fbank`. R1 freezes a 400-sample frame,
160-sample shift, and `snip_edges=true`, but retains `frontend_parity_status` as unresolved until
the locked environment produces the empirical frame-count and timestamp fixtures.

## Tap graph

The official forward graph is:

```text
FBank
→ conv1
→ layer1 = out1
→ layer2 = out2
→ layer3 = out3
→ layer4 = out4
                    ┐
out3 → layer3_ds ───┴→ fuse34 = fuse_out34
                         → TSTP
                         → seg_1
                         → official 192-d embedding
```

`fuse_out34` is exactly the tensor passed to the official temporal statistics pool. It is therefore
the primary `FUSED` pre-pooling tap. Stage taps are retained as development candidates because the
final fused map has a much larger temporal receptive field.

| Tap | Official tensor | Shape before time pooling | Temporal stride | Maximum convolutional receptive field |
| --- | --- | --- | ---: | ---: |
| `S1` | `out1` | `B × 128 × 80 × T1` | 10 ms | 165 ms |
| `S2` | `out2` | `B × 256 × 40 × T2` | 20 ms | 485 ms |
| `S3` | `out3` | `B × 512 × 20 × T3` | 40 ms | 1,445 ms |
| `S4` | `out4` | `B × 1024 × 10 × T4` | 80 ms | 2,405 ms |
| `FUSED` | `fuse_out34` | `B × 1024 × 10 × T4` | 80 ms | 2,405 ms |

The receptive-field values are maximum paths through the padded convolution graph, expressed on
the official FBank timeline. Near a short-window edge much of that theoretical field is padding,
and every tap remains bounded by the supplied trailing waveform. Empirical impulse/timestamp tests
remain mandatory before any localization claim.

The receptive-field table is not a post-context frame-localization guarantee. SSL self-attention
and any ERes within-window feature mixing can spread a localized source mutation across output
indices. R1 therefore records empirical changed-index spans for multiple source coordinates at
every selected SSL layer and ERes tap, while assigning the pooled representation only the
window-end availability frontier.

Each tap has `10,240` channel-frequency values per output time step. R3 flattens channel and
frequency for each time step, mean-pools only valid time steps, then L2-normalizes. It does not
apply the official TSTP or 192-d projection to the pre-pooling representation condition.

## Short-window caveat

A 100 ms waveform yields eight official FBank frames and one `FUSED` time step. Official TSTP uses
the default sample variance, which is nonfinite for one time step. The 192-d final embedding is
therefore expected to be structurally unavailable at this duration unless the official behavior is
changed, which R1 does not permit.

The pre-pooling tensor itself can still be finite and evaluable at 100 ms. R1 records the final
embedding as structured short-input missingness and does not impute or alter the variance formula.

## Required parity evidence

For every fixture with at least two `FUSED` frames:

```text
official model forward
vs
captured FUSED → official TSTP → official seg_1
```

must agree within `1e-6` maximum absolute error. The smoke also requires strict state-dictionary
loading, deterministic repeated extraction, batch/single agreement, future-mutation invariance,
finite pre-pooling features, and empirical output-length agreement. Failure marks the ERes
pre-pooling condition `not_evaluable`; it is not scored as a performance loss.
