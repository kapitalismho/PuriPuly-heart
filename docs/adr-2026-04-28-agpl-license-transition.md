# ADR: Adopt AGPL-3.0-or-later as the Project License

- Status: Accepted after contributor consent
- Date: 2026-04-28
- Work ref: `agpl-license-transition`

## Context

PuriPuly Heart is currently licensed under MIT. The project includes a consumer app and a broker/network-service component. The project owner wants stronger reciprocal source-availability obligations for modified versions, especially when modified service versions are made available for remote network interaction.

Technical third-party license metadata review found no obvious blocker to relicensing the project's own code to AGPL-3.0-or-later. This is not legal advice. Third-party components remain under their original licenses and must continue to be noticed separately.

Known notice-sensitive components include LGPL-2.1-or-later soxr/libsoxr, MPL-2.0 packages such as certifi/pathspec, PyInstaller's GPL-2.0-or-later WITH Bootloader-exception, OpenVR BSD-3-Clause, SIL OFL fonts, Silero VAD MIT, and official Apache-2.0 Qwen3-ASR upstream assets. Pinned Qwen mirror/converted assets still require provenance verification before release.

An external contributor, RICHARDwuxiaofei, contributed code and i18n changes. The relicensing proceeds after obtaining consent or preserving a documented consent reference.

## Decision

Relicense the repository's own code and project metadata from MIT to AGPL-3.0-or-later.

Use SPDX identifier `AGPL-3.0-or-later` in package metadata where supported.

Do not relicense third-party dependencies or bundled assets. Preserve their original licenses and notices in `src/puripuly_heart/data/THIRD_PARTY_NOTICES.txt` and release compliance materials.

Broker AGPL §13 source-offer handling is satisfied for the public broker by exposing the repository source URL through an API `Link` response header and `GET /source`.

## Consequences

- Downstream distributors and network service operators may need to provide corresponding source for modified versions according to AGPL-3.0-or-later, including its remote-network-interaction terms.
- Previously published MIT versions remain available under MIT; this decision only changes the repository from the transition point forward.
- Binary releases must continue to include AGPL license text plus third-party notices.
- LGPL soxr compliance materials must continue to be packaged.
- Contributor consent evidence must be retained.
- Broker release/deploy readiness must preserve the AGPL §13 source-offer header and endpoint.
