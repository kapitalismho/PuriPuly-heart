# Peer captions

The selected policy is F′P with RP-B′ display, from [#278's owner decision](https://github.com/kapitalismho/PuriPuly-heart/issues/278#issuecomment-6580384532), implemented under [#356](https://github.com/kapitalismho/PuriPuly-heart/issues/356). That issue retains the exact word lists, rules and acceptance criteria. This document describes the product contract, not a new policy comparison.

## Route boundaries

| Behavior | Soniox, Deepgram, Gemini Transcribe, Qwen Audio | Local Peer |
| --- | --- | --- |
| Audio delivery | Pause opportunities at 4 s / 192 ms and 5 s / 192 ms; hard cut at 10 s | Unchanged, including the existing 6 s hard cut |
| Preview | First check 5 s after the first nonempty partial; subsequent checks every 4 s | None |
| Adjacent caption joining | R0 text-shape or R1 cut-signal rule | None |
| Display | Two-line pages and protected upper slot | Same display policy |

Other ASR families retain their existing recognition and translation policy. Self and manual text retain their own translation lifecycle. Peer output remains excluded from the VRChat chatbox. Deepgram's finalize bundle is one final; native `is_final` fragments do not introduce caption boundaries.

## Source, requests and history

A caption unit and its translation versions are distinct from an audio segment or provider task. Explicit unit/version identity survives request admission and output projection. A second valid version is not deduplicated merely because a translation for the unit was already shown. Older completions cannot overwrite newer visible versions or resurrect retired captions.

Existing final language/speaker spans remain separate source children with their original attribution and source ranges. The first child settles an existing segment preview; subsequent children retain source order and independent failure/timeout recovery. A child boundary without an independently observed first partial has an unknown gap and therefore does not join.

The first nonempty partial after a final starts the next segment's preview clock. Each check uses the latest cumulative partial. Qwen alone retains its last nonempty partial when it temporarily clears its text; an empty replacement from another provider retracts the pending source. Select the last sentence/clause boundary whose prefix covers at least half of the partial, or the whole partial if there is no such boundary. Refresh only when both the partial and selected source grew relative to the last requested preview.

Each open segment has at most one running preview and one newest pending source. A final supersedes the preview and does not wait for its timer. Completed preview translations replace the same caption, rather than appending independent speech. Previews never enter committed conversation history or translation context. Request diagnostics distinguish preview, final, repair and continuation so request amplification remains observable.

Joining considers only the immediately preceding caption. The gap is measured from its last final to the first nonempty partial of the new segment. An unknown gap, a gap over 1.5 s or a reply-only source stops both rules.

- **R0:** open source punctuation, an unfinished sentence, the specified English/Korean ending words, or a lowercase English continuation.
- **R1:** a hard cut, a 4/5-second cut with an actual pause below 0.5 s, a Smart Turn incomplete judgment, or an open mark. Gemini uses only the open-mark signal. Question/exclamation endings stop R1. R1 alone also stops when the estimate exceeds four translated lines.
- **Merge at two estimated lines or fewer:** retranslate the combined source as a normal repair of the preceding caption, with that caption's original context. Commit the combined source once. Retire any absorbed preview only when the repair is displayed.
- **Continue above two lines:** translate only the new final in its own caption with the continuation instruction. Keep the previous caption unchanged and commit the new source once. This applies to either joining rule; an R1-only join refused by the four-line stop stays independent with the normal prompt.

A merged unit carries the cumulative identities of superseded units across later repairs. A continuation is an ordinary caption for the next boundary: a later small merge uses the normal prompt, while a later continuation uses the continuation prompt again.

Translation context and GUI conversation history retain one current record per caption unit. Source and translation components of the same accepted revision converge on that record, regardless of which delivery path arrives first. A newer source-only settlement clears the old translation. Persisted conversation logs remain append-only: caption records identify their stable unit, revision and `commit` or `replace` action, so a repair is an explicit revision rather than another accepted speech contribution. Stale and duplicate versions are rejected; genuinely repeated speech has a distinct unit identity.

Joining does not add speaker inference or a new speaker-change stop. A repair combining distinct or ambiguous speaker assignments uses the existing `mixed` semantic attribution instead of claiming a single speaker or obsolete single-final source range. Each accepted version updates the overlay's retained semantic assignment; identity-dependent decisions such as the overflow-speaker divider use that current assignment. The overlay retains its existing immutable first-readable style; that retained style is not new evidence that every word in a repaired caption came from one speaker.

The source-layout estimate shares the product's caption width basis and effective text-scale calculation, using the current destination calibration when the join is decided. Retained experiment ratios are 0.856 for English-to-Korean and 1.617 for Korean-to-English; same-language estimates use 1.0. Other directions use the historical 1.617 fallback. That fallback is a conservative estimate, not measured pair-specific expansion. The estimate is not a guarantee that the completed translation occupies the predicted number of lines.

## Context and editable prompts

Context retains the existing format: at most five recent Peer finals within 60 seconds, rendered as `- [peer] "…"`. Normal final/repair requests reuse the context captured at the caption's first request, including its first preview. A continuation uses current context with the preceding caption's complete source as the last entry, replacing the immediately preceding final entry when appropriate and keeping the five-entry bound.

Both [standard](../prompts/translation_prompt.md) and [Luna](../prompts/translation_prompt_luna.md) templates use these variables:

| Variable | Rendering |
| --- | --- |
| `${completeSourceOnly}` … `${endCompleteSourceOnly}` | Keeps the enclosed Ellipsis line except for a Peer preview |
| `${peerProvisionalSection}` | Appends [the provisional instruction](../prompts/peer_provisional_instruction.md) for a Peer preview only |
| `${peerContinuationSection}` | Appends [the continuation instruction](../prompts/peer_continuation_instruction.md) for a continuation only |

No request renders both instructions. Empty conditional sections leave no heading, marker or extra blank line. Ordinary Self, local Peer, final and repair rendering remains unchanged. Instruction prose is editable in the two files rather than assembled through string surgery in application code.

Prompt files use the existing process cache; restart the application to load edits. A custom prompt root replaces the bundled root, not overlays it. A custom template opting into a request-specific section requires the corresponding nonempty instruction file when that request is rendered. Missing files raise `FileNotFoundError`; empty files raise `ValueError`. Ordinary requests and legacy custom templates without those variables remain supported.

## Two-block display

Each Peer block displays up to two translation lines. Longer translations advance through two-line pages every 3 seconds. Every new preview, final or repair version restarts at its first page; final and repair updates to visible captions apply immediately.

A failed cloud final or repair settles its version as source-only: the authoritative source replaces any obsolete preview translation, remains readable even when secondary originals are hidden, and uses the same paging rules. Absorbed preview identities are retired on that settlement. An authoritative empty settlement retires the provisional caption rather than leaving withdrawn text visible.

A new caption enters the lower slot and promotes its predecessor to the upper slot. Upper-slot protection lasts until the later of:

- the current page's last actual visible change plus its reading budget;
- promotion time plus 2 seconds.

The reading budget is 0.3 seconds plus 1/9 second per wide character and 1/17 second per narrow character of changed text, clamped to 1–3 seconds. Corrections and page turns recompute protection. A new visible occurrence starts a new clock; an older scheduled expiry cannot remove it early. Identical visible text does not manufacture another visible change.

A new Peer caption waits at most 1 second for a protected upper slot, then evicts it. A superseding version retains that caption's first admission-attempt time instead of restarting W. Rare unread-page loss at this bound is an accepted policy limit. Same-caption updates do not wait behind new-caption admission, including when an absorbed preview is already waiting in the output queue. Self admission bypasses Peer protection and waiting; Peer cannot delay or hide Self.

Python owns these page and lifetime decisions. Desktop and native renderers follow the ordered scene and retain occupant identity through promotion. Renderer fitting/clipping remains separate from Python pagination. Native acknowledgments and state tests do not establish physical headset visibility.

## Evidence limits

The historical 424-cell results—similar median first display, about 8% of cells over 0.5 seconds later, and about 27% additional translation requests—came from a replay harness, not this application's runtime. They are sanity references, not thresholds or claims of measured application performance. A synthetic smoke is not a repeat of that corpus, a cloud-provider qualification or an HMD readability study.
