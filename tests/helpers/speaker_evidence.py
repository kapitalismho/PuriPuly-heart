from dataclasses import replace


def temporal_frames(items):
    result = []
    for item in items:
        if item.frame is None or len(item.source_map) != 1:
            result.append(item)
            continue
        piece = item.source_map[0]
        span = piece.capture
        if span is None or span.normalized_start_sample is None:
            result.append(item)
            continue
        start, end = piece.compact_start_sample, piece.compact_end_sample
        if end - start <= 160 or span.normalized_sample_count != end - start:
            result.append(item)
            continue
        for left in range(start, end, 160):
            right = min(left + 160, end)
            result.append(
                replace(
                    item,
                    frame=replace(item.frame, start_sample=left, end_sample=right),
                    source_map=(
                        replace(
                            piece,
                            compact_start_sample=left,
                            compact_end_sample=right,
                            capture=span.slice_normalized(
                                span.normalized_start_sample + left - start,
                                span.normalized_start_sample + right - start,
                            ),
                        ),
                    ),
                )
            )
    return tuple(result)
