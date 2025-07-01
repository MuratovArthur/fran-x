def snap_boundaries(span, raw_text):
    """Heuristic post-processor that fixes common off-by-one or punctuation issues.

    1. Extends span to the right if the next character is alphanumeric (fixes CO→COP).
    2. Trims leading/trailing whitespace, quotes and punctuation (e.g. " Biden's " → "Biden").
    """
    start, end = span["start"], span["end"]

    # ------------------------- extend right -------------------------
    while end < len(raw_text) and raw_text[end].isalnum():
        end += 1

    # ------------------------- trim left ----------------------------
    # Include whitespace in left trimming
    left_punct = "'\\\"""''\t\n\r "
    while start < end and raw_text[start] in left_punct:
        start += 1

    # ------------------------- trim right ---------------------------
    # Include whitespace in right trimming
    right_punct = "'\\\".,;:!?""''\t\n\r "
    while start < end and raw_text[end - 1] in right_punct:
        end -= 1

    # ------------------------- final whitespace trim ---------------
    # Additional comprehensive whitespace trimming
    while start < end and raw_text[start].isspace():
        start += 1
    while start < end and raw_text[end - 1].isspace():
        end -= 1

    # update span
    span["start"], span["end"] = start, end
    span["word"] = raw_text[start:end].replace("\n", " ").replace("\t", " ")
    return span 