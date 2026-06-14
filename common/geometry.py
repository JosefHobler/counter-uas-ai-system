"""Pure geometry helpers shared across nodes (no cv2 / numpy needed).

These back the cue/confirm pipeline: cropping a candidate region out of a full
frame, mapping a box detected *inside* that crop back to full-frame coordinates,
and IoU-based association between two sets of boxes. Kept dependency-free so the
math stays trivially unit-testable.

Box convention everywhere here is ``(x, y, w, h)`` in pixels, with ``x``/``y``
the top-left corner.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple


def bbox_iou(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    """Intersection-over-union of two ``(x, y, w, h)`` boxes, in ``[0, 1]``."""
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b

    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh

    inter_x1 = max(ax, bx)
    inter_y1 = max(ay, by)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    if inter_area <= 0:
        return 0.0

    area_a = max(1, aw * ah)
    area_b = max(1, bw * bh)
    union = area_a + area_b - inter_area
    return inter_area / union if union > 0 else 0.0


def clamp_crop_region(
    bbox: Sequence[float], frame_w: int, frame_h: int, margin: float = 0.0
) -> Tuple[int, int, int, int]:
    """Return ``(x0, y0, x1, y1)`` crop rectangle in full-frame coordinates.

    ``margin`` is a fraction of the box's longer side added on every side, so the
    heavy model gets some context around the candidate. The result is clamped to
    the frame; a box fully outside the frame yields an empty (zero-area) rect,
    which callers should treat as "nothing to crop".
    """
    x, y, w, h = bbox
    pad = margin * max(w, h)

    x0 = int(min(frame_w, max(0, x - pad)))
    y0 = int(min(frame_h, max(0, y - pad)))
    x1 = int(min(frame_w, max(0, x + w + pad)))
    y1 = int(min(frame_h, max(0, y + h + pad)))

    # Keep the rect non-inverted; an off-frame box stays zero-area on purpose.
    if x1 < x0:
        x1 = x0
    if y1 < y0:
        y1 = y0
    return x0, y0, x1, y1


def map_local_box_to_full(
    local_box: Sequence[float], origin: Sequence[float]
) -> Tuple[float, float, float, float]:
    """Translate a box expressed inside a crop back to full-frame coordinates."""
    lx, ly, lw, lh = local_box
    ox, oy = origin
    return (lx + ox, ly + oy, lw, lh)


def associate_by_iou(
    boxes_a: List[Sequence[float]],
    boxes_b: List[Sequence[float]],
    iou_threshold: float = 0.3,
) -> List[Tuple[int, int, float]]:
    """Greedy one-to-one matching of two box lists by descending IoU.

    Returns ``(index_in_a, index_in_b, iou)`` for each matched pair. This is the
    single-camera association used by cue/confirm: confirmations and edge tracks
    live in the same frame, so plain pixel IoU is the right (and precise) tool,
    not the angular gate used for the separate multi-sender merge.
    """
    candidates = []
    for i, a in enumerate(boxes_a):
        for j, b in enumerate(boxes_b):
            iou = bbox_iou(a, b)
            if iou >= iou_threshold:
                candidates.append((iou, i, j))
    candidates.sort(reverse=True)

    used_a: set = set()
    used_b: set = set()
    matches: List[Tuple[int, int, float]] = []
    for iou, i, j in candidates:
        if i in used_a or j in used_b:
            continue
        used_a.add(i)
        used_b.add(j)
        matches.append((i, j, iou))
    return matches
