"""Descriptor de apariencia por histograma de color (SPIKE-003: mejor AUC que ResNet18 y OSNet en A2/K2)."""

import numpy as np

from volley_cv.appearance import ColorHistEmbedder


def frame_with(colors: list[tuple[int, int, int]]) -> np.ndarray:
    f = np.zeros((200, 400, 3), dtype=np.uint8)
    for k, c in enumerate(colors):
        f[20:180, 20 + 120 * k : 100 + 120 * k] = c
        f[100:180, 20 + 120 * k : 100 + 120 * k] = (c[0] // 3, c[1] // 3, c[2] // 3)  # "pantalón" más oscuro
    return f


def dist(a: np.ndarray, b: np.ndarray) -> float:
    return 1.0 - float(a @ b)


def test_same_person_is_close_and_different_colors_are_far() -> None:
    f = frame_with([(200, 200, 200), (30, 30, 200), (200, 200, 200)])
    boxes = [(20.0, 20.0, 100.0, 180.0), (140.0, 20.0, 220.0, 180.0), (260.0, 20.0, 340.0, 180.0)]
    e = ColorHistEmbedder().embed(f, boxes)
    assert all(v is not None for v in e)
    a, b, c = (np.asarray(v) for v in e)
    assert abs(float(np.linalg.norm(a)) - 1.0) < 1e-5  # normalizado
    assert dist(a, c) < 0.01  # misma ropa
    assert dist(a, b) > 0.5  # otra ropa


def test_empty_crop_gives_none() -> None:
    f = frame_with([(200, 200, 200)])
    assert ColorHistEmbedder().embed(f, [(500.0, 10.0, 520.0, 50.0)]) == [None]


def test_small_shift_of_box_keeps_descriptor_close() -> None:
    f = frame_with([(30, 120, 220)])
    e = ColorHistEmbedder()
    (a,) = e.embed(f, [(20.0, 20.0, 100.0, 180.0)])
    (b,) = e.embed(f, [(24.0, 24.0, 104.0, 184.0)])
    assert a is not None and b is not None
    assert dist(np.asarray(a), np.asarray(b)) < 0.1
