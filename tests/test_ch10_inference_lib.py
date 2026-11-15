"""分数 → 过线标签的唯一实现:兜底与 sigmoid 都只此一处。"""
import numpy as np

from scripts.ch10.inference_lib import apply_threshold, sigmoid


def test_apply_threshold_marks_labels_at_or_above_line():
    probs = np.array([[0.9, 0.1, 0.4]])
    assert apply_threshold(probs, 0.4).tolist() == [[1, 0, 1]]


def test_apply_threshold_falls_back_to_argmax_when_none_pass():
    # 全不过线:取最高分那一类,绝不产出空标签
    assert apply_threshold(np.array([[0.2, 0.3, 0.25]]), 0.9).tolist() == [[0, 1, 0]]


def test_apply_threshold_is_per_row():
    probs = np.array([[0.9, 0.1, 0.1], [0.1, 0.1, 0.2]])
    assert apply_threshold(probs, 0.5).tolist() == [[1, 0, 0], [0, 0, 1]]


def test_sigmoid_maps_logits_to_probabilities():
    out = sigmoid(np.array([0.0, 1.0, -1.0]))
    assert np.allclose(out, [0.5, 0.73105858, 0.26894142])


def test_sigmoid_is_monotone_and_bounded():
    out = sigmoid(np.array([-50.0, 0.0, 50.0]))
    assert out[0] < out[1] < out[2]
    assert 0.0 <= out.min() and out.max() <= 1.0
