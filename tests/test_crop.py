import numpy as np

from gator.agent.blender.crop import crop_camera, crop_window, project


def test_window_holds_mask_and_points_with_margin():
    mask = np.zeros((480, 640), bool)
    mask[100:150, 200:260] = True
    uv = np.array([[190.0, 140.0], [300.0, 160.0]])  # the render reaches past the mask
    u0, v0, u1, v1 = crop_window(mask, uv, 640, 480)
    assert u0 < 190 and v0 < 100 and u1 > 300 and v1 > 160
    assert (u1 - u0) <= (300 - 190 + 1) * 1.2 + 2


def test_window_is_clipped_and_falls_back_to_the_image():
    mask = np.zeros((48, 64), bool)
    mask[0:10, 50:64] = True
    u0, v0, u1, v1 = crop_window(mask, [], 64, 48)
    assert u0 >= 0 and v0 == 0 and u1 == 64 and v1 <= 48
    assert crop_window(np.zeros((48, 64), bool), [], 64, 48) == (0, 0, 64, 48)
    assert crop_window(None, [[np.nan, 1.0]], 64, 48) == (0, 0, 64, 48)


def test_small_object_gets_a_minimum_window():
    mask = np.zeros((480, 640), bool)
    mask[200:203, 300:302] = True
    u0, v0, u1, v1 = crop_window(mask, [], 640, 480, min_size=32)
    assert u1 - u0 >= 32 and v1 - v0 >= 32


def test_crop_camera_maps_pixels_consistently():
    K = np.array([[500.0, 0, 319.5], [0, 500.0, 239.5], [0, 0, 1]])
    w2c = np.eye(4)
    pts = np.random.default_rng(0).uniform([-0.3, -0.2, 2.0], [0.3, 0.2, 3.0], (50, 3))
    uv = project(pts, K, w2c)
    window = (100, 80, 420, 300)
    K2, w, h, s = crop_camera(K, window, 768)
    assert (w, h) == (768, round(220 * 768 / 320)) and s == 768 / 320
    uv2 = project(pts, K2, w2c)
    assert np.allclose(uv2, (uv - [100, 80] + 0.5) * s - 0.5)
    assert len(project(np.array([[0.0, 0.0, -1.0]]), K, w2c)) == 0
