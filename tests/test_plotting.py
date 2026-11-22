import matplotlib

matplotlib.use("Agg")  # no display needed when running tests

import matplotlib.pyplot as plt

from creditrisk import plotting


def test_save_figure_writes_a_png(tmp_path):
    plotting.apply_style()
    fig, ax = plt.subplots()
    ax.bar(["a", "b"], [1, 2], color=plotting.BLUE)
    path = plotting.save_figure(fig, "demo", figures_dir=tmp_path)
    plt.close(fig)
    assert path == tmp_path / "demo.png"
    assert path.stat().st_size > 0


def test_sequential_scale_goes_from_light_to_dark():
    light = plotting.SEQUENTIAL(0.0)
    dark = plotting.SEQUENTIAL(1.0)
    assert sum(light[:3]) > sum(dark[:3])
