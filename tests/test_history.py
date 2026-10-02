import numpy as np
import pytest
from PyQt5.QtWidgets import QApplication, QDialog

import AmpMap_GUI
import AmpMap_Widgets
import tools


class FakeFFTTools:
    """FFT Tools dialog where the user applies a low-pass filter and clicks Apply."""

    def __init__(self, original_image, fft_data, parent=None):
        c = fft_data.shape[0] // 2
        self.processed_fft = tools.apply_lowpass_filter_fft(fft_data, c, c, 20)

    def exec_(self):
        return QDialog.Accepted


class FakeTools:
    """Tools dialog where the user applies a Gaussian low-pass and clicks Apply."""

    def __init__(self, image_data, parent=None):
        self.processed_image = tools.apply_gaussian_lowpass(np.asarray(image_data), 3)

    def exec_(self):
        return QDialog.Accepted


# One application and window for the whole run, kept alive until the
# process exits: tearing Qt/matplotlib windows down from Python (between tests
# or during garbage collection) can crash the interpreter, which has nothing
# to do with the code under test.
_KEEP_ALIVE = []


@pytest.fixture(scope="module")
def window():
    if not _KEEP_ALIVE:
        app = QApplication.instance() or QApplication([])
        _KEEP_ALIVE.extend([app, AmpMap_GUI.MainWindow()])
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(AmpMap_Widgets, "FFTToolsDialog", FakeFFTTools)
        mp.setattr(AmpMap_Widgets, "ToolsDialog", FakeTools)
        yield _KEEP_ALIVE


@pytest.fixture
def widget(window):
    """The main widget, reset to the freshly loaded image with empty undo history."""
    w = window[1].myWidget
    w.Z = w.Z_original.copy()
    w.image_history["Input_figure"] = [w.Z.copy()]
    w.history_index["Input_figure"] = 0
    w.image_history["FFT_figure"] = []
    w.history_index["FFT_figure"] = -1
    w._refresh_panel(w.Input_figure, w.Input_Canvas, w.Z, show_axes=True)
    return w


def panel(w, name):
    return getattr(w, name + "_figure"), getattr(w, name + "_Canvas"), name + "_figure"


def shown(fig):
    return np.asarray(fig.axes[0].images[0].get_array())


def test_fft_undo_keeps_real_space_image(widget):
    real = widget.Z.copy()
    fig, canvas, name = panel(widget, "FFT")
    widget._btn_tools(None, fig, canvas, fig_name=name)
    widget._btn_undo(None, fig, canvas, fig_name=name)
    np.testing.assert_allclose(widget.Z, real)
    # the FFT panel shows the unfiltered FFT magnitude again
    np.testing.assert_allclose(shown(fig), np.abs(np.fft.fftshift(np.fft.fft2(real - real.mean()))))


def test_fft_redo_keeps_real_space_image(widget):
    real = widget.Z.copy()
    fig, canvas, name = panel(widget, "FFT")
    widget._btn_tools(None, fig, canvas, fig_name=name)
    filtered_display = shown(fig).copy()
    widget._btn_undo(None, fig, canvas, fig_name=name)
    widget._btn_redo(None, fig, canvas, fig_name=name)
    np.testing.assert_allclose(widget.Z, real)
    np.testing.assert_allclose(shown(fig), filtered_display)


def test_input_undo_redo_still_restores_image(widget):
    real = widget.Z.copy()
    fig, canvas, name = panel(widget, "Input")
    widget._btn_tools(None, fig, canvas, fig_name=name)
    filtered = widget.Z.copy()
    assert not np.allclose(filtered, real)
    widget._btn_undo(None, fig, canvas, fig_name=name)
    np.testing.assert_allclose(widget.Z, real)
    widget._btn_redo(None, fig, canvas, fig_name=name)
    np.testing.assert_allclose(widget.Z, filtered)
