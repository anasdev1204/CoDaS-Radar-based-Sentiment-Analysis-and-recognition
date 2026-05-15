import numpy as np
from skimage.transform import resize
from pyts.image import RecurrencePlot, GramianAngularField, MarkovTransitionField
from scipy.signal import spectrogram
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from io import BytesIO
from PIL import Image
from scipy.stats import skew, kurtosis


def _safe_skew_kurtosis(sig: np.ndarray, eps: float = 1e-8):
    sig = np.asarray(sig, dtype=np.float64)
    sig = np.nan_to_num(sig, nan=0.0, posinf=0.0, neginf=0.0)
    if np.std(sig) < eps:
        return 0.0, 0.0
    try:
        return float(skew(sig, bias=False)), float(kurtosis(sig, bias=False))
    except TypeError:
        return float(skew(sig)), float(kurtosis(sig))

def _resize(img, out_size):
    img = np.asarray(img, dtype=np.float32)
    if img.ndim != 2:
        raise ValueError(f"_resize expects 2D image, got shape={img.shape}")

    try:
        out_size = int(out_size)
    except (TypeError, ValueError):
        out_size = 128

    if out_size <= 0:
        out_size = img.shape[0] if img.shape[0] > 0 else 600

    if img.shape[0] == 0 or img.shape[1] == 0:
        return np.zeros((out_size, out_size), dtype=np.float32)

    return resize(img, (out_size, out_size), anti_aliasing=True).astype(np.float32)


def generate_LinePlot(window, image_size=128, resize_flag=True):
    T, C = window.shape
    images = []

    for ch in range(C):
        fig, ax = plt.subplots(figsize=(2, 2), dpi=image_size // 2)
        ax.plot(window[:, ch], color='black', linewidth=1)
        ax.axis('off')
        fig.tight_layout(pad=0)

        canvas = FigureCanvas(fig)
        buf = BytesIO()
        canvas.print_png(buf)
        buf.seek(0)
        img = Image.open(buf).convert("L")  # grayscale
        img = img.resize((image_size, image_size), Image.BILINEAR) if resize_flag else img

        images.append(np.array(img, dtype=np.float32) / 255.0)
        plt.close(fig)

    return np.stack(images, axis=0)  # (C, H, W)


def generate_RP(window, image_size=128, threshold="point", percentage=20, resize_flag=True):
    """
    window: (T, C)
    output: (C, H, W)  OR  (C, T, T) if resize_flag=False
    """
    rp = RecurrencePlot(threshold=threshold, percentage=percentage)

    T, C = window.shape
    images = []

    for ch in range(C):
        sig = window[:, ch].reshape(1, -1)
        rp_img = rp.fit_transform(sig)[0]   # (T, T)

        if resize_flag:
            rp_img = _resize(rp_img, image_size)

        images.append(rp_img.astype(np.float32))

    return np.stack(images, axis=0)


def generate_GAF(window, image_size=128, method="summation", resize_flag=True):
    """
    window: (T, C)
    output: (C, H, W)  OR raw-size output if resize_flag=False
    """
    gaf = GramianAngularField(image_size=image_size, method=method)

    T, C = window.shape
    images = []

    for ch in range(C):
        sig = window[:, ch]
        img = gaf.fit_transform(sig.reshape(1, -1))[0]  # usually image_size × image_size

        if resize_flag and img.shape[0] != image_size:
            img = _resize(img, image_size)

        images.append(img.astype(np.float32))

    return np.stack(images, axis=0)


def generate_Spectrogram(
    window,
    image_size=128,
    fs=100,
    resize_flag=True,
    nperseg=64,
    noverlap=32,
    nfft=128
):
    T, C = window.shape
    images = []

    for ch in range(C):
        sig = window[:, ch]

        f, t, Sxx = spectrogram(
            sig,
            fs=fs,
            nperseg=nperseg,
            noverlap=noverlap,
            nfft=nfft,
        )

        # log-scale
        Sxx = np.log1p(Sxx)

        if resize_flag:
            Sxx = _resize(Sxx, image_size)

        images.append(Sxx.astype(np.float32))

    return np.stack(images, axis=0)


def generate_MTF(window, image_size=128, n_bins=8, resize_flag=True):
    """
    window: (T, C)
    output: (C, H, W)  OR raw-size if resize_flag=False
    """
    mtf = MarkovTransitionField(n_bins=n_bins)

    T, C = window.shape
    images = []

    for ch in range(C):
        sig = window[:, ch].reshape(1, -1)
        img = mtf.fit_transform(sig)[0]    # (T, T)

        if resize_flag:
            img = _resize(img, image_size)

        images.append(img.astype(np.float32))

    return np.stack(images, axis=0)

def generate_features(window, image_size=128, resize_flag=True):
    T, C = window.shape
    features = []

    for ch in range(C):
        sig = np.nan_to_num(window[:, ch], nan=0.0, posinf=0.0, neginf=0.0)
        s, k = _safe_skew_kurtosis(sig)
        features.extend([
            np.mean(sig),
            np.std(sig),
            np.min(sig),
            np.max(sig),
            s,
            k,
        ])

    return np.array(features, dtype=np.float32)

METHODS = {
    "RP": generate_RP,
    "GAF": generate_GAF,
    "MTF": generate_MTF,
    "SPEC": generate_Spectrogram,
    "FEAT": generate_features,
}
