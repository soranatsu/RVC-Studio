import numpy as np
import torch
from tools.cuda_graph import run_cuda_graph
from tqdm import tqdm


def make_padding(width, cropsize, offset):
    left = offset
    roi_size = cropsize - left * 2
    if roi_size == 0:
        roi_size = cropsize
    right = roi_size - (width % roi_size) + left

    return left, right, roi_size


def inference(X_spec, device, model, aggressiveness, data, progress_callback=None):
    """
    data : dic configs
    """

    def _execute(X_mag_pad, roi_size, n_window, device, model, aggressiveness,
                 is_half=True, progress_start=0.0, progress_span=1.0):
        model.eval()
        with torch.no_grad():
            preds = []

            iterations = [n_window]

            total_iterations = sum(iterations)
            for i in tqdm(range(n_window)):
                start = i * roi_size
                X_mag_window = X_mag_pad[None, :, :, start : start + data["window_size"]]
                X_mag_window = torch.from_numpy(X_mag_window)
                if is_half:
                    X_mag_window = X_mag_window.half()
                X_mag_window = X_mag_window.to(device)

                pred = run_cuda_graph(
                    model,
                    "uvr-vr-%s" % repr(aggressiveness),
                    lambda window: model.predict(window, aggressiveness),
                    X_mag_window,
                )

                pred = pred.detach().cpu().numpy()
                preds.append(pred[0])
                if progress_callback is not None:
                    progress_callback(progress_start + progress_span * (i + 1) / n_window)

            pred = np.concatenate(preds, axis=2)
        return pred

    def preprocess(X_spec):
        X_mag = np.abs(X_spec)
        X_phase = np.angle(X_spec)

        return X_mag, X_phase

    X_mag, X_phase = preprocess(X_spec)

    coef = X_mag.max()
    if not np.isfinite(coef):
        raise ValueError("音频包含无效数值")
    if coef <= 1e-12:
        # Avoid 0/0 on silent inputs and skip model execution entirely.
        return np.zeros_like(X_mag), X_mag, np.exp(1.0j * X_phase)
    X_mag_pre = X_mag / coef

    n_frame = X_mag_pre.shape[2]
    pad_l, pad_r, roi_size = make_padding(n_frame, data["window_size"], model.offset)
    n_window = int(np.ceil(n_frame / roi_size))

    X_mag_pad = np.pad(X_mag_pre, ((0, 0), (0, 0), (pad_l, pad_r)), mode="constant")

    if list(model.state_dict().values())[0].dtype == torch.float16:
        is_half = True
    else:
        is_half = False
    pred = _execute(X_mag_pad, roi_size, n_window, device, model, aggressiveness,
                    is_half, 0.0, 0.5 if data["tta"] else 1.0)
    pred = pred[:, :, :n_frame]

    if data["tta"]:
        pad_l += roi_size // 2
        pad_r += roi_size // 2
        n_window += 1

        X_mag_pad = np.pad(X_mag_pre, ((0, 0), (0, 0), (pad_l, pad_r)), mode="constant")

        pred_tta = _execute(X_mag_pad, roi_size, n_window, device, model, aggressiveness,
                            is_half, 0.5, 0.5)
        pred_tta = pred_tta[:, :, roi_size // 2 :]
        pred_tta = pred_tta[:, :, :n_frame]

        return (pred + pred_tta) * 0.5 * coef, X_mag, np.exp(1.0j * X_phase)
    else:
        return pred * coef, X_mag, np.exp(1.0j * X_phase)
