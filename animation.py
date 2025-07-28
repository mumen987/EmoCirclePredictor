import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.image as mpimg
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.widgets import Slider
from pathlib import Path

try:
    from unity_sender import send_va_to_unity  # Optional import; fails silently if not present
except ImportError:
    send_va_to_unity = None  # If unity_sender not imported, disable sending

def clamp_to_unit_circle(v_raw: float, a_raw: float):
    """
    If (v_raw, a_raw) is outside the unit circle, normalize it to the circumference:
      r = sqrt(v_raw^2 + a_raw^2)
      v_clamped = v_raw / r
      a_clamped = a_raw / r
    Otherwise, keep unchanged.
    """
    r = np.sqrt(v_raw**2 + a_raw**2)
    if r > 1.0:
        return v_raw / r, a_raw / r
    else:
        return v_raw, a_raw

def animate_with_stimulation_ui(model,
                                init_window_norm: torch.Tensor,
                                scaler_X,
                                scaler_Y,
                                emoji_folder: Path,
                                device,
                                interval=200,
                                window_size=None,
                                send_to_unity=False):
    """
    Animate the emotional circle using BiLSTM predictions, combined with real-time slider for stimulation.
    - model: Trained BiLSTMPredictor
    - init_window_norm: (1, seq_length, 4) normalized initial input window
    - scaler_X, scaler_Y: For normalization/inverse normalization
    - emoji_folder: Path to emoji_q1.png~emoji_q4.png
    - device: 'cpu' or 'cuda'
    - window_size: If specified, trajectory shows only recent window_size frames
    - send_to_unity: If True and unity_sender available, send VA to Unity
    """
    model.to(device)
    model.eval()

    # Preload emoji images and create OffsetImages
    emoji_happy = mpimg.imread(str(emoji_folder / "emoji_q1.png"))  # Quadrant 1
    emoji_angry = mpimg.imread(str(emoji_folder / "emoji_q2.png"))  # Quadrant 2
    emoji_sad = mpimg.imread(str(emoji_folder / "emoji_q3.png"))  # Quadrant 3
    emoji_excited = mpimg.imread(str(emoji_folder / "emoji_q4.png"))  # Quadrant 4

    emoji_images = {
        "excited": OffsetImage(emoji_excited, zoom=0.05),
        "angry": OffsetImage(emoji_angry, zoom=0.05),
        "sad": OffsetImage(emoji_sad, zoom=0.05),
        "happy": OffsetImage(emoji_happy, zoom=0.05),
    }

    def get_emoji_key(a, v):
        """
        Return emoji key based on (a, v) quadrant:
          - v >= 0 and a >= 0 → "excited" (Quadrant 4)
          - v < 0 and a >= 0 → "angry" (Quadrant 2)
          - v < 0 and a < 0 → "sad" (Quadrant 3)
          - v >= 0 and a < 0 → "happy" (Quadrant 1)
        Note: x-axis is valence (v), y-axis is arousal (a).
        """
        if v >= 0 and a >= 0:
            return "excited"
        elif v < 0 and a >= 0:
            return "angry"
        elif v < 0 and a < 0:
            return "sad"
        else:
            return "happy"

    # Current normalized window (1, seq_length, 4)
    current_window = init_window_norm.clone().to(device)

    # Lists to store raw predictions
    valence_list = []
    arousal_list = []
    stimulation_list = []

    # Create figure, leaving space for slider
    fig = plt.figure(figsize=(6, 7))
    ax = fig.add_axes([0.1, 0.25, 0.8, 0.7])  # Main plot on top
    ax_slider = fig.add_axes([0.25, 0.1, 0.5, 0.03])  # Slider at bottom center

    # Draw unit circle and axes
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel("Valence")
    ax.set_ylabel("Arousal")
    ax.set_title("Real-Time Emotion Circle with Model Prediction (Clamped)")

    circle = plt.Circle((0, 0), 1.0, fill=False, color='gray', linestyle='--', linewidth=1.5)
    ax.add_artist(circle)
    ax.axhline(0, color='black', linewidth=1)
    ax.axvline(0, color='black', linewidth=1)

    # Trajectory line and current point
    path_line, = ax.plot([], [], 'b-', linewidth=1)
    point, = ax.plot([], [], 'ro', markersize=5)
    xdata, ydata = [], []  # xdata: valence, ydata: arousal

    # Text for current stimulation value
    text_stim = ax.text(0.95, 0.05, '', transform=ax.transAxes, ha='right', va='bottom')

    # AnnotationBbox for emoji
    default_box = emoji_images["happy"]
    ab = AnnotationBbox(default_box, (0, 0), frameon=False)
    ax.add_artist(ab)

    # Create slider [-1, 1], initial 0
    slider = Slider(ax_slider, 'Stimulation', -1.0, 1.0, valinit=0.0)

    def init():
        path_line.set_data([], [])
        point.set_data([], [])
        xdata.clear()
        ydata.clear()
        text_stim.set_text('')
        ab.offsetbox = emoji_images["happy"]
        ab.xybox = (0, 0)
        # Clear record lists
        valence_list.clear()
        arousal_list.clear()
        stimulation_list.clear()
        return path_line, point, ab, text_stim

    def update(frame):
        nonlocal current_window
        # Get current raw stimulation from slider
        s_raw = slider.val

        # Predict next (a_norm, v_norm, s_norm) with model
        with torch.no_grad():
            out_norm = model(current_window)  # (1, 3)
        out_norm_np = out_norm.cpu().numpy().flatten()  # array([a_norm, v_norm, s_norm])

        # Inverse normalize to raw scale
        a_raw, v_raw, s_pred = scaler_Y.inverse_transform(out_norm_np.reshape(1, -1)).flatten()

        # Clamp if outside unit circle
        v_raw, a_raw = clamp_to_unit_circle(v_raw, a_raw)

        # Override predicted s with slider s_raw; record raw values
        arousal_list.append(a_raw)
        valence_list.append(v_raw)
        stimulation_list.append(s_raw)

        # Optionally send to Unity if enabled and function available
        if send_to_unity and send_va_to_unity is not None:
            send_va_to_unity(v_raw, a_raw)

        # Update trajectory (limit to window_size if set)
        xdata.append(v_raw)
        ydata.append(a_raw)
        if window_size is not None and len(xdata) > window_size:
            xdata.pop(0)
            ydata.pop(0)
        path_line.set_data(xdata, ydata)

        # Update current point
        point.set_data([v_raw], [a_raw])

        # Switch emoji based on quadrant
        key = get_emoji_key(a_raw, v_raw)
        ab.offsetbox = emoji_images[key]
        ab.xybox = (v_raw, a_raw)

        # Update stimulation text
        text_stim.set_text(f"stimulation: {s_raw:.3f}")

        # Build next input feature [delay=0.0, a_raw, v_raw, s_raw]
        raw_next = np.array([[0.0, a_raw, v_raw, s_raw]], dtype=np.float32)  # (1,4)
        norm_next = scaler_X.transform(raw_next)  # (1,4) normalized
        next_norm_tensor = torch.from_numpy(norm_next).to(device).unsqueeze(0)  # (1,1,4)

        # Update window: shift out oldest, append new
        current_window = torch.cat([current_window[:, 1:, :], next_norm_tensor], dim=1)

        return path_line, point, ab, text_stim

    # Start animation
    ani = animation.FuncAnimation(
        fig, update, frames=np.arange(10000),  # Large frame count; stop as needed
        init_func=init, blit=True, interval=interval
    )
    plt.show()
    return ani