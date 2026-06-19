import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import os

st.set_page_config(page_title="Letter Recognizer", layout="centered")

# ── Change this to the path of your .pth file ────────────────────────────────
MODEL_PATH = "letter_recognizer_final.pth"
# ─────────────────────────────────────────────────────────────────────────────

LABELS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
MEAN, STD = 0.1307, 0.3081


class LetterNet(nn.Module):
    def __init__(self, num_classes=26):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.1),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.1),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(3),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 9, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    model = LetterNet(26)
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()
    return model


def preprocess(img_array):
    """
    img_array: RGBA numpy array (H x W x 4) from st_canvas.
    The canvas draws white strokes on black background,
    so pixel brightness lives in the RGB channels — not alpha.
    """
    # Convert RGB → grayscale (ignore alpha, it's unreliable)
    rgb = img_array[:, :, :3].astype(np.float32)
    gray = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]

    if gray.max() < 10:          # blank canvas
        return None, None

    img = Image.fromarray(gray.astype(np.uint8), mode="L")
    img = img.resize((28, 28), Image.LANCZOS)

    arr = np.array(img).astype(np.float32) / 255.0
    arr = (arr - MEAN) / STD

    tensor = torch.tensor(arr).unsqueeze(0).unsqueeze(0)   # (1,1,28,28)

    # Preview: denormalise back to 0-255 for display
    preview_arr = np.clip(arr * STD + MEAN, 0, 1)
    preview = Image.fromarray((preview_arr * 255).astype(np.uint8))
    return tensor, preview


@torch.no_grad()
def predict(model, tensor):
    logits = model(tensor)
    probs = F.softmax(logits, dim=1).squeeze().tolist()
    return sorted(zip(LABELS, probs), key=lambda x: x[1], reverse=True)


# ── UI ────────────────────────────────────────────────────────────────────────
st.title("Letter Recognizer")

model = load_model()
if model is None:
    st.error(
        f"Model file not found: `{MODEL_PATH}`\n\n"
        f"Place `letter_recognizer_final.pth` in the same folder as `app.py`, "
        f"or update the `MODEL_PATH` variable at the top of the file."
    )
    st.stop()

st.success(f"Model loaded from `{MODEL_PATH}` — {sum(p.numel() for p in model.parameters()):,} parameters")

try:
    from streamlit_drawable_canvas import st_canvas
except ImportError:
    st.error("Run: `pip install streamlit-drawable-canvas`")
    st.stop()

with st.sidebar:
    st.header("Settings")
    stroke_width = st.slider("Pen width", 8, 30, 18)
    canvas_size   = st.slider("Canvas size (px)", 200, 400, 300, step=50)
    st.divider()
    st.markdown(f"**Model** `{os.path.basename(MODEL_PATH)}`")
    st.markdown(f"Parameters: `{sum(p.numel() for p in model.parameters()):,}`")

col_canvas, col_result = st.columns([1.2, 1])

with col_canvas:
    st.subheader("Draw here")
    canvas_result = st_canvas(
        fill_color="rgba(0,0,0,0)",
        stroke_width=stroke_width,
        stroke_color="#FFFFFF",
        background_color="#000000",
        height=canvas_size,
        width=canvas_size,
        drawing_mode="freedraw",
        key="canvas",
    )
    if st.button("Clear canvas", use_container_width=True):
        st.rerun()

with col_result:
    st.subheader("Prediction")

    if canvas_result.image_data is not None:
        tensor, preview = preprocess(canvas_result.image_data)

        if tensor is None:
            st.caption("Nothing drawn yet.")
        else:
            ranked = predict(model, tensor)
            best_label, best_prob = ranked[0]

            st.markdown(
                f"<div style='font-size:72px;font-weight:600;line-height:1;'>{best_label}</div>",
                unsafe_allow_html=True,
            )
            st.caption(f"{best_prob * 100:.1f}% confidence")
            st.divider()

            st.markdown("**Top 5**")
            for label, prob in ranked[:5]:
                c1, c2, c3 = st.columns([0.12, 0.70, 0.18])
                c1.markdown(f"**{label}**")
                c2.progress(float(prob))
                c3.caption(f"{prob*100:.1f}%")

            st.divider()
            st.caption("28×28 input the model sees")
            st.image(preview.resize((112, 112), Image.NEAREST))
    else:
        st.caption("Draw something on the canvas.")
