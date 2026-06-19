import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import os

st.set_page_config(page_title="Handwriting Board", layout="wide")

# ── Change this to your .pth path ────────────────────────────────────────────
MODEL_PATH = "letter_recognizer_final.pth"
# ─────────────────────────────────────────────────────────────────────────────

LABELS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
NUM_CLASSES = 26
MEAN, STD = 0.1307, 0.3081


class LetterNet(nn.Module):
    def __init__(self, num_classes=26):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout2d(0.1),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout2d(0.1),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(3),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 9, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    m = LetterNet(NUM_CLASSES)
    m.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    m.eval()
    return m


def preprocess(img_array):
    rgb  = img_array[:, :, :3].astype(np.float32)
    gray = 0.299*rgb[:,:,0] + 0.587*rgb[:,:,1] + 0.114*rgb[:,:,2]
    if gray.max() < 10:
        return None, None
    img = Image.fromarray(gray.astype(np.uint8), mode="L")
    img = img.resize((28, 28), Image.LANCZOS)
    img = img.rotate(-90)
    img = img.transpose(Image.FLIP_LEFT_RIGHT)
    arr = np.array(img).astype(np.float32) / 255.0
    arr_norm = (arr - MEAN) / STD
    tensor  = torch.tensor(arr_norm).unsqueeze(0).unsqueeze(0)
    preview = Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8))
    return tensor, preview


@torch.no_grad()
def predict(model, tensor):
    probs = F.softmax(model(tensor), dim=1).squeeze().tolist()
    return sorted(zip(LABELS, probs), key=lambda x: x[1], reverse=True)


# ── Session state ─────────────────────────────────────────────────────────────
if "board"          not in st.session_state: st.session_state.board = ""
if "canvas_version" not in st.session_state: st.session_state.canvas_version = 0
if "last_pred"      not in st.session_state: st.session_state.last_pred = None
if "last_preview"   not in st.session_state: st.session_state.last_preview = None


# ── Load model ────────────────────────────────────────────────────────────────
model = load_model()
if model is None:
    st.error(
        f"Model not found at `{MODEL_PATH}`. "
        "Place `letter_recognizer_final.pth` next to `app.py` or update `MODEL_PATH`."
    )
    st.stop()


# ── Canvas import ─────────────────────────────────────────────────────────────
try:
    from streamlit_drawable_canvas import st_canvas
except ImportError:
    st.error("Run: `pip install streamlit-drawable-canvas`")
    st.stop()


# ── Layout ────────────────────────────────────────────────────────────────────
st.title("Handwriting board")

# TEXT BOARD
board_display = st.session_state.board + "▌"
st.markdown(
    f"""
    <div style="
        font-family: monospace;
        font-size: 26px;
        line-height: 1.6;
        min-height: 110px;
        padding: 12px 16px;
        border: 0.5px solid #ccc;
        border-radius: 10px;
        background: #f9f9f9;
        white-space: pre-wrap;
        word-break: break-all;
        margin-bottom: 0.75rem;
    ">{board_display}</div>
    """,
    unsafe_allow_html=True,
)

# Board controls
bc1, bc2, bc3, bc4, bc5 = st.columns([1, 1, 1, 1, 3])
if bc1.button("Space",     use_container_width=True): st.session_state.board += " ";  st.rerun()
if bc2.button("Backspace", use_container_width=True): st.session_state.board = st.session_state.board[:-1]; st.rerun()
if bc3.button("New line",  use_container_width=True): st.session_state.board += "\n"; st.rerun()
if bc4.button("Clear board", use_container_width=True): st.session_state.board = "";  st.rerun()

st.divider()

# DRAWING AREA + PREDICTION PANEL
left, right = st.columns([1.4, 1])

with left:
    st.caption("Draw a letter")
    canvas_result = st_canvas(
        fill_color="rgba(0,0,0,0)",
        stroke_width=18,
        stroke_color="#FFFFFF",
        background_color="#000000",
        height=300,
        width=300,
        drawing_mode="freedraw",
        key=f"canvas_{st.session_state.canvas_version}",
        update_streamlit=False,   # only sends data when you interact with buttons
    )

    a1, a2 = st.columns(2)

    # ADD TO BOARD
    if a1.button("✓  Add to board", use_container_width=True, type="primary"):
        if canvas_result.image_data is not None:
            tensor, preview = preprocess(canvas_result.image_data)
            if tensor is not None:
                ranked = predict(model, tensor)
                best_label, best_prob = ranked[0]
                st.session_state.board       += best_label
                st.session_state.last_pred   = ranked
                st.session_state.last_preview = preview
                st.session_state.canvas_version += 1   # clears canvas
                st.rerun()
            else:
                st.warning("Nothing drawn yet.")
        else:
            st.warning("Nothing drawn yet.")

    # CLEAR CANVAS
    if a2.button("Clear canvas", use_container_width=True):
        st.session_state.canvas_version += 1
        st.session_state.last_pred    = None
        st.session_state.last_preview = None
        st.rerun()

with right:
    if st.session_state.last_pred:
        ranked  = st.session_state.last_pred
        best_label, best_prob = ranked[0]

        st.markdown(
            f"<div style='font-size:64px;font-weight:500;line-height:1;margin-bottom:4px;'>{best_label}</div>",
            unsafe_allow_html=True,
        )
        st.caption(f"{best_prob*100:.1f}% confidence")
        st.divider()

        st.caption("Top 5")
        for label, prob in ranked[:5]:
            c1, c2, c3 = st.columns([0.1, 0.72, 0.18])
            c1.markdown(f"**{label}**")
            c2.progress(float(prob))
            c3.caption(f"{prob*100:.1f}%")

        if st.session_state.last_preview:
            st.divider()
            st.caption("28×28 model input")
            st.image(st.session_state.last_preview.resize((84, 84), Image.NEAREST))
    else:
        st.caption("Draw a letter and click **✓ Add to board**.")

# Sidebar info
with st.sidebar:
    st.header("Model")
    st.markdown(f"`{os.path.basename(MODEL_PATH)}`")
    st.caption(f"{sum(p.numel() for p in model.parameters()):,} parameters")
    st.divider()
    st.markdown("**How to use**")
    st.markdown(
        "1. Draw a letter on the canvas\n"
        "2. Click **✓ Add to board**\n"
        "3. The letter appears on the board\n"
        "4. Use **Space / Backspace / New line** to edit"
    )
