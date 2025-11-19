import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt

st.title("🔍 Autoencoder-Based Anomaly Detection")
st.write("Upload an image and the model will reconstruct it to detect anomalies.")


@st.cache_resource
def load_model():
    return tf.keras.models.load_model("autoencoder_model.keras")

autoencoder = load_model()



def preprocess(img):
    img = img.convert("L").resize((28, 28))     
    img = np.array(img) / 255.0
    img = img.reshape(1, 28, 28, 1)
    return img


# Inference function
def detect_anomaly(img):
    x = preprocess(img)
    recon = autoencoder.predict(x)[0]

    error = np.mean((x[0] - recon) ** 2)
    status = " Anomaly" if error > 0.02 else " Normal"

    return error, recon



uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    img = Image.open(uploaded_file)

    st.subheader(" Original Image")
    st.image(img, width=200)

   
    error, recon = detect_anomaly(img)

    st.subheader(" Result")
    st.write(f"**Reconstruction Error:** `{error:.5f}`")
    st.write(f"**Status:** {(' Normal' if error < 0.02 else ' Anomaly')}")

   
    col1, col2 = st.columns(2)

    with col1:
        st.image(img.resize((200, 200)), caption="Original")

    with col2:
        recon_img = (recon.reshape(28, 28) * 255).astype("uint8")
        st.image(recon_img, caption="Reconstructed")

    st.subheader("🔧 Reconstruction Error Plot")

    fig, ax = plt.subplots(figsize=(5, 2))
    ax.bar(["Reconstruction Error"], [error])
    ax.set_ylim(0, max(0.05, error + 0.01))
    ax.set_ylabel("Error")
    ax.set_title("Reconstruction Error Chart")

    st.pyplot(fig)

    st.success("Processing complete!")
