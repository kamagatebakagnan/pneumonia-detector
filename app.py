import streamlit as st
import requests
from PIL import Image
import base64
import io
from datetime import datetime

# Configuration
st.set_page_config(
    page_title="Pneumonia Detector",
    page_icon="🫁",
    layout="wide"
)

# Initialiser l'historique dans la session
if "historique" not in st.session_state:
    st.session_state.historique = []

# Header
st.title("🫁 Pneumonia Detector")
st.markdown("Uploadez une radio pulmonaire pour obtenir un diagnostic automatique avec visualisation Grad-CAM.")
st.divider()

# Layout principal
col_upload, col_result = st.columns(2)

with col_upload:
    st.subheader("📷 Radio pulmonaire")
    uploaded_file = st.file_uploader(
        "Choisir une image",
        type=["jpeg", "jpg", "png"],
    )

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, use_container_width=True)

        if st.button("🔬 Analyser", type="primary", use_container_width=True):
            with st.spinner("Analyse en cours..."):
                uploaded_file.seek(0)
                response = requests.post(
                    "http://localhost:8000/predict",
                    files={"file": (uploaded_file.name, uploaded_file, "image/jpeg")}
                )

            if response.status_code == 200:
                result = response.json()

                # Sauvegarder dans l'historique
                st.session_state.historique.append({
                    "datetime"  : datetime.now().strftime("%H:%M:%S"),
                    "fichier"   : uploaded_file.name,
                    "prediction": result["prediction"],
                    "confiance" : result["confidence"],
                    "gradcam"   : result["gradcam_b64"],
                })

                # Stocker le résultat courant
                st.session_state.current = result
            else:
                st.error("❌ Erreur API. Vérifiez que FastAPI tourne sur le port 8000.")

with col_result:
    st.subheader("🔬 Diagnostic")

    if "current" in st.session_state:
        result = st.session_state.current
        prediction  = result["prediction"]
        confidence  = result["confidence"]
        probs       = result["probabilities"]
        gradcam_b64 = result["gradcam_b64"]

        # Résultat
        if prediction == "PNEUMONIA":
            st.error(f"🔴 **PNEUMONIE DÉTECTÉE**")
        else:
            st.success(f"🟢 **NORMAL**")

        st.metric("Confiance", f"{confidence}%")

        # Probabilités
        st.markdown("**Probabilités :**")
        st.progress(probs["NORMAL"],
                    text=f"Normal : {round(probs['NORMAL']*100, 1)}%")
        st.progress(probs["PNEUMONIA"],
                    text=f"Pneumonie : {round(probs['PNEUMONIA']*100, 1)}%")

        st.divider()

        # Grad-CAM
        st.subheader("🌡️ Grad-CAM — Zone analysée")
        st.caption("Les zones rouges indiquent où le modèle a focalisé son attention.")
        gradcam_bytes = base64.b64decode(gradcam_b64)
        gradcam_image = Image.open(io.BytesIO(gradcam_bytes))
        st.image(gradcam_image, use_container_width=True)

    else:
        st.info("👆 Uploadez une radio et cliquez sur **Analyser** pour obtenir un diagnostic.")

st.divider()

# Historique des diagnostics
if st.session_state.historique:
    st.subheader("📋 Historique des diagnostics")

    for i, entry in enumerate(reversed(st.session_state.historique)):
        with st.expander(
            f"{'🔴' if entry['prediction'] == 'PNEUMONIA' else '🟢'} "
            f"{entry['datetime']} — {entry['fichier']} "
            f"({entry['prediction']}, {entry['confiance']}%)"
        ):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"**Fichier :** {entry['fichier']}")
                st.markdown(f"**Heure :** {entry['datetime']}")
                st.markdown(f"**Résultat :** {entry['prediction']}")
                st.markdown(f"**Confiance :** {entry['confiance']}%")
            with col2:
                gradcam_bytes = base64.b64decode(entry["gradcam"])
                gradcam_image = Image.open(io.BytesIO(gradcam_bytes))
                st.image(gradcam_image, caption="Grad-CAM",
                         use_container_width=True)

st.caption("Modèle : ResNet50 Transfer Learning | AUC-ROC : 0.9983 | Dataset : Chest X-Ray (5263 images)")
