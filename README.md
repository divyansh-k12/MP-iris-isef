import streamlit as st
import pandas as pd
import torch
import os
import random
from PIL import Image

# Import the logic from the previous pipeline script
# (Assuming your functions are in a file named logic.py or defined here)
from logic import (
    select_random_participant, 
    run_exposure_model, 
    run_digital_twin, 
    get_histology_image, 
    run_cnn_inference,
    ALLOWED_PARTICIPANT_IDS
)

# --- PAGE CONFIG ---
st.set_page_config(page_title="MP-Twin: Science Fair Demo", layout="wide")

st.title("🔬 MP-Twin: Computational Toxicology Pipeline")
st.markdown("""
**Goal:** This demonstration simulates how microplastics (MPs) enter the human body based on lifestyle data, 
predicts organ-wise accumulation using a **Digital Twin**, and compares **Human vs AI** counting accuracy.
""")

# --- SIDEBAR: Controls & Info ---
with st.sidebar:
    st.header("Pipeline Controls")
    uploaded_file = st.file_uploader("Upload Survey CSV", type="csv")
    
    st.divider()
    st.info("💡 This is an educational demonstration. Organ images are synthetic and based on model predictions.")

# --- STEP 1 & 2: Participant Selection ---
if uploaded_file:
    if st.button("🚀 Select Random Participant & Run Pipeline"):
        # Reset state or clear previous results if needed
        st.session_state['participant'] = select_random_participant(uploaded_file)
        st.success(f"Selected Participant ID: {st.session_state['participant']['Participant_Index']}")

# Check if a participant has been selected
if 'participant' in st.session_state:
    p_data = st.session_state['participant']
    
    # --- STEP 3: Exposure Results ---
    st.header("1. Exposure Assessment")
    exposure_results = run_exposure_model(p_data)
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Ingestion (particles)", int(sum(exposure_results['oral_total'].values())))
    col2.metric("Inhalation (particles)", int(sum(exposure_results['inhalation_total'].values())))
    col3.metric("Dermal (particles)", int(sum(exposure_results['dermal_total'].values())))

    # --- STEP 4: Digital Twin Simulation ---
    st.header("2. Digital Twin Organ Accumulation")
    with st.spinner("Running compartmental simulation..."):
        burdens_df = run_digital_twin(exposure_data=exposure_results)
        st.dataframe(burdens_df, use_container_width=True)

    # --- STEP 5: Synthetic Image Selection ---
    st.header("3. Histology Analysis (AI vs Human)")
    
    # For demo, we select the Liver
    target_organ = "Liver"
    burden = burdens_df.loc[burdens_df['Organ'] == target_organ, 'Microplastic_Count'].values[0]
    img_path = get_histology_image(target_organ, burden)
    
    col_img, col_ui = st.columns([2, 1])
    
    with col_img:
        st.subheader(f"Synthetic {target_organ} Section")
        image = Image.open(img_path)
        st.image(image, caption=f"Predicted Burden: {burden:.1f} particles/unit", use_container_width=True)

    with col_ui:
        st.subheader("Counting Challenge")
        human_guess = st.number_input("How many MPs do you count manually?", min_value=0, step=1)
        
        if st.button("Run CNN Inference"):
            with st.spinner("CNN Analysis in progress..."):
                cnn_count = run_cnn_inference(img_path)
                
                # Show results
                st.write(f"**Human Estimate:** {human_guess}")
                st.write(f"**CNN Count:** {cnn_count}")
                
                error = abs(human_guess - cnn_count)
                st.error(f"**Absolute Error:** {round(error, 2)}")
                
                if error < 5:
                    st.balloons()
                    st.success("Great job! You are as accurate as the AI.")
                else:
                    st.warning("The CNN detected particles that might be too small for the human eye.")

else:
    st.warning("Please upload the 'survey_data1.csv' file in the sidebar to begin.")

# --- FOOTER ---
st.divider()
st.caption("Scientific Disclaimer: All data generated is based on a deterministic compartmental model for research demonstration purposes only.")
