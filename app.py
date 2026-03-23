import streamlit as st
from inference import predict_emotion
from PIL import Image

# Set up the page layout (added 'centered' layout for a tighter, mobile-friendly look)
st.set_page_config(
    page_title="Facial Expression Recognition", page_icon="🎭", layout="centered"
)

# --- Header Section ---
st.title("🎭 Facial Expression Recognition")
st.markdown(
    "Upload a photo or take a picture to predict the emotion. The model will automatically detect and isolate the face."
)
st.divider()

# --- Sidebar ---
# Moving settings to the sidebar keeps the main UI clean
with st.sidebar:
    st.header("⚙️ Settings")
    input_method = st.radio("Choose Input Method:", ("Upload Image", "Use Webcam"))

# --- Main Input Area ---
uploaded_file = None

if input_method == "Upload Image":
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
else:
    uploaded_file = st.camera_input("Take a picture")

# --- Processing & Results ---
if uploaded_file is not None:
    # Display the input image centrally
    image = Image.open(uploaded_file)
    # use_container_width makes it responsive to the screen size
    st.image(image, caption="Input Image", use_container_width=True)

    st.write("")  # Add a little vertical breathing room

    # A primary button stands out visually
    if st.button("Predict Emotion", type="primary", use_container_width=True):
        with st.spinner("Detecting face and analyzing..."):
            try:
                # Unpack the exact three variables you requested
                emotion, confidence, cropped_img = predict_emotion(uploaded_file)

                st.divider()
                st.subheader("Analysis Results")

                # Create a 3-column layout for a modern dashboard look
                col1, col2, col3 = st.columns(3)

                with col1:
                    # st.metric creates beautiful, large-number data callouts
                    st.metric(label="Predicted Emotion", value=emotion)

                with col2:
                    st.metric(label="Confidence", value=f"{confidence:.2%}")

                with col3:
                    # Show the cropped face aligned with the data
                    st.image(cropped_img, caption="Detected Face", width=120)

            except RuntimeError as re:
                st.error(f"System Error: {re}")
            except ValueError as ve:
                st.warning(str(ve))
            except Exception as e:
                st.error(f"An unexpected error occurred during prediction: {e}")
