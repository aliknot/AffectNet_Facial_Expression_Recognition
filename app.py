import streamlit as st
from inference import predict_emotion
from PIL import Image

# Set up the page layout
st.set_page_config(page_title="Facial Expression Recognition", page_icon="🎭")

st.title("🎭 Facial Expression Recognition")
st.write("Upload a tightly cropped image of a face to predict the emotion.")

# Create a file uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", width=300)

    # Add a button to trigger the prediction
    if st.button("Predict Emotion"):
        with st.spinner("Analyzing face..."):
            try:
                # Call the function from your inference.py
                emotion, confidence = predict_emotion(uploaded_file)

                # Display the results
                st.success(f"**Predicted Emotion:** {emotion}")
                st.info(f"**Confidence:** {confidence:.2%}")

            except Exception as e:
                st.error(f"An error occurred during prediction: {e}")
