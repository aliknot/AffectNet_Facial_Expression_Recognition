import streamlit as st
from inference import predict_emotion
from PIL import Image

# Set up the page layout
st.set_page_config(page_title="Facial Expression Recognition", page_icon="🎭")

st.title("🎭 Facial Expression Recognition")
st.write(
    "Upload a photo to predict the emotion. The app will automatically detect and crop the face."
)

# Create a file uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", width=300)

    # Add a button to trigger the prediction
    # Add a button to trigger the prediction
    if st.button("Predict Emotion"):
        with st.spinner("Detecting face and analyzing..."):
            try:
                # Unpack all THREE return values
                emotion, confidence, cropped_img = predict_emotion(uploaded_file)

                # Create two columns for a clean layout
                col1, col2 = st.columns(2)

                with col1:
                    # Display the text results in the left column
                    st.success(f"**Predicted Emotion:** {emotion}")
                    st.info(f"**Confidence:** {confidence:.2%}")

                with col2:
                    # Display the cropped face in the right column
                    st.image(
                        cropped_img, caption="Detected Face (Model Input)", width=150
                    )

            except RuntimeError as re:
                st.error(f"System Error: {re}")
            except ValueError as ve:
                st.warning(str(ve))
            except Exception as e:
                st.error(f"An unexpected error occurred during prediction: {e}")
