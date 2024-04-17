import streamlit as st
from PIL import Image
import torch
from torchvision.transforms import ToTensor
from model_file import ViTForImageClassification

# Load the model
model = ViTForImageClassification()
model.load_state_dict(torch.load('E:\SFT\Psmt\model.pt'))
model.eval()

# Define class labels
class_labels = {0: "No Dr.", 1: "Mild", 2: "Moderate", 3: "Non PDR", 4: "Severe"}

# Function to predict image
def predict(image):
    # Preprocess image
    image = ToTensor()(image).unsqueeze(0)
    # Make prediction
    with torch.no_grad():
        output, _ = model(image, None)
        predicted_class = torch.argmax(output, 1).item()
    return class_labels[predicted_class]

# Streamlit app
def main():
    st.title("Diabetic Retinopathy Severity Predictor")
    st.sidebar.title("About")

    st.write("This app predicts the severity of diabetic retinopathy from fundus images.")

    # File uploader
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Predict and display result
        result = predict(image)
        st.write(f"Predicted Severity: {result}")

if __name__ == "__main__":
    main()
