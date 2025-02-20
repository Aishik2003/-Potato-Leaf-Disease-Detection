import streamlit as st
from predict import PotatoDiseaseClassifier
import os
from PIL import Image
import hashlib

# Configure Streamlit for better performance
st.set_page_config(
    page_title="Potato Disease Classification",
    page_icon="🥔",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Initialize the classifier at startup and cache it
@st.cache_resource(show_spinner=False)
def load_classifier():
    try:
        model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'potato_disease_model.h5')
        classifier = PotatoDiseaseClassifier(model_path)
        return classifier
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def get_image_hash(image_data):
    """Generate a hash for the image data to ensure unique caching"""
    try:
        # If image_data is a file-like object, read its content
        if hasattr(image_data, 'read'):
            content = image_data.read()
            image_data.seek(0)  # Reset file pointer
        else:
            content = image_data.tobytes()
        
        return hashlib.md5(content).hexdigest()
    except Exception:
        # If we can't generate a hash, return a random one
        return os.urandom(16).hex()

def process_image(image_data):
    """Process image and return prediction results"""
    try:
        # Get the cached classifier
        classifier = load_classifier()
        if classifier is None:
            st.error("Model could not be loaded. Please check if the model file exists.")
            return None

        # Convert to PIL Image if needed
        if not isinstance(image_data, Image.Image):
            image = Image.open(image_data).convert('RGB')
        else:
            image = image_data.convert('RGB')
        
        # Make prediction
        result = classifier.predict(image)
        
        if result:
            st.session_state['last_prediction'] = result
        return result
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
        return None

def display_prediction(result):
    """Display prediction results"""
    if result and 'class' in result and 'confidence' in result and result['confidence'] > 0.5:
        # Show the diagnosis
        st.write("### Diagnosis")
        diagnosis_class = result['class'].replace('_', ' ')
        
        if result['class'] == 'Healthy':
            st.success(f"✅ Your potato plant appears to be {diagnosis_class}!")
            st.write("""
            ### Recommendations
            - Continue regular monitoring
            - Maintain current care practices
            - Ensure proper watering and fertilization
            - Watch for any signs of disease development
            """)
            
        elif result['class'] == 'Early_Blight':
            st.warning(f"⚠️ Your potato plant shows signs of {diagnosis_class}")
            st.write("""
            ### Recommendations
            - Remove affected leaves immediately
            - Improve air circulation between plants
            - Apply appropriate fungicide as per local guidelines
            - Avoid overhead watering
            - Monitor other plants for similar symptoms
            """)
            
        elif result['class'] == 'Late_Blight':
            st.error(f"🚨 Your potato plant shows signs of {diagnosis_class}")
            st.write("""
            ### Recommendations
            - Take immediate action as this disease spreads rapidly
            - Apply appropriate fungicide treatment
            - Remove and destroy affected plants
            - Improve drainage in the field
            - Increase plant spacing for better air circulation
            - Monitor weather conditions
            """)
    else:
        st.warning("""
        ⚠️ Unable to make a reliable diagnosis. This could mean:
        - The image quality is not optimal
        - The symptoms are not clear enough
        - The leaf might be in an early stage of infection
        
        Please try uploading a clearer image or consult with an expert.
        """)

def main():
    st.title("🥔 Potato Disease Classification")
    st.write("""
    Upload an image of a potato plant leaf to detect diseases.
    
    **Note:** For best results:
    - Use clear, well-lit images
    - Focus on the leaf area
    - Ensure the image shows the symptoms clearly
    - Avoid blurry or dark images
    """)
    
    # Initialize session state for tracking changes
    if 'last_uploaded_file' not in st.session_state:
        st.session_state['last_uploaded_file'] = None
    if 'last_prediction' not in st.session_state:
        st.session_state['last_prediction'] = None
    
    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        try:
            # Check if this is a new file
            current_file_hash = get_image_hash(uploaded_file)
            
            # Create two columns for layout
            col1, col2 = st.columns([1, 1.5])
            
            with col1:
                # Display the uploaded image with smaller dimensions
                image = Image.open(uploaded_file)
                # Resize image while maintaining aspect ratio
                basewidth = 300
                wpercent = (basewidth/float(image.size[0]))
                hsize = int((float(image.size[1])*float(wpercent)))
                image = image.resize((basewidth, hsize), Image.Resampling.LANCZOS)
                st.image(image, caption='Uploaded Image')
            
            with col2:
                # Make prediction if this is a new file
                if (st.session_state['last_uploaded_file'] != current_file_hash):
                    st.session_state['last_uploaded_file'] = current_file_hash
                    with st.spinner('Analyzing image...'):
                        result = process_image(uploaded_file)
                        display_prediction(result)
                else:
                    # Display the last prediction if it exists
                    if st.session_state['last_prediction']:
                        display_prediction(st.session_state['last_prediction'])
                
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")
    else:
        # Clear the last uploaded file hash when no file is uploaded
        st.session_state['last_uploaded_file'] = None
        st.session_state['last_prediction'] = None

if __name__ == '__main__':
    main()
