import os
import sys
from pathlib import Path
import streamlit as st
import cv2
import numpy as np
from PIL import Image
import time
# Add this with your other imports at the top
from app.processor import ImageProcessor

# Set page config FIRST
st.set_page_config(
    page_title="Industrial Equipment Analyzer",
    page_icon="🏭", 
    layout="wide"
)

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Now import your app modules
from app.cnn_model import EquipmentDefectModel
from app.yolo_detector import SafetyGearDetector
from app.processor import ImageProcessor
from app.config import Config

def main():
    """Main dashboard function"""
    # Initialize components
    processor = ImageProcessor()
    cnn_model = EquipmentDefectModel()
    yolo_detector = SafetyGearDetector()
    
    try:
        cnn_model.load_weights()
        st.success("✅ CNN Model loaded successfully")
    except Exception as e:
        st.error(f"❌ Failed to load CNN model: {str(e)}")
        return
    
    # Dashboard UI
    st.title("🏭 Industrial Equipment Analysis Dashboard")
    st.markdown("Upload images to detect equipment defects and safety compliance")

    # Enhanced Settings Section
    with st.sidebar:
        st.title("⚙️ Analysis Settings")
        
        with st.expander("Model Configuration", expanded=True):
            confidence_threshold = st.slider(
                "Detection Confidence Threshold",
                min_value=0.1,
                max_value=1.0,
                value=0.7,
                step=0.05,
                help="Adjust the minimum confidence level for defect detection"
            )
            
            processing_mode = st.selectbox(
                "Processing Mode",
                ["Fast", "Balanced", "Accurate"],
                index=1,
                help="Trade-off between speed and accuracy"
            )
            
            show_technical = st.toggle(
                "Show Technical Details",
                value=False,
                help="Display advanced model information"
            )
        
        if show_technical:
            st.markdown("---")
            st.markdown("### Technical Specifications")
            st.text(f"CNN Input Size: {Config.CNN_INPUT_SIZE}")
            st.text(f"YOLO Model: {Config.YOLO_MODEL}")
            st.text(f"CNN Model: {os.path.basename(Config.CNN_MODEL_PATH)}")

    # File uploader
    uploaded_file = st.file_uploader(
        "Choose an equipment image...", 
        type=["jpg", "jpeg", "png"],
        help="Upload images of industrial equipment for analysis"
    )

    if uploaded_file is not None:
        try:
            # Display original image
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Original Image")
                image = Image.open(uploaded_file)
                
                # Convert image to RGB if it has transparency
                if image.mode == 'RGBA':
                    image = image.convert('RGB')
                
                st.image(image, caption="Uploaded Image", use_column_width=True)
                
                # Save temporarily for processing
                temp_path = "temp_upload.jpg"
                image.save(temp_path, quality=95)
            
            with col2:
                st.subheader("Analysis Results")
                
                with st.spinner("Analyzing image..."):
                    try:
                        # Load and process image
                        img_array = processor.load_image(temp_path)
                        
                        # CNN Prediction
                        start_time = time.time()
                        
                        # Fixed preprocessing call
                        img_resized = cv2.resize(img_array, Config.CNN_INPUT_SIZE)
                        img_normalized = img_resized.astype('float32') / 255.0
                        cnn_input = np.expand_dims(img_normalized, axis=0)
                        
                        cnn_pred = cnn_model.predict(cnn_input)[0]
                        processing_time = time.time() - start_time
                        
                        defect_prob = cnn_pred[0]
                        status = "DEFECT" if defect_prob > 0.5 else "NORMAL"
                        
                        # YOLO Detection
                        yolo_input = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                        detections = yolo_detector.detect(yolo_input)
                        filtered_detections = [
                            d for d in detections 
                            if d['confidence'] > confidence_threshold
                        ]
                        
                        # Display results
                        st.markdown("### Equipment Status")
                        # With this:
                        if status == "DEFECT":
                            st.error(f"🚨 Defect Detected ({defect_prob:.2%} confidence)")
                            st.progress(ImageProcessor.convert_to_display_prob(defect_prob))
                        else:
                            st.success(f"✅ Normal Equipment ({1-defect_prob:.2%} confidence)")
                            st.progress(ImageProcessor.convert_to_display_prob(1 - defect_prob))
                        
                        st.markdown("### Safety Gear Detection")
                        if filtered_detections:
                            detection_table = []
                            for det in filtered_detections:
                                detection_table.append([
                                    det['label'].upper(),
                                    f"{det['confidence']:.2%}",
                                    det['bbox']
                                ])
                            
                            st.dataframe(
                                detection_table,
                                column_config={
                                    "0": "Item",
                                    "1": "Confidence",
                                    "2": "Bounding Box"
                                },
                                hide_index=True,
                                use_container_width=True
                            )
                            
                            # Show visualization
                            vis_img = processor.visualize_detections(img_array, filtered_detections)
                            st.image(vis_img, caption="Safety Gear Detection", use_column_width=True)
                        else:
                            st.warning("⚠️ No safety gear detected above confidence threshold")
                        
                        st.caption(f"⏱️ Processing time: {processing_time:.2f} seconds | Mode: {processing_mode}")
                    
                    except Exception as e:
                        st.error(f"Analysis failed: {str(e)}")
                    finally:
                        # Clean up
                        if os.path.exists(temp_path):
                            os.remove(temp_path)
        
        except Exception as e:
            st.error(f"Error loading image: {str(e)}")

    # How to Use Section
    with st.expander("📖 How to Use This Dashboard", expanded=True):
        st.markdown("""
        ### Quick Start Guide
        1. **Upload an image** using the file uploader
        2. View automatic analysis results including:
           - Equipment defect probability
           - Safety gear detection
        3. Adjust settings in the sidebar to:
           - Change detection sensitivity
           - Select processing mode
           - View technical details
        
        ### Understanding Results
        - 🚨 **Red Alert**: High probability of equipment defect
        - ✅ **Green Check**: Equipment appears normal
        - 🛡️ **Blue Boxes**: Detected safety gear items
        - ⚠️ **Warning**: No safety gear detected
        """)

    # Footer
    st.markdown("---")
    st.markdown("""
    <style>
    .footer {
        position: relative;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: transparent;
        text-align: center;
        padding: 10px;
        color: #666;
        font-size: 0.8em;
    }
    </style>
    <div class="footer">
        <p>Industrial Equipment Analysis System v1.1 | © 2025</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()