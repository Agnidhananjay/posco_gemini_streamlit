import streamlit as st
import os
import json
import tempfile
import shutil
from dotenv import load_dotenv
from google import genai
import time

# Import custom modules
from utils import (
    pdf_to_images_2,
    classify_images,
    process_engineering_images,
    merge_engineering_data
)
from prompts import (
    CLASSIFICATION_PROMPT,
    PROMPT_MAP,
    PROMPT_TABLE
)
from pydantic_models import (
    Borehole_data,
    Extracted_data
)

# Load environment variables
load_dotenv()

# Set page config
st.set_page_config(
    page_title="Geotechnical Report Analysis",
    layout="wide",
    initial_sidebar_state="collapsed",
    menu_items={
        'About': "Extract and analyze drilling data from geotechnical reports"
    }
)

# Default configuration
DEFAULT_CONFIG = {
    "max_workers": 4,
    "fixed_width": 3000,
    "max_concurrent": 6
}

# Custom CSS for better UI
st.markdown("""
<style>
    .stProgress > div > div > div > div {
        background-color: #1f77b4;
    }
    .success-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        margin: 1rem 0;
    }
    .error-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 0.5rem;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .step-header {
        font-size: 1.1rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
        color: #2c3e50;
    }
    .step-container {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None
if 'file_paths' not in st.session_state:
    st.session_state.file_paths = []
if 'classification_results' not in st.session_state:
    st.session_state.classification_results = None
if 'processing_complete' not in st.session_state:
    st.session_state.processing_complete = False
if 'last_file_hash' not in st.session_state:
    st.session_state.last_file_hash = None

def reset_session_state():
    """Reset all session state variables"""
    st.session_state.processed_data = None
    st.session_state.file_paths = []
    st.session_state.classification_results = None
    st.session_state.processing_complete = False
    st.session_state.last_file_hash = None

def get_api_key():
    """Get API key from environment or Streamlit secrets"""
    # Try Streamlit secrets first (for cloud deployment)
    try:
        return st.secrets["GEMINI_API_KEY"]
    except:
        # Fall back to environment variable
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            st.error("❌ GEMINI_API_KEY not found. Please set it in your .env file or Streamlit secrets.")
            st.stop()
        return api_key

def main():
    # Header
    st.title("🏗️ Geotechnical Report Analysis System")
    st.markdown("Extract and analyze drilling data from geotechnical PDF reports")
    st.markdown("---")
    
    # Get API key
    api_key = get_api_key()
    
    # File upload section
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.markdown("### 📄 Upload PDF Document")
        uploaded_file = st.file_uploader(
            "Choose a geotechnical report PDF",
            type="pdf",
            help="Upload a PDF containing boring logs and site maps",
            on_change=reset_session_state,
            label_visibility="collapsed"
        )
    
    with col2:
        if uploaded_file:
            st.metric("File Size", f"{uploaded_file.size / 1024:.1f} KB")
    
    with col3:
        if uploaded_file:
            st.metric("File Name", uploaded_file.name[:20] + "..." if len(uploaded_file.name) > 20 else uploaded_file.name)
    
    if uploaded_file is not None:
        # Check if this is the same file that was already processed
        file_hash = hash(uploaded_file.read())
        uploaded_file.seek(0)  # Reset file pointer
        
        if 'last_file_hash' not in st.session_state:
            st.session_state.last_file_hash = None
        
        # Only process if it's a new file or not yet processed
        if (st.session_state.last_file_hash != file_hash or 
            not st.session_state.processing_complete or 
            st.session_state.processed_data is None):
            
            st.session_state.last_file_hash = file_hash
            
            # Create temporary directory for processing
            temp_dir = tempfile.mkdtemp()
            pdf_path = os.path.join(temp_dir, "input.pdf")
            output_dir = os.path.join(temp_dir, "images")
            os.makedirs(output_dir, exist_ok=True)
            
            # Save uploaded file
            with open(pdf_path, "wb") as f:
                f.write(uploaded_file.getvalue())
            
            try:
                # Initialize Gemini client
                client = genai.Client(api_key=api_key)
                
                # Processing pipeline
                st.markdown("---")
                st.markdown("### 🔄 Processing Pipeline")
                
                # Create a container for all processing steps
                processing_container = st.container()
                
                with processing_container:
                    # Step 1: PDF to Images
                    with st.container():
                        col1, col2 = st.columns([4, 1])
                        with col1:
                            st.markdown('<p class="step-header">🖼️ Step 1: Converting PDF to Images...</p>', unsafe_allow_html=True)
                            progress_1 = st.progress(0)
                        
                        start_time = time.time()
                        _, file_paths = pdf_to_images_2(
                            pdf_path, 
                            output_dir, 
                            DEFAULT_CONFIG["fixed_width"], 
                            DEFAULT_CONFIG["max_workers"]
                        )
                        st.session_state.file_paths = file_paths
                        
                        progress_1.progress(100)
                        with col2:
                            st.success(f"✅ {len(file_paths)} pages")
                            st.caption(f"⏱️ {time.time() - start_time:.1f}s")
                    
                    # Step 2: Classification
                    with st.container():
                        col1, col2 = st.columns([4, 1])
                        with col1:
                            st.markdown('<p class="step-header">🔍 Step 2: Classifying Pages...</p>', unsafe_allow_html=True)
                            progress_2 = st.progress(0)
                        
                        start_time = time.time()
                        classified_images = classify_images(
                            file_paths, CLASSIFICATION_PROMPT, client
                        )
                        st.session_state.classification_results = classified_images
                        
                        progress_2.progress(100)
                        with col2:
                            st.success("✅ Complete")
                            st.caption(f"⏱️ {time.time() - start_time:.1f}s")
                    
                    # Show classification results
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.info(f"📍 **Maps:** {len(classified_images['map'])}")
                    with col2:
                        st.info(f"📊 **Tables:** {len(classified_images['table'])}")
                    with col3:
                        st.info(f"📄 **Other:** {len(classified_images['neither'])}")
                    
                    # Step 3: Data Extraction
                    if len(classified_images["map"]) > 0 or len(classified_images["table"]) > 0:
                        with st.container():
                            col1, col2 = st.columns([4, 1])
                            with col1:
                                st.markdown('<p class="step-header">📊 Step 3: Extracting Data from Images...</p>', unsafe_allow_html=True)
                                progress_3 = st.progress(0)
                                
                                # Calculate total API calls needed
                                total_api_calls = len(classified_images["map"]) + len(classified_images["table"])
                                progress_text = st.empty()
                                progress_text.text(f"Processing {total_api_calls} images concurrently...")
                            
                            start_time = time.time()
                            
                            # Use the original concurrent processing function
                            table_data, map_data = process_engineering_images(
                                classified_images, 
                                client, 
                                PROMPT_MAP, 
                                PROMPT_TABLE,
                                Borehole_data, 
                                Extracted_data, 
                                DEFAULT_CONFIG["max_concurrent"]
                            )
                            
                            # Update progress to 100% after completion
                            progress_3.progress(100)
                            progress_text.text(f"Completed {total_api_calls} images!")
                            
                            with col2:
                                st.success("✅ Complete")
                                st.caption(f"⏱️ {time.time() - start_time:.1f}s")
                        
                        # Step 4: Data Merging
                        with st.container():
                            col1, col2 = st.columns([4, 1])
                            with col1:
                                st.markdown('<p class="step-header">🔗 Step 4: Merging Extracted Data...</p>', unsafe_allow_html=True)
                                progress_4 = st.progress(0)
                            
                            start_time = time.time()
                            final_data = merge_engineering_data(table_data, map_data)
                            st.session_state.processed_data = final_data
                            st.session_state.processing_complete = True
                            
                            progress_4.progress(100)
                            with col2:
                                st.success("✅ Complete")
                                st.caption(f"⏱️ {time.time() - start_time:.1f}s")
                        
                        # Final success message
                        st.success(f"🎉 Successfully processed {len(final_data)} boreholes!")
                        
                    else:
                        st.warning("⚠️ No maps or tables detected in the PDF. Please check your document.")
                
            except Exception as e:
                st.error(f"❌ Error during processing: {str(e)}")
                with st.expander("🔍 Error Details"):
                    st.exception(e)
            
            finally:
                # Cleanup temporary files
                try:
                    shutil.rmtree(temp_dir)
                except:
                    pass
        
        # Show results if already processed
        elif st.session_state.processing_complete and st.session_state.processed_data:
            st.markdown("---")
            st.info("ℹ️ Using previously processed results. Upload a new file to reprocess.")
        
        # Results section - shown whether newly processed or from session state
        if st.session_state.processing_complete and st.session_state.processed_data:
            st.markdown("---")
            st.markdown("### 📋 Results & Analysis")
            
            final_data = st.session_state.processed_data
            
            # Create tabs for different views
            tab1, tab2, tab3 = st.tabs(["📊 Summary", "🔍 Detailed View", "💾 Export"])
            
            with tab1:
                # Overall statistics
                col1, col2, col3, col4 = st.columns(4)
                
                total_samples = sum(
                    sum(len(layer.get('sample_test', [])) 
                        for layer in borehole.get('soil_data', []))
                    for borehole in final_data.values()
                )
                
                total_layers = sum(
                    len(borehole.get('soil_data', []))
                    for borehole in final_data.values()
                )
                
                col1.metric("📍 Total Boreholes", len(final_data))
                col2.metric("🗿 Total Soil Layers", total_layers)
                col3.metric("🧪 Total Samples", total_samples)
                col4.metric("📄 Pages Processed", len(st.session_state.file_paths))
                
                # Borehole summary table
                st.markdown("#### Borehole Summary")
                summary_data = []
                for bh_id, bh_data in final_data.items():
                    summary_data.append({
                        "Borehole ID": bh_id,
                        "Soil Layers": len(bh_data.get('soil_data', [])),
                        "Total Samples": sum(len(layer.get('sample_test', [])) 
                                           for layer in bh_data.get('soil_data', [])),
                        "Has Map Data": "✅" if 'map_data' in bh_data else "❌"
                    })
                
                st.dataframe(
                    summary_data, 
                    use_container_width=True,
                    hide_index=True
                )
            
            with tab2:
                # Borehole selector
                selected_borehole = st.selectbox(
                    "Select Borehole to View Details",
                    options=list(final_data.keys()),
                    format_func=lambda x: f"Borehole {x}"
                )
                
                if selected_borehole:
                    borehole_data = final_data[selected_borehole]
                    
                    col1, col2 = st.columns([2, 1])
                    with col1:
                        st.markdown(f"#### Borehole {selected_borehole} Details")
                    with col2:
                        st.metric("Total Layers", len(borehole_data.get('soil_data', [])))
                    
                    # Display metadata
                    if borehole_data.get('metadata'):
                        with st.expander("📋 Metadata", expanded=True):
                            st.json(borehole_data.get('metadata', {}))
                    
                    # Display soil layers
                    st.markdown("##### 🗿 Soil Layers")
                    for idx, layer in enumerate(borehole_data.get('soil_data', [])):
                        with st.expander(f"Layer {idx + 1}: {layer.get('range', 'Unknown')}"):
                            col1, col2 = st.columns([3, 1])
                            with col1:
                                st.write("**Observation:**", layer.get('observation', 'N/A'))
                                st.write("**Soil Name:**", layer.get('soil_name', 'N/A'))
                                st.write("**Soi Colour:**",layer.get('soil_color', 'N/A') )
                            
                            with col2:
                                st.metric("Samples", len(layer.get('sample_test', [])))
                            
                            # Show sample data if available
                            if layer.get('sample_test'):
                                st.write("**Sample Test Results:**")
                                st.dataframe(layer['sample_test'], use_container_width=True)
            
            with tab3:
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    # Complete dataset
                    download_data = json.dumps(final_data, indent=2)
                    st.download_button(
                        label="📥 Download Complete Dataset (JSON)",
                        data=download_data,
                        file_name=f"geotechnical_data_{uploaded_file.name.replace('.pdf', '')}.json",
                        mime="application/json",
                        use_container_width=True
                    )
                    
                    # Summary statistics
                    st.markdown("##### 📊 Export Summary")
                    st.write(f"- Total Boreholes: {len(final_data)}")
                    st.write(f"- Total Data Size: {len(download_data) / 1024:.1f} KB")
                
                with col2:
                    # Individual borehole
                    export_borehole = st.selectbox(
                        "Export Individual Borehole",
                        options=list(final_data.keys()),
                        format_func=lambda x: f"Borehole {x}"
                    )
                    
                    if export_borehole:
                        individual_data = json.dumps(
                            final_data[export_borehole], indent=2
                        )
                        st.download_button(
                            label=f"📥 Download {export_borehole} Data",
                            data=individual_data,
                            file_name=f"borehole_{export_borehole}.json",
                            mime="application/json",
                            use_container_width=True
                        )
                
                # Preview
                with st.expander("👁️ Preview Export Data", expanded=False):
                    st.json(final_data)
    
    # Footer with instructions
    with st.expander("📖 How to Use", expanded=False):
        st.markdown("""
        1. **Upload PDF**: Select your geotechnical report containing boring logs and site maps
        2. **Automatic Processing**: The system will automatically:
           - Convert PDF pages to high-resolution images
           - Classify pages as maps, tables, or neither
           - Extract data from identified pages
           - Merge and structure the data
        3. **View Results**: Explore the data in Summary or Detailed view tabs
        4. **Export Data**: Download results as JSON for further analysis
        
        **Note**: Processing time depends on PDF size and complexity. Large files may take several minutes.
        """)

if __name__ == "__main__":
    main()