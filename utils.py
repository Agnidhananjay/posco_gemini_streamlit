import os
import fitz  # PyMuPDF
from concurrent.futures import ThreadPoolExecutor
import PIL.Image
import os
from dotenv import load_dotenv
from google import genai
import time
import json
import asyncio
from google.genai import types
# def classify_images(file_paths, prompt, client=None, api_key=None):
#     """
#     Classify images as either 'map', 'table', or 'neither' using Google Generative AI.
    
#     Args:
#         file_paths: List of file paths to images
#         prompt: Prompt text for classification
#         client: Optional Google GenAI client instance
#         api_key: Optional API key (if not provided, loads from .env file)
        
#     Returns:
#         dict: Dictionary with 'map', 'table', and 'neither' keys containing lists of file paths
#     """
    
#     # Load environment variables from .env file
#     load_dotenv()
    
#     # Create client if not provided
#     if client is None:
#         if api_key is None:
#             api_key = os.getenv('GEMINI_API_KEY')
#             if api_key is None:
#                 raise ValueError("GEMINI_API_KEY not found in environment variables or .env file")
        
#         client = genai.Client(api_key=api_key)
    
#     images = {"map": [], "table": [], "neither": []}
#     contents = [prompt]
#     table_found = False

#     for path in file_paths:
#         if table_found:
#             images["table"].append(path)
#             continue

#         # Open and process the image
#         image = PIL.Image.open(path)
#         contents_with_image = contents + [image]

#         # Generate content using the model
#         response = client.models.generate_content(
#             model="gemini-2.0-flash",
#             contents=contents_with_image
#         )

#         # Classify based on response text
#         if "map" in response.text.lower():
#             images["map"].append(path)
#         elif "table" in response.text.lower():
#             images["table"].append(path)
#             table_found = True
#         else:
#             images["neither"].append(path)

#     return images


# Example usage:
# 
# Method 1: Let function handle everything (requires GEMINI_API_KEY in .env file)
# images = classify_images(file_paths, prompt)
# # Returns: {"map": [...], "table": [...], "neither": [...]}
#
# Method 2: Provide your own client
# from google import genai
# client = genai.Client(api_key="your-api-key")
# images = classify_images(file_paths, prompt, client=client)
#
# Method 3: Provide API key directly
# images = classify_images(file_paths, prompt, api_key="your-api-key")

# Required dependencies:
# pip install google-generativeai python-dotenv pillow
import os
import fitz
from concurrent.futures import ThreadPoolExecutor

async def classify_images_async(file_paths, prompt, client=None, api_key=None, max_concurrent=6):
    """
    Classify images as either 'map', 'table', or 'neither' using Google Generative AI.
    Processes images concurrently with up to 6 simultaneous API calls using semaphore.
    
    Args:
        file_paths: List of file paths to images
        prompt: Prompt text for classification
        client: Optional Google GenAI client instance
        api_key: Optional API key (if not provided, loads from .env file)
        max_concurrent: Maximum number of concurrent API calls (default: 6)
        
    Returns:
        dict: Dictionary with 'map', 'table', and 'neither' keys containing lists of file paths
    """
    
    # Load environment variables from .env file
    load_dotenv()
    
    # Create client if not provided
    if client is None:
        if api_key is None:
            api_key = os.getenv('GEMINI_API_KEY')
            if api_key is None:
                raise ValueError("GEMINI_API_KEY not found in environment variables or .env file")
        
        from google import genai
        client = genai.Client(api_key=api_key)
    
    images = {"map": [], "table": [], "neither": []}
    
    async def classify_single_image(path, index, semaphore):
        """Classify a single image asynchronously"""
        async with semaphore:
            try:
                # Open and process the image
                image = PIL.Image.open(path)
                contents_with_image = [prompt, image]
                
                # Run API call in thread pool (since genai client is sync)
                loop = asyncio.get_event_loop()
                response = await loop.run_in_executor(
                    None,
                    lambda: client.models.generate_content(
                        model="gemini-2.5-flash",
                        contents=contents_with_image
                    )
                )
                
                # Classify based on response text
                response_lower = response.text.lower()
                if "map" in response_lower:
                    return (path, 'map', index)
                elif "table" in response_lower:
                    return (path, 'table', index)
                else:
                    return (path, 'neither', index)
                    
            except Exception as e:
                print(f"Error classifying {path}: {e}")
                return (path, 'neither', index)
    
    # Create semaphore to limit concurrent requests
    semaphore = asyncio.Semaphore(max_concurrent)
    
    # Create tasks for all images
    tasks = [
        classify_single_image(path, i, semaphore) 
        for i, path in enumerate(file_paths)
    ]
    
    # Process all images concurrently
    results = await asyncio.gather(*tasks)
    
    # Sort results by index to maintain original order
    results.sort(key=lambda x: x[2])
    
    # Assign classifications
    for path, classification, index in results:
        images[classification].append(path)
    
    return images


def classify_images(file_paths, prompt, client=None, api_key=None, max_concurrent=6):
    """
    Synchronous wrapper for classify_images_async.
    
    Args:
        file_paths: List of file paths to images
        prompt: Prompt text for classification
        client: Optional Google GenAI client instance
        api_key: Optional API key (if not provided, loads from .env file)
        max_concurrent: Maximum number of concurrent API calls (default: 6)
        
    Returns:
        dict: Dictionary with 'map', 'table', and 'neither' keys containing lists of file paths
    """
    return asyncio.run(classify_images_async(file_paths, prompt, client, api_key, max_concurrent))

def process_page(pdf_path, page_number, scale, base_name, output_dir):
    """Process a single page by opening the PDF in this thread for thread safety"""
    # Open PDF in this thread to avoid threading issues
    doc = fitz.open(pdf_path)
    try:
        page = doc[page_number]
        
        # Render the page to an image (pixmap) using the transformation matrix
        matrix = fitz.Matrix(scale, scale)
        pix = page.get_pixmap(matrix=matrix)

        # Define the output image path
        image_filename = f"{base_name}_page_{page_number + 1}.jpg"
        image_path = os.path.join(output_dir, image_filename)

        # Save the image in JPG format
        pix.save(image_path)
        return image_path
    finally:
        doc.close()

def pdf_to_images_2(pdf_path, output_dir, fixed_width=3000, max_workers=4):
    """
    Convert PDF pages to images with fixed width, maintaining aspect ratio.
    
    Args:
        pdf_path: Path to the PDF file
        output_dir: Directory to save images
        fixed_width: Fixed width for output images (height will scale proportionally)
        max_workers: Number of parallel workers
    
    Returns:
        tuple: (output_dir, list of image file paths)
    """
    # Ensure the PDF file exists
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"The file {pdf_path} does not exist.")

    # Extract the base file name (without extension) for directory naming
    base_name = os.path.splitext(os.path.basename(pdf_path))[0]

    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Open the PDF file to get page count and calculate scale
    try:
        doc = fitz.open(pdf_path)
        page_count = len(doc)
        
        # Get the original width of the first page to calculate scale
        original_width = doc[0].rect.width
        
        # Calculate the scaling factor to achieve the fixed width
        scale = fixed_width / original_width
        
        doc.close()
    except Exception as e:
        raise RuntimeError(f"Failed to open PDF: {e}")

    # Use ThreadPoolExecutor to process pages concurrently
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(process_page, pdf_path, page_number, scale, base_name, output_dir)
            for page_number in range(page_count)
        ]

        # Collect results in order
        file_paths = [future.result() for future in futures]

    print(f"PDF converted to images with fixed width {fixed_width}px and saved to: {output_dir}")
    return output_dir, file_paths

async def process_engineering_images_fast(images, client, prompt_map, prompt_table, borehole_schema, extracted_schema, max_concurrent=5):
    """
    Process engineering images in parallel to extract map and table data using Google Generative AI.
    
    Args:
        images: Dictionary with 'map' and 'table' keys containing lists of image paths
        client: Google GenAI client instance
        prompt_map: Prompt for processing map images
        prompt_table: Prompt for processing table images
        borehole_schema: Schema for borehole data extraction from maps
        extracted_schema: Schema for data extraction from tables
        max_concurrent: Maximum number of concurrent API calls (default: 5)
        
    Returns:
        tuple: (table_data, map_data) - Processed data from tables and maps
    """
    
    async def process_single_map(path):
        """Process a single map image"""
        contents = [prompt_map]
        image = PIL.Image.open(path)
        contents.append(image)

        # Run the API call in a thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor() as executor:
            response = await loop.run_in_executor(
                executor,
                lambda: client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=contents,
                    config={
                        'response_mime_type': 'application/json',
                        'response_schema': borehole_schema,
                    },
                )
            )
        return json.loads(response.text)
    
    async def process_single_table(path):
        """Process a single table image"""
        contents = [prompt_table]
        image = PIL.Image.open(path)
        contents.append(image)
        
        # Run the API call in a thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor() as executor:
            response = await loop.run_in_executor(
                executor,
                lambda: client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=contents,
                    config={
                        'response_mime_type': 'application/json',
                        'response_schema': extracted_schema,
                    },
                )
            )
        return json.loads(response.text), path
    
    # Create semaphore to limit concurrent requests
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def limited_process_map(path):
        async with semaphore:
            return await process_single_map(path)
    
    async def limited_process_table(path):
        async with semaphore:
            return await process_single_table(path)
    
    # Process all map images in parallel
    map_tasks = [limited_process_map(path) for path in images["map"]]
    map_results = await asyncio.gather(*map_tasks, return_exceptions=True)
    
    # Process all table images in parallel
    table_tasks = [limited_process_table(path) for path in images["table"]]
    table_results = await asyncio.gather(*table_tasks, return_exceptions=True)
    
    # Process map data (same logic as before)
    from collections import OrderedDict

    map_data = OrderedDict()

    for result in map_results:
        if isinstance(result, Exception):
            continue
        for item in result.get("metadata", []):
            name = item.get("Name")
            number = item.get("Number")
            if name and isinstance(number, int):
                map_data[name] = item  # Will overwrite duplicates with the latest one

    # If needed as a list
    map_data = list(map_data.values())
    print(f"Map data extracted: {map_data}")
    # Process table data with merging logic
    table_data = []
    for result in table_results:
        if not isinstance(result, Exception):
            response, path = result
            
            # Same merging logic as original
            if len(table_data) > 0:
                if table_data[-1]["metadata"]['HOLE_NO'] == response["metadata"]['HOLE_NO']:
                    table_data[-1]["sample_data"].extend(response["sample_data"])
                    table_data[-1]["soil_data"].extend(response["soil_data"])
                else:
                    table_data.append(response)
            else:
                table_data.append(response)
    
    return table_data, map_data


def process_engineering_images(images, client, prompt_map, prompt_table, borehole_schema, extracted_schema, max_concurrent=5):
    """
    Synchronous wrapper for the async parallel processing function.
    
    Args:
        images: Dictionary with 'map' and 'table' keys containing lists of image paths
        client: Google GenAI client instance
        prompt_map: Prompt for processing map images
        prompt_table: Prompt for processing table images
        borehole_schema: Schema for borehole data extraction from maps
        extracted_schema: Schema for data extraction from tables
        max_concurrent: Maximum number of concurrent API calls (default: 5)
        
    Returns:
        tuple: (table_data, map_data) - Processed data from tables and maps
    """
    return asyncio.run(
        process_engineering_images_fast(
            images, client, prompt_map, prompt_table, 
            borehole_schema, extracted_schema, max_concurrent
        )
    )


# Example usage:
# 
# # Basic usage (same interface as before, but much faster!)
# table_data, map_data = process_engineering_images(
#     images, client, prompt_map, prompt_table, borehole_schema, extracted_schema
# )
#
# # Control concurrency to avoid rate limits
# table_data, map_data = process_engineering_images(
#     images, client, prompt_map, prompt_table, borehole_schema, extracted_schema,
#     max_concurrent=3  # Reduce if hitting rate limits
# )
#
# # For async environments, use the async version directly:
# table_data, map_data = await process_engineering_images_fast(
#     images, client, prompt_map, prompt_table, borehole_schema, extracted_schema
# )

# Performance improvement: Processes all images in parallel instead of sequentially
# Speed increase depends on number of images - typically 3-10x faster!

# Required dependencies:
# pip install google-generativeai pillow

def merge_engineering_data(table_data, map_data):
    """
    Merge table and map data, then combine samples with soil data based on depth ranges.
    Handles overlapping soil layers by removing redundant ranges.
    Merges consecutive layers with same soil name and color.
    
    Args:
        table_data: List of table data from drill logs
        map_data: List of map data from boring location maps
        
    Returns:
        dict: Merged data with samples integrated into soil layers by depth
    """
    
    def convert_to_float(range_str):
        """Convert range string like '5.0m' to float"""
        if range_str.lower() == 'null' or not range_str.strip():
            return None
        return float(range_str.replace('m', '').strip())
    
    def parse_range(range_str):
        """Parse range string and return (start, end) tuple"""
        try:
            if '~' in range_str:
                range_start, range_end = map(convert_to_float, range_str.split('~'))
            else:
                range_start = range_end = convert_to_float(range_str)
            return (range_start, range_end)
        except:
            return (None, None)
    
    def remove_strikethrough(text):
        """Remove strikethrough text from observations"""
        if not text:
            return text
        
        import re
        
        # Remove HTML strikethrough tags
        text = re.sub(r'<strike>.*?</strike>', '', text, flags=re.IGNORECASE)
        text = re.sub(r'<s>.*?</s>', '', text, flags=re.IGNORECASE)
        text = re.sub(r'<del>.*?</del>', '', text, flags=re.IGNORECASE)
        
        # Remove markdown strikethrough (~~text~~)
        text = re.sub(r'~~.*?~~', '', text)
        
        # Remove Unicode strikethrough characters (U+0336)
        text = re.sub(r'[^\u0336]+\u0336', '', text)
        
        # Clean up extra spaces
        text = ' '.join(text.split())
        
        return text.strip()
    
    def merge_similar_layers(soil_layers):
        """Merge consecutive soil layers with same name and color"""
        if not soil_layers:
            return soil_layers
        
        # First, sort layers by depth
        layers_with_ranges = []
        for layer in soil_layers:
            start, end = parse_range(layer.get('range', ''))
            if start is not None and end is not None:
                layers_with_ranges.append({
                    'layer': layer,
                    'start': start,
                    'end': end
                })
        
        # Sort by start position
        layers_with_ranges.sort(key=lambda x: x['start'])
        
        # Merge consecutive layers with same properties
        merged_layers = []
        i = 0
        
        while i < len(layers_with_ranges):
            current = layers_with_ranges[i]
            current_layer = current['layer']
            
            # Initialize merged layer
            merged_start = current['start']
            merged_end = current['end']
            # Clean observation before adding
            cleaned_obs = remove_strikethrough(current_layer.get('observation', ''))
            merged_observations = [cleaned_obs] if cleaned_obs else []
            
            # Look for consecutive layers with same name and color
            j = i + 1
            while j < len(layers_with_ranges):
                next_layer = layers_with_ranges[j]
                next_layer_data = next_layer['layer']
                
                # Check if layers have same soil_name and soil_color
                same_name = current_layer.get('soil_name', '').strip().lower() == next_layer_data.get('soil_name', '').strip().lower()
                same_color = current_layer.get('soil_color', '').strip().lower() == next_layer_data.get('soil_color', '').strip().lower()
                
                # Check if layers are consecutive (next starts where current ends)
                is_consecutive = abs(merged_end - next_layer['start']) < 0.1  # Small tolerance for float comparison
                
                if same_name and same_color and is_consecutive:
                    # Merge this layer
                    merged_end = next_layer['end']
                    # Clean observation before adding
                    cleaned_obs = remove_strikethrough(next_layer_data.get('observation', ''))
                    if cleaned_obs and cleaned_obs not in merged_observations:
                        merged_observations.append(cleaned_obs)
                    j += 1
                else:
                    break
            
            # Create merged layer
            merged_layer = current_layer.copy()
            merged_layer['range'] = f"{merged_start}~{merged_end}m"
            
            # Combine observations
            merged_layer['observation'] = "; ".join(merged_observations) if merged_observations else ""
            
            merged_layers.append(merged_layer)
            i = j
        
        return merged_layers
    
    def remove_overlapping_layers(soil_layers):
        """Remove soil layers that are completely contained within other layers"""
        if not soil_layers:
            return soil_layers
        
        # Parse all ranges
        layers_with_ranges = []
        for layer in soil_layers:
            start, end = parse_range(layer.get('range', ''))
            if start is not None and end is not None:
                layers_with_ranges.append({
                    'layer': layer,
                    'start': start,
                    'end': end,
                    'range_size': end - start
                })
        
        # Sort by range size (larger ranges first) and then by start position
        layers_with_ranges.sort(key=lambda x: (-x['range_size'], x['start']))
        
        # Keep track of non-overlapping layers
        filtered_layers = []
        
        for i, current in enumerate(layers_with_ranges):
            is_contained = False
            
            # Check if current layer is contained within any previously accepted layer
            for accepted in filtered_layers:
                # Check if current layer is completely contained within accepted layer
                if (current['start'] >= accepted['start'] and 
                    current['end'] <= accepted['end'] and
                    current != accepted):  # Don't compare with itself
                    is_contained = True
                    break
            
            # If not contained in any accepted layer, keep it
            if not is_contained:
                filtered_layers.append(current)
        
        # Sort filtered layers by start position for consistent ordering
        filtered_layers.sort(key=lambda x: x['start'])
        
        # Return only the layer data
        return [item['layer'] for item in filtered_layers]
    
    # Fix overlapping layers and merge similar layers in table_data before processing
    for table_entry in table_data:
        if 'soil_data' in table_entry:
            # Clean strikethrough text from all observations first
            for layer in table_entry['soil_data']:
                if 'observation' in layer:
                    layer['observation'] = remove_strikethrough(layer['observation'])
            
            # First merge similar consecutive layers
            table_entry['soil_data'] = merge_similar_layers(table_entry['soil_data'])
            # Then remove overlapping layers
            table_entry['soil_data'] = remove_overlapping_layers(table_entry['soil_data'])
    
    # Step 1: Create initial merged structure
    data_extracted = {}
    
    if len(map_data) > 0:
        for b in table_data:
            if b["metadata"]['HOLE_NO'] in [a['Name'] for a in map_data]:
                a = next((x for x in map_data if x['Name'] == b["metadata"]['HOLE_NO']), None)
                if a:
                    b['map_data'] = {
                        'Name': a['Name'],
                        'Number': a['Number'],
                        'Elevation_level': a['Elevation_level']
                    }
                    data_extracted[a['Name']] = b
            else:
                for b in table_data:
                    data_extracted[b["metadata"]['HOLE_NO']] = b
    else:
        for b in table_data:
            data_extracted[b["metadata"]['HOLE_NO']] = b
    
    # Step 2: Merge samples with soil data based on depth ranges
    merged_data = {}

    # Iterate over all boreholes in the data
    for bh_id, bh_data in data_extracted.items():
        merged_soil_data = []

        # Add metadata for each borehole
        borehole_metadata = bh_data.get('metadata', {})
        Elevation_level = borehole_metadata.get('Elevation_level', None)
        borehole_mapdata = bh_data.get('map_data', {})

        # Iterate over soil layers
        for soil_layer in bh_data['soil_data']:
            range_str = soil_layer['range']
            try:
                # Parse the range
                range_start, range_end = parse_range(range_str)

                # If either range_start or range_end is None, skip this soil layer
                if range_start is None or range_end is None:
                    continue

                # Create a list to hold samples within the current range
                sample_data_within_range = []

                # Iterate over sample data and check if depth falls within the range
                for sample in bh_data['sample_data']:
                    sample["Elevation_level"] = Elevation_level - sample['Depth'] if Elevation_level else None
                    if range_start < sample['Depth'] <= range_end:
                        sample_data_within_range.append(sample)

                # Add samples to soil layer (even if empty list)
                soil_layer['sample_test'] = sample_data_within_range
                merged_soil_data.append(soil_layer)
                
            except Exception as e:
                # Skip problematic soil layers
                print(f"Error processing soil layer: {e}")
                continue

        # Save the merged data for this borehole, including metadata
        merged_data[bh_id] = {
            'metadata': borehole_metadata,
            'soil_data': merged_soil_data,
            'map_data': borehole_mapdata
        }

    return merged_data


# Example usage:
# 
# # After processing your images
# table_data, map_data = process_engineering_images(...)
# 
# # Merge the data
# final_data = merge_engineering_data(table_data, map_data)
# 
# # Access merged data
# for borehole_id, data in final_data.items():
#     print(f"Borehole: {borehole_id}")
#     print(f"Metadata: {data['metadata']}")
#     for soil_layer in data['soil_data']:
#         print(f"Soil range: {soil_layer['range']}")
#         print(f"Samples in range: {len(soil_layer['sample_test'])}")
# from ultralytics import YOLO

# def classify_images_yolo(file_paths, model_path, progress_callback=None):
#     """
#     Classify images using YOLO model optimized for Streamlit Cloud.
#     Processes one image at a time with 1080px size.
    
#     Args:
#         file_paths: List of file paths to images
#         model_path: Path to trained YOLO model (.pt file)
#         progress_callback: Optional Streamlit progress bar object
        
#     Returns:
#         dict: Dictionary with 'map', 'table', and 'neither' keys containing lists of file paths
#     """
    
#     # Load model once
#     model = YOLO(model_path)
#     model.to('cpu')
    
#     # Initialize results dictionary
#     images = {"map": [], "table": [], "neither": []}
#     total_images = len(file_paths)
    
#     # Process images one by one
#     for i, image_path in enumerate(file_paths):
#         try:
#             # Run prediction on single image
#             results = model(
#                 image_path,
#                 imgsz=1080,        # Image size as requested
#                 device='cpu',      # Force CPU for Streamlit Cloud
#                 verbose=False,     # Reduce console output
#                 max_det=1,         # Only need top detection
#                 conf=0.25          # Confidence threshold
#             )
            
#             # Get prediction
#             result = results[0]
#             top1_idx = result.probs.top1
#             predicted_class = result.names[top1_idx].lower()
#             confidence = result.probs.top1conf.item()
            
#             # Map YOLO class to categories (adjust based on your model's classes)
#             if predicted_class in ['map', 'site_map', 'boring_map', 'location_map']:
#                 images['map'].append(image_path)
#             elif predicted_class in ['table', 'drill_log', 'boring_log', 'data_table']:
#                 images['table'].append(image_path)
#             else:
#                 images['neither'].append(image_path)
                
#         except Exception as e:
#             print(f"Error classifying {image_path}: {e}")
#             images['neither'].append(image_path)
        
#         # Update progress if callback provided
#         if progress_callback:
#             progress = (i + 1) / total_images
#             progress_callback.progress(int(progress * 100))
    
#     return images
def calculate_thickness(range_str):
    """
    Calculate thickness from a range string.
    Expected formats:
    - "0.0 - 2.5"
    - "2.5 - 5.0"
    - "0.0-2.5" (without spaces)
    - "2.5-5.0" 
    """
    if not range_str or range_str == 'N/A' or range_str == 'Unknown':
        return 'N/A'
    
    try:
        # Clean the string and split by dash
        range_clean = str(range_str).replace(' ', '')
        
        # Try different splitting patterns
        if '-' in range_clean:
            parts = range_clean.split('-')
            if len(parts) == 2:
                start = float(parts[0])
                end = float(parts[1])
                thickness = abs(end - start)
                return f"{thickness:.1f}"
        
        # If splitting by dash doesn't work, try other patterns
        # Handle formats like "from 0.0 to 2.5"
        import re
        numbers = re.findall(r'\d+\.?\d*', range_str)
        if len(numbers) >= 2:
            start = float(numbers[0])
            end = float(numbers[1])
            thickness = abs(end - start)
            return f"{thickness:.1f}"
        
        return 'N/A'
    
    except (ValueError, IndexError, AttributeError):
        return 'N/A'