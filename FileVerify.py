import streamlit as st
import cv2
import numpy as np
import requests
from PIL import Image
import os
import pandas as pd
import base64
import json
from io import BytesIO
from dotenv import load_dotenv
import anthropic
import re
from difflib import SequenceMatcher
from datetime import datetime

# Page Config
st.set_page_config(page_title="Table Extraction Validator with AI Agent", layout="wide")

def prepare_image_from_upload(uploaded_file):
    """Converts and resizes an uploaded image file to a base64 encoded string."""
    try:
        image_bytes = uploaded_file.getvalue()
        nparr = np.frombuffer(image_bytes, np.uint8)
        img_cv = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        max_dim = 2048
        h, w, _ = img_cv.shape
        if h > max_dim or w > max_dim:
            if h > w:
                new_h, new_w = max_dim, int(w * (max_dim / h))
            else:
                new_h, new_w = int(h * (max_dim / w)), max_dim
            img_cv = cv2.resize(img_cv, (new_w, new_h))

        _, buffer = cv2.imencode('.png', img_cv)
        base64_image = base64.b64encode(buffer).decode('utf-8')
        return base64_image
    except Exception as e:
        st.error(f"Error preparing image: {e}")
        return None

def validate_with_ai_comparison(base64_image_data, extracted_df, api_key, model_name, provider):
    """Have AI directly compare the image with the CSV data to identify discrepancies."""
    
    # Convert DataFrame to a structured format for the AI
    csv_data = extracted_df.to_dict('records')
    headers = list(extracted_df.columns)
    
    prompt = f"""
    You are a data validation expert. I have an image of a table and extracted data from that table. 
    Your task is to compare the image with the extracted data and identify any discrepancies.

    EXTRACTED DATA (CSV):
    Headers: {headers}
    Data: {json.dumps(csv_data, indent=2)}

    IMPORTANT INSTRUCTIONS:
    - Look very carefully at the image and compare each cell with the CSV data
    - Be extremely precise - don't assume the CSV is wrong just because values look different
    - Consider that both the image reading AND the CSV extraction could have errors
    - Pay special attention to: digit recognition (0 vs O, 1 vs l, 5 vs S), decimal points, commas, spacing
    - Only report discrepancies when you are highly confident there's actually a difference

    For each discrepancy you find, analyze:
    1. What do you actually see in the image (be very specific)
    2. What's in the CSV data
    3. Which one is more likely to be correct based on context, formatting, and clarity
    4. Your confidence level in the discrepancy

    Return your analysis as a JSON object with this structure:
    {{
        "validation_results": [
            {{
                "row_index": 0,
                "column_name": "column_name",
                "issue_type": "number_mismatch|text_mismatch|missing_data|extra_data|unclear_image",
                "image_value": "what you actually see in the image (be specific)",
                "csv_value": "what's in the CSV",
                "likely_correct_value": "which value is more likely correct",
                "confidence": 0.9,
                "csv_likely_correct": true,
                "description": "Brief description of the issue and why you think one is more correct",
                "reasoning": "Detailed explanation of your analysis"
            }}
        ],
        "overall_accuracy": 0.95,
        "total_issues": 3,
        "summary": "Brief summary of findings"
    }}

    CRITICAL: 
    - Set "csv_likely_correct" to true if the CSV value appears more accurate than what you see in the image
    - Set "likely_correct_value" to whichever value you believe is actually correct
    - Only report issues where you're confident there's a real discrepancy (confidence > 0.7)
    - Consider image quality, OCR challenges, and context when making decisions
    """
    
    if provider == "Google Gemini":
        return validate_with_gemini(base64_image_data, api_key, model_name, prompt)
    else:
        return validate_with_claude(base64_image_data, api_key, model_name, prompt)

def get_smart_corrections(base64_image_data, issues, extracted_df, api_key, model_name, provider):
    """Get AI-powered smart corrections, considering that CSV might be correct."""
    
    if not issues:
        return {}
    
    # Separate issues where CSV is likely correct vs image is likely correct
    csv_correct_issues = [issue for issue in issues if issue.get('csv_likely_correct', False)]
    image_correct_issues = [issue for issue in issues if not issue.get('csv_likely_correct', False)]
    
    if not image_correct_issues:
        st.info("All identified issues suggest the CSV data is already correct. No corrections needed!")
        return {}
    
    # Prepare context for the AI
    csv_data = extracted_df.to_dict('records')
    headers = list(extracted_df.columns)
    
    issues_summary = []
    for issue in image_correct_issues:
        issues_summary.append({
            "row_index": issue.get('row_index'),
            "column_name": issue.get('column_name'),
            "current_csv_value": issue.get('csv_value'),
            "what_ai_sees_in_image": issue.get('image_value'),
            "likely_correct_value": issue.get('likely_correct_value'),
            "issue_type": issue.get('issue_type'),
            "original_reasoning": issue.get('reasoning', '')
        })
    
    prompt = f"""
    You are a data correction specialist. Based on previous analysis, the following issues have been identified where the IMAGE appears to contain the correct values and the CSV needs correction.
    
    CONTEXT:
    - Original CSV Data: {json.dumps(csv_data, indent=2)}
    - Headers: {headers}
    
    ISSUES TO CORRECT (where IMAGE is likely correct):
    {json.dumps(issues_summary, indent=2)}
    
    For each issue, look at the image again very carefully and provide the corrected CSV value. Consider:
    1. The previous analysis already determined the image value is more likely correct
    2. OCR reading accuracy from the image
    3. Context clues from surrounding data
    4. Formatting consistency
    
    IMPORTANT: Only correct values where you're confident the image reading is accurate.
    If you're unsure about what you see in the image, set confidence < 0.8.
    
    Return ONLY a JSON object with this exact structure:
    {{
        "corrections": [
            {{
                "row_index": 0,
                "column_name": "column_name",
                "corrected_value": "the exact value as seen in the image",
                "confidence": 0.95,
                "reasoning": "why this correction should be made based on image analysis"
            }}
        ]
    }}
    
    Be very precise with the corrected values. Match exactly what you see in the image.
    """
    
    if provider == "Google Gemini":
        result = validate_with_gemini(base64_image_data, api_key, model_name, prompt)
    else:
        result = validate_with_claude(base64_image_data, api_key, model_name, prompt)
    
    if result and 'corrections' in result:
        corrections_dict = {}
        for correction in result['corrections']:
            key = (correction.get('row_index'), correction.get('column_name'))
            corrections_dict[key] = {
                'corrected_value': correction.get('corrected_value'),
                'confidence': correction.get('confidence', 0.0),
                'reasoning': correction.get('reasoning', 'No reasoning provided')
            }
        return corrections_dict
    
    return {}

def apply_corrections_to_dataframe(df, corrections_dict):
    """Apply corrections to the dataframe and return the corrected version."""
    corrected_df = df.copy()
    correction_log = []
    
    for (row_idx, col_name), correction_info in corrections_dict.items():
        if row_idx < len(corrected_df) and col_name in corrected_df.columns:
            old_value = corrected_df.iloc[row_idx][col_name]
            new_value = correction_info['corrected_value']
            
            # Try to maintain data type
            if pd.api.types.is_numeric_dtype(corrected_df[col_name]):
                try:
                    new_value = pd.to_numeric(new_value)
                except:
                    pass  # Keep as string if conversion fails
            
            corrected_df.iloc[row_idx, corrected_df.columns.get_loc(col_name)] = new_value
            
            correction_log.append({
                'row': row_idx + 1,
                'column': col_name,
                'old_value': old_value,
                'new_value': new_value,
                'confidence': correction_info['confidence'],
                'reasoning': correction_info['reasoning']
            })
    
    return corrected_df, correction_log

def validate_with_gemini(base64_image_data, api_key, model_name, prompt):
    """Validate using Gemini API."""
    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}, {"inlineData": {"mimeType": "image/png", "data": base64_image_data}}]}],
        "generationConfig": {"responseMimeType": "application/json"}
    }
    api_url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"

    try:
        response = requests.post(api_url, headers={'Content-Type': 'application/json'}, json=payload, timeout=120)
        response.raise_for_status()
        result = response.json()
        json_response_text = result['candidates'][0]['content']['parts'][0]['text']
        parsed_json = json.loads(json_response_text)
        return parsed_json
    except Exception as e:
        st.error(f"Gemini validation error: {e}")
        return None

def validate_with_claude(base64_image_data, api_key, model_name, prompt):
    """Validate using Claude API."""
    try:
        client = anthropic.Anthropic(api_key=api_key)
        message = client.messages.create(
            model=model_name,
            max_tokens=4096,
            messages=[
                {"role": "user", "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": base64_image_data}},
                    {"type": "text", "text": prompt}
                ]}
            ],
        )
        json_response_text = message.content[0].text
        # Clean up the response if it contains markdown code blocks
        if "```json" in json_response_text:
            json_response_text = json_response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in json_response_text:
            json_response_text = json_response_text.split("```")[1].strip()
        
        parsed_json = json.loads(json_response_text)
        return parsed_json
    except Exception as e:
        st.error(f"Claude validation error: {e}")
        return None

def load_data_file(uploaded_file):
    """Load CSV or Excel file into a DataFrame with robust error handling."""
    try:
        # Reset file pointer to beginning
        uploaded_file.seek(0)
        
        if uploaded_file.name.endswith('.csv'):
            # Try different CSV reading approaches
            try:
                # First attempt: standard reading
                df = pd.read_csv(uploaded_file)
            except pd.errors.EmptyDataError:
                st.error("The CSV file appears to be empty.")
                return None
            except pd.errors.ParserError as e:
                st.error(f"CSV parsing error: {e}")
                # Try with different parameters
                uploaded_file.seek(0)
                try:
                    df = pd.read_csv(uploaded_file, header=None)
                    st.warning("Loaded CSV without headers. First row treated as data.")
                except Exception as e2:
                    st.error(f"Could not parse CSV file: {e2}")
                    return None
            except Exception as e:
                # Try reading without header if standard approach fails
                uploaded_file.seek(0)
                try:
                    df = pd.read_csv(uploaded_file, header=None)
                    st.warning("Loaded CSV without headers. First row treated as data.")
                except Exception as e2:
                    st.error(f"Could not read CSV file: {e}")
                    return None
                    
        elif uploaded_file.name.endswith(('.xlsx', '.xls')):
            try:
                df = pd.read_excel(uploaded_file)
            except Exception as e:
                st.error(f"Could not read Excel file: {e}")
                try:
                    # Try reading without header
                    df = pd.read_excel(uploaded_file, header=None)
                    st.warning("Loaded Excel without headers. First row treated as data.")
                except Exception as e2:
                    st.error(f"Could not parse Excel file: {e2}")
                    return None
        else:
            st.error("Unsupported file format. Please upload CSV or Excel file.")
            return None
        
        # Check if DataFrame is empty
        if df.empty:
            st.error("The uploaded file contains no data.")
            return None
            
        # Check if DataFrame has no columns
        if len(df.columns) == 0:
            st.error("The uploaded file has no columns.")
            return None
            
        # Handle cases where all columns are unnamed
        if all(col.startswith('Unnamed:') for col in df.columns):
            st.warning("File appears to have no proper headers. Using default column names.")
            df.columns = [f'Column_{i+1}' for i in range(len(df.columns))]
        
        # Remove completely empty rows and columns
        df = df.dropna(how='all').dropna(axis=1, how='all')
        
        if df.empty:
            st.error("The file contains no valid data after removing empty rows/columns.")
            return None
            
        return df
        
    except Exception as e:
        st.error(f"Unexpected error loading file: {e}")
        st.info("Please ensure your file is a valid CSV or Excel file with data.")
        return None

def main():
    load_dotenv()
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
    
    st.title("🔍 Table Extraction Validator with AI Correction Agent")
    st.markdown("Upload an image of a table and its extracted data (CSV/Excel) to identify extraction errors and get AI-powered corrections.")
    
    # Initialize session state
    if 'validation_results' not in st.session_state:
        st.session_state.validation_results = None
    if 'corrected_df' not in st.session_state:
        st.session_state.corrected_df = None
    if 'original_df' not in st.session_state:
        st.session_state.original_df = None
    if 'base64_image' not in st.session_state:
        st.session_state.base64_image = None
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        ai_provider = st.selectbox("Choose AI Provider:", ("Google Gemini", "Anthropic Claude"))
        
        if ai_provider == "Google Gemini":
            model_options = ["gemini-2.0-flash-exp", "gemini-2.0-flash", "gemini-1.5-pro"]
        else:
            model_options = ["claude-3-sonnet-20240229", "claude-3-haiku-20240307"]
        
        selected_model = st.selectbox("Choose AI Model:", options=model_options)
        
        st.header("🎯 Comparison Settings")
        show_confidence = st.checkbox("Show confidence scores", True)
        min_confidence = st.slider("Minimum confidence threshold", 0.0, 1.0, 0.7, 0.1)
        st.info("Only issues above this confidence level will be reported")
        
        st.header("🤖 Correction Agent Settings")
        auto_correct_confidence = st.slider("Auto-correct confidence threshold", 0.8, 1.0, 0.9, 0.05)
        st.info("Issues above this confidence will be auto-corrected")
    
    # File upload section
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📄 Upload Table Image")
        uploaded_image = st.file_uploader("Choose an image file", type=["png", "jpg", "jpeg"])
        
        if uploaded_image:
            st.image(uploaded_image, caption="Original Table Image", use_column_width=True)
            # Store base64 image in session state
            st.session_state.base64_image = prepare_image_from_upload(uploaded_image)
    
    with col2:
        st.subheader("📊 Upload Extracted Data")
        uploaded_data = st.file_uploader("Choose CSV or Excel file", type=["csv", "xlsx", "xls"])
        
        if uploaded_data:
            extracted_df = load_data_file(uploaded_data)
            if extracted_df is not None:
                st.write("**Extracted Data Preview:**")
                st.dataframe(extracted_df.head())
                
                # Store original dataframe in session state
                st.session_state.original_df = extracted_df.copy()
                
                # Show file info
                st.info(f"📄 Loaded file: {uploaded_data.name} | Shape: {extracted_df.shape[0]} rows × {extracted_df.shape[1]} columns")
            else:
                st.error("Could not load the uploaded file. Please check the file format and content.")
    
    # Main processing
    if uploaded_image and uploaded_data and st.session_state.original_df is not None:
        if st.button("🔍 Validate Extraction", type="primary"):
            # Get API key
            api_key = gemini_api_key if ai_provider == "Google Gemini" else anthropic_api_key
            
            if not api_key:
                st.error(f"Please add your {ai_provider} API key to the .env file.")
                return
            
            if not st.session_state.base64_image:
                st.error("Could not process the image. Please try uploading again.")
                return
            
            with st.spinner(f"AI is comparing the image with your CSV data using {ai_provider}..."):
                # Direct AI comparison
                validation_results = validate_with_ai_comparison(
                    st.session_state.base64_image, st.session_state.original_df, api_key, selected_model, ai_provider
                )
                
                if not validation_results:
                    st.error("Could not complete validation. Please try again.")
                    return
                
                # Store validation results in session state
                st.session_state.validation_results = validation_results
        
        # Display validation results if available
        if st.session_state.validation_results:
            validation_results = st.session_state.validation_results
            
            # Display results
            st.subheader("🔍 AI Validation Results")
            
            # Get issues and other data from validation results
            all_issues = validation_results.get('validation_results', [])
            overall_accuracy = validation_results.get('overall_accuracy', 0)
            summary = validation_results.get('summary', 'No summary provided')
            
            # Filter issues by confidence threshold
            issues = []
            if all_issues:
                if min_confidence > 0:
                    issues = [issue for issue in all_issues if issue.get('confidence', 0) >= min_confidence]
                    if len(issues) < len(all_issues):
                        st.info(f"Filtered out {len(all_issues) - len(issues)} low-confidence issues (below {min_confidence:.1%})")
                else:
                    issues = all_issues
            
            # Summary metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Overall Accuracy", f"{overall_accuracy:.1%}")
            with col2:
                st.metric("Issues Found", len(issues))
            with col3:
                total_cells = len(st.session_state.original_df) * len(st.session_state.original_df.columns)
                error_rate = (len(issues) / total_cells * 100) if total_cells > 0 else 0
                st.metric("Error Rate", f"{error_rate:.2f}%")
            
            # Summary
            st.info(f"**AI Summary:** {summary}")
            
            if not issues:
                st.success("🎉 No discrepancies found! The extracted data matches the image perfectly.")
            else:
                st.error(f"❌ Found {len(issues)} discrepancies:")
                
                # AI Correction Agent Section
                st.subheader("🤖 AI Correction Agent")
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("🚀 Get Smart Corrections", type="secondary"):
                        api_key = gemini_api_key if ai_provider == "Google Gemini" else anthropic_api_key
                        
                        with st.spinner("AI is analyzing the image to provide smart corrections..."):
                            corrections_dict = get_smart_corrections(
                                st.session_state.base64_image, issues, st.session_state.original_df, 
                                api_key, selected_model, ai_provider
                            )
                            
                            if corrections_dict:
                                corrected_df, correction_log = apply_corrections_to_dataframe(
                                    st.session_state.original_df, corrections_dict
                                )
                                st.session_state.corrected_df = corrected_df
                                
                                st.success(f"✅ AI has suggested corrections for {len(correction_log)} issues!")
                                
                                # Show correction log
                                st.write("**Correction Summary:**")
                                for log_entry in correction_log:
                                    confidence_emoji = "🟢" if log_entry['confidence'] > 0.9 else "🟡" if log_entry['confidence'] > 0.7 else "🔴"
                                    st.write(f"{confidence_emoji} Row {log_entry['row']}, {log_entry['column']}: `{log_entry['old_value']}` → `{log_entry['new_value']}` (Confidence: {log_entry['confidence']:.1%})")
                                    with st.expander(f"Reasoning for Row {log_entry['row']}, {log_entry['column']}"):
                                        st.write(log_entry['reasoning'])
                            else:
                                st.warning("Could not get corrections from AI. Please try again.")
                
                with col2:
                    if st.session_state.corrected_df is not None:
                        if st.button("📥 Download Corrected File", type="secondary"):
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            
                            if uploaded_data.name.endswith('.csv'):
                                csv_data = st.session_state.corrected_df.to_csv(index=False)
                                st.download_button(
                                    label="Download Corrected CSV",
                                    data=csv_data,
                                    file_name=f"corrected_{timestamp}.csv",
                                    mime="text/csv"
                                )
                            else:
                                # For Excel files, we'll provide CSV download
                                csv_data = st.session_state.corrected_df.to_csv(index=False)
                                st.download_button(
                                    label="Download Corrected Data (CSV)",
                                    data=csv_data,
                                    file_name=f"corrected_{timestamp}.csv",
                                    mime="text/csv"
                                )
                
                # Show comparison if corrected data exists
                if st.session_state.corrected_df is not None:
                    st.subheader("📊 Before vs After Comparison")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("**Original Data:**")
                        st.dataframe(st.session_state.original_df, use_container_width=True)
                    
                    with col2:
                        st.write("**Corrected Data:**")
                        st.dataframe(st.session_state.corrected_df, use_container_width=True)
                
                # Display detailed issues
                st.subheader("📋 Detailed Issues Analysis")
                
                # Separate issues by whether CSV or image is likely correct
                csv_correct_issues = [issue for issue in issues if issue.get('csv_likely_correct', False)]
                image_correct_issues = [issue for issue in issues if not issue.get('csv_likely_correct', False)]
                
                if csv_correct_issues:
                    st.info(f"✅ **{len(csv_correct_issues)} issues where your CSV data appears correct:**")
                    st.write("These are likely false positives or image reading errors.")
                    
                    for i, issue in enumerate(csv_correct_issues):
                        row_idx = issue.get('row_index', 'Unknown')
                        column_name = issue.get('column_name', 'Unknown')
                        image_value = issue.get('image_value', 'N/A')
                        csv_value = issue.get('csv_value', 'N/A')
                        confidence = issue.get('confidence', 0)
                        reasoning = issue.get('reasoning', 'No reasoning provided')
                        
                        with st.expander(f"✅ Row {row_idx + 1 if isinstance(row_idx, int) else row_idx}, Column '{column_name}' - CSV appears correct"):
                            col1, col2 = st.columns(2)
                            with col1:
                                st.write("**AI's Image Reading:**")
                                st.code(str(image_value))
                                st.caption("(Possibly incorrect)")
                            with col2:
                                st.write("**Your CSV Value:**")
                                st.code(str(csv_value))
                                st.caption("✅ (Likely correct)")
                            
                            if show_confidence:
                                st.write(f"**Confidence in discrepancy:** {confidence:.1%}")
                            st.write(f"**Analysis:** {reasoning}")
                
                if image_correct_issues:
                    st.warning(f"⚠️ **{len(image_correct_issues)} issues where the image appears more accurate:**")
                    st.write("These may need correction in your CSV data.")
                    
                    for i, issue in enumerate(image_correct_issues):
                        row_idx = issue.get('row_index', 'Unknown')
                        column_name = issue.get('column_name', 'Unknown')
                        image_value = issue.get('image_value', 'N/A')
                        csv_value = issue.get('csv_value', 'N/A')
                        likely_correct = issue.get('likely_correct_value', 'N/A')
                        confidence = issue.get('confidence', 0)
                        reasoning = issue.get('reasoning', 'No reasoning provided')
                        
                        with st.expander(f"⚠️ Row {row_idx + 1 if isinstance(row_idx, int) else row_idx}, Column '{column_name}' - Image appears more accurate"):
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.write("**Image Value:**")
                                st.code(str(image_value))
                                st.caption("✅ (Likely correct)")
                            with col2:
                                st.write("**CSV Value:**")
                                st.code(str(csv_value))
                                st.caption("(Needs correction)")
                            with col3:
                                st.write("**Recommended Value:**")
                                st.code(str(likely_correct))
                            
                            if show_confidence:
                                st.write(f"**Confidence:** {confidence:.1%}")
                            st.write(f"**Analysis:** {reasoning}")
                
                # Group remaining issues by type for backward compatibility
                other_issues = [issue for issue in issues if 'csv_likely_correct' not in issue]
                if other_issues:
                    st.subheader("📋 Other Issues")
                    issue_types = {}
                    for issue in other_issues:
                        issue_type = issue.get('issue_type', 'unknown')
                        if issue_type not in issue_types:
                            issue_types[issue_type] = []
                        issue_types[issue_type].append(issue)
                    
                    # Display issues by type
                    for issue_type, type_issues in issue_types.items():
                        st.write(f"**{issue_type.replace('_', ' ').title()} ({len(type_issues)} issues):**")
                        
                        for i, issue in enumerate(type_issues):
                            row_idx = issue.get('row_index', 'Unknown')
                            column_name = issue.get('column_name', 'Unknown')
                            image_value = issue.get('image_value', 'N/A')
                            csv_value = issue.get('csv_value', 'N/A')
                            confidence = issue.get('confidence', 0)
                            description = issue.get('description', 'No description')
                            suggested_correction = issue.get('suggested_correction', 'No suggestion')
                            
                            with st.expander(f"Issue {i+1}: Row {row_idx + 1 if isinstance(row_idx, int) else row_idx}, Column '{column_name}'"):
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.write("**Image Value:**")
                                    st.code(str(image_value))
                                with col2:
                                    st.write("**CSV Value:**")
                                    st.code(str(csv_value))
                                with col3:
                                    st.write("**AI Suggestion:**")
                                    st.code(str(suggested_correction))
                                
                                if show_confidence:
                                    st.write(f"**Confidence:** {confidence:.1%}")
                                st.write(f"**Description:** {description}")
                
                # Create downloadable error report with enhanced analysis
                error_report = []
                for issue in issues:
                    csv_likely_correct = issue.get('csv_likely_correct', False)
                    status = "CSV_CORRECT" if csv_likely_correct else "NEEDS_CORRECTION"
                    
                    error_report.append({
                        'Row': issue.get('row_index', 'Unknown') + 1 if isinstance(issue.get('row_index'), int) else issue.get('row_index', 'Unknown'),
                        'Column': issue.get('column_name', 'Unknown'),
                        'Status': status,
                        'Issue_Type': issue.get('issue_type', 'Unknown'),
                        'Image_Value': issue.get('image_value', 'N/A'),
                        'CSV_Value': issue.get('csv_value', 'N/A'),
                        'Recommended_Value': issue.get('likely_correct_value', issue.get('suggested_correction', 'N/A')),
                        'Confidence': issue.get('confidence', 0),
                        'Analysis': issue.get('reasoning', issue.get('description', 'No analysis'))
                    })
                
                if error_report:
                    error_df = pd.DataFrame(error_report)
                    csv = error_df.to_csv(index=False)
                    
                    st.download_button(
                        label="📥 Download Enhanced Validation Report",
                        data=csv,
                        file_name="enhanced_validation_report.csv",
                        mime="text/csv"
                    )

if __name__ == '__main__':
    main()