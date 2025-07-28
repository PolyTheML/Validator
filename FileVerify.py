# -*- coding: utf-8 -*-
"""
Streamlit Web App for Table Extraction Validation

This app compares an image of a table with extracted data (CSV/Excel) to identify
rows where the numbers don't match. It uses AI vision models to directly compare
the image with the CSV data.

To run this app:
1. Install required libraries:
   pip install streamlit opencv-python numpy pandas requests Pillow python-dotenv openpyxl anthropic
2. Set up your .env file with GEMINI_API_KEY and/or ANTHROPIC_API_KEY
3. Run: streamlit run table_validator.py
"""

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

# Page Config
st.set_page_config(page_title="Table Extraction Validator", layout="wide")

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

    Please analyze the image and compare it with the provided extracted data. Look for:
    1. Numbers that don't match (pay attention to digits, decimal places, commas)
    2. Text that's been misread or misspelled
    3. Missing or extra data
    4. Wrong values in cells

    Return your analysis as a JSON object with this structure:
    {{
        "validation_results": [
            {{
                "row_index": 0,
                "column_name": "column_name",
                "issue_type": "number_mismatch|text_mismatch|missing_data|extra_data",
                "image_value": "what you see in the image",
                "csv_value": "what's in the CSV",
                "confidence": 0.9,
                "description": "Brief description of the issue"
            }}
        ],
        "overall_accuracy": 0.95,
        "total_issues": 3,
        "summary": "Brief summary of findings"
    }}

    If no discrepancies are found, return an empty validation_results array.
    Be very thorough and precise in your comparison.
    """
    
    if provider == "Google Gemini":
        return validate_with_gemini(base64_image_data, api_key, model_name, prompt)
    else:
        return validate_with_claude(base64_image_data, api_key, model_name, prompt)

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
    
    st.title("🔍 Table Extraction Validator")
    st.markdown("Upload an image of a table and its extracted data (CSV/Excel) to identify extraction errors.")
    
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
    
    # File upload section
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📄 Upload Table Image")
        uploaded_image = st.file_uploader("Choose an image file", type=["png", "jpg", "jpeg"])
        
        if uploaded_image:
            st.image(uploaded_image, caption="Original Table Image", use_column_width=True)
    
    with col2:
        st.subheader("📊 Upload Extracted Data")
        uploaded_data = st.file_uploader("Choose CSV or Excel file", type=["csv", "xlsx", "xls"])
        
        if uploaded_data:
            extracted_df = load_data_file(uploaded_data)
            if extracted_df is not None:
                st.write("**Extracted Data Preview:**")
                st.dataframe(extracted_df.head())
                
                # Show file info
                st.info(f"📄 Loaded file: {uploaded_data.name} | Shape: {extracted_df.shape[0]} rows × {extracted_df.shape[1]} columns")
            else:
                st.error("Could not load the uploaded file. Please check the file format and content.")
    
    # Main processing
    if uploaded_image and uploaded_data:
        if st.button("🔍 Validate Extraction", type="primary"):
            # Get API key
            api_key = gemini_api_key if ai_provider == "Google Gemini" else anthropic_api_key
            
            if not api_key:
                st.error(f"Please add your {ai_provider} API key to the .env file.")
                return
            
            extracted_df = load_data_file(uploaded_data)
            if extracted_df is None:
                return
            
            with st.spinner(f"AI is comparing the image with your CSV data using {ai_provider}..."):
                # Prepare image
                base64_image = prepare_image_from_upload(uploaded_image)
                if not base64_image:
                    return
                
                # Direct AI comparison
                validation_results = validate_with_ai_comparison(
                    base64_image, extracted_df, api_key, selected_model, ai_provider
                )
                
                if not validation_results:
                    st.error("Could not complete validation. Please try again.")
                    return
                
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
                    total_cells = len(extracted_df) * len(extracted_df.columns)
                    error_rate = (len(issues) / total_cells * 100) if total_cells > 0 else 0
                    st.metric("Error Rate", f"{error_rate:.2f}%")
                
                # Summary
                st.info(f"**AI Summary:** {summary}")
                
                if not issues:
                    st.success("🎉 No discrepancies found! The extracted data matches the image perfectly.")
                else:
                    st.error(f"❌ Found {len(issues)} discrepancies:")
                    
                    # Group issues by type
                    issue_types = {}
                    for issue in issues:
                        issue_type = issue.get('issue_type', 'unknown')
                        if issue_type not in issue_types:
                            issue_types[issue_type] = []
                        issue_types[issue_type].append(issue)
                    
                    # Display issues by type
                    for issue_type, type_issues in issue_types.items():
                        st.subheader(f"📋 {issue_type.replace('_', ' ').title()} ({len(type_issues)} issues)")
                        
                        for i, issue in enumerate(type_issues):
                            row_idx = issue.get('row_index', 'Unknown')
                            column_name = issue.get('column_name', 'Unknown')
                            image_value = issue.get('image_value', 'N/A')
                            csv_value = issue.get('csv_value', 'N/A')
                            confidence = issue.get('confidence', 0)
                            description = issue.get('description', 'No description')
                            
                            with st.expander(f"Issue {i+1}: Row {row_idx + 1 if isinstance(row_idx, int) else row_idx}, Column '{column_name}'"):
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.write("**What AI sees in image:**")
                                    st.code(str(image_value))
                                with col2:
                                    st.write("**What's in your CSV:**")
                                    st.code(str(csv_value))
                                
                                if show_confidence:
                                    st.write(f"**Confidence:** {confidence:.1%}")
                                st.write(f"**Description:** {description}")
                    
                    # Create downloadable error report
                    error_report = []
                    for issue in issues:
                        error_report.append({
                            'Row': issue.get('row_index', 'Unknown') + 1 if isinstance(issue.get('row_index'), int) else issue.get('row_index', 'Unknown'),
                            'Column': issue.get('column_name', 'Unknown'),
                            'Issue_Type': issue.get('issue_type', 'Unknown'),
                            'Image_Value': issue.get('image_value', 'N/A'),
                            'CSV_Value': issue.get('csv_value', 'N/A'),
                            'Confidence': issue.get('confidence', 0),
                            'Description': issue.get('description', 'No description')
                        })
                    
                    if error_report:
                        error_df = pd.DataFrame(error_report)
                        csv = error_df.to_csv(index=False)
                        
                        st.download_button(
                            label="📥 Download Detailed Error Report",
                            data=csv,
                            file_name="ai_validation_report.csv",
                            mime="text/csv"
                        )
                
                # Show the data for reference
                with st.expander("📊 View Your Extracted Data"):
                    st.dataframe(extracted_df, use_container_width=True)

if __name__ == '__main__':
    main()