import pandas as pd
import requests
import json
import time
import os
from pathlib import Path
import csv
from datetime import datetime

class IndustryRatingExtractor:
    def __init__(self, model="qwen2.5:7b-instruct"):
        self.url = "http://localhost:11434/api/generate"
        self.model = model
        self.output_file = "industry_ratings_output.csv"
        self.progress_file = "extraction_progress.json"
        
        # Load the variables template
        self.variables_df = pd.read_csv("carbon_intensity_variables.csv")
        self.variables = self._prepare_variables_template()
        
    def _prepare_variables_template(self):
        """Convert variables CSV to a structured template"""
        variables = {}
        for _, row in self.variables_df.iterrows():
            variables[row['variable']] = {
                'description': row['description'],
                'scale_1': row['scale_1'],
                'scale_1_examples': row['scale_1_examples'],  # Add this line
                'scale_4': row['scale_4'],
                'scale_4_examples': row['scale_4_examples'],  # Add this line
                'scale_7': row['scale_7'],
                'scale_7_examples': row['scale_7_examples'],  # Add this line
                'category': row['category']
            }
        return variables
    
    def _create_extraction_prompt(self, industry_code, industry_name, industry_description):
        """Create a prompt for extracting ratings using the industry description"""
        
        # Format the variables for the prompt
        variables_text = ""
        for var_name, var_info in self.variables.items():
            variables_text += f"""
    {var_name}:
    - Description: {var_info['description']}
    - Scale 1: {var_info['scale_1']}
    Examples: {var_info['scale_1_examples']}
    - Scale 4: {var_info['scale_4']}
    Examples: {var_info['scale_4_examples']}
    - Scale 7: {var_info['scale_7']}
    Examples: {var_info['scale_7_examples']}
    - Category: {var_info['category']}
    """
        
        prompt = f"""You are an expert analyst extracting carbon intensity ratings for industries.

    INDUSTRY CODE: {industry_code}
    INDUSTRY NAME: {industry_name}

    INDUSTRY DESCRIPTION:
    {industry_description}

    RATING SCALE DEFINITIONS:
    {variables_text}

    TASK:
    For each variable above, rate this industry on a scale of 1-7 based SOLELY on the provided industry description.
    Base your ratings ONLY on evidence found in the description text.

    IMPORTANT RULES:
    1. Only use ratings 1, 2, 3, 4, 5, 6, or 7 (whole numbers only, no decimals)
    2. Provide a brief 1-sentence justification for each rating, citing specific evidence from the description
    3. Return ONLY valid JSON without any other text

    OUTPUT FORMAT:
    {{
        "industry_code": "{industry_code}",
        "industry_name": "{industry_name}",
        "ratings": {{
            "process_emission_intensity": {{"score": 4, "justification": "Brief reason based on description"}},
            "material_processing_depth": {{"score": 3, "justification": "Brief reason based on description"}},
            "thermal_process_intensity": {{"score": 5, "justification": "Brief reason based on description"}},
            "electrification_feasibility": {{"score": 2, "justification": "Brief reason based on description"}},
            "continuous_operations_intensity": {{"score": 6, "justification": "Brief reason based on description"}},
            "material_throughput_scale": {{"score": 3, "justification": "Brief reason based on description"}},
            "chemical_intensity": {{"score": 4, "justification": "Brief reason based on description"}},
            "capital_vs_labor_intensity": {{"score": 5, "justification": "Brief reason based on description"}}
        }}
    }}"""
        
        return prompt
    
    def _load_progress(self):
        """Load progress from file if exists"""
        if os.path.exists(self.progress_file):
            with open(self.progress_file, 'r') as f:
                return json.load(f)
        return {'processed': [], 'output_rows': []}
    
    def _save_progress(self, progress):
        """Save progress to file"""
        with open(self.progress_file, 'w') as f:
            json.dump(progress, f, indent=2)
    
    def _append_to_csv(self, row_data):
        """Append a single row to CSV file"""
        file_exists = os.path.exists(self.output_file)
        
        with open(self.output_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # Write header if file doesn't exist
            if not file_exists:
                header = ['industry_code', 'industry_name', 'timestamp'] + \
                        [f"{var}_score" for var in self.variables.keys()] + \
                        [f"{var}_justification" for var in self.variables.keys()]
                writer.writerow(header)
            
            # Prepare the row data
            row = [
                row_data['industry_code'],
                row_data['industry_name'],
                datetime.now().isoformat()
            ]
            
            # Add scores
            for var in self.variables.keys():
                row.append(row_data['ratings'].get(var, {}).get('score', ''))
            
            # Add justifications
            for var in self.variables.keys():
                row.append(row_data['ratings'].get(var, {}).get('justification', ''))
            
            writer.writerow(row)
    
    def extract_industry(self, industry_code, industry_name, industry_description, max_retries=3):
        """Extract ratings for a single industry with retries"""
        
        prompt = self._create_extraction_prompt(industry_code, industry_name, industry_description)
        
        for attempt in range(max_retries):
            try:
                payload = {
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "num_gpu": 1,
                        "temperature": 0.2,
                        "num_ctx": 4096,
                        "f16_kv": True,
                        "stop": ["\n\n\n"]
                    }
                }
                
                response = requests.post(self.url, json=payload, timeout=120)
                
                if response.status_code == 200:
                    result = response.json()
                    response_text = result['response'].strip()
                    
                    # Extract JSON from response
                    try:
                        json_start = response_text.find('{')
                        json_end = response_text.rfind('}') + 1
                        if json_start >= 0 and json_end > json_start:
                            json_str = response_text[json_start:json_end]
                            ratings_data = json.loads(json_str)
                            
                            # Validate structure
                            if 'ratings' in ratings_data:
                                return ratings_data
                    except:
                        pass
                
                time.sleep(2)  # Wait before retry
                
            except Exception as e:
                print(f"   Attempt {attempt + 1} failed: {e}")
                time.sleep(5)
        
        return None
    
    def process_all_industries(self, industries_file):
        """Process all industries from the Excel file"""
        
        print("🚀 Starting industry rating extraction")
        print("=" * 60)
        
        # Load industries
        industries_df = pd.read_excel(industries_file, sheet_name='Metadata')
        industries_df.columns = ['sector_code', 'sector_name', 'description']
        
        # Load progress
        progress = self._load_progress()
        processed = set(progress.get('processed', []))
        
        print(f"📊 Total industries: {len(industries_df)}")
        print(f"✅ Already processed: {len(processed)}")
        print(f"⏳ Remaining: {len(industries_df) - len(processed)}")
        print("=" * 60)
        
        # Process each industry
        for idx, row in industries_df.iterrows():
            code = str(row['sector_code']).strip()
            name = str(row['sector_name']).strip()
            description = str(row['description']).strip()

            
            if code in processed:
                print(f"⏩ Skipping {code} - already processed")
                continue
            
            print(f"\n📌 Processing [{idx+1}/{len(industries_df)}]: {code} - {name}")
            
            result = self.extract_industry(code, name, description)
            
            if result:
                # Append to CSV
                self._append_to_csv(result)
                
                # Update progress
                processed.add(code)
                progress['processed'] = list(processed)
                progress['output_rows'].append({
                    'industry_code': code,
                    'timestamp': datetime.now().isoformat()
                })
                self._save_progress(progress)
                
                print(f"✅ Successfully extracted ratings")
                
                # Show sample of results
                sample_var = list(self.variables.keys())[0]
                score = result['ratings'].get(sample_var, {}).get('score', 'N/A')
                print(f"   Sample: {sample_var} = {score}")
            else:
                print(f"❌ Failed to extract after retries")
            
            # Save after each industry
            print(f"💾 Progress saved. Total completed: {len(processed)}")
            
            # Small delay to prevent overwhelming the API
            time.sleep(1)
        
        print("\n" + "=" * 60)
        print("✅ EXTRACTION COMPLETE!")
        print(f"📁 Output saved to: {self.output_file}")
        print(f"📊 Total industries processed: {len(processed)}")

def main():
    # Initialize extractor
    extractor = IndustryRatingExtractor()
    
    # Process all industries
    extractor.process_all_industries("data/ceda_metadata.xlsx")
    
    # Show summary of results
    if os.path.exists(extractor.output_file):
        print("\n📊 First few rows of results:")
        df = pd.read_csv(extractor.output_file)
        print(df.head())
        print(f"\n📈 Total rows: {len(df)}")

if __name__ == "__main__":
    main()