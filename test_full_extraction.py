import pandas as pd
import anthropic
import json
import time
import os
from tqdm import tqdm
import os
from dotenv import load_dotenv

load_dotenv()

client = anthropic.Anthropic(
    api_key=os.getenv("ANTHROPIC_API_KEY")
)

DISTORTIONS = {
    "all_or_nothing": "All-or-Nothing Thinking",
    "overgeneralization": "Overgeneralization",
    "mental_filter": "Mental Filter",
    "mind_reading": "Mind Reading",
    "fortune_telling": "Fortune Telling",
    "catastrophizing": "Catastrophizing",
    "emotional_reasoning": "Emotional Reasoning",
    "should_statements": "Should Statements",
    "labeling": "Labeling",
    "personalization": "Personalization"
}

def extract_distortions(text: str, status: str):
    """Extract distortions from one statement"""
    
    prompt = f"""Analyze this statement for cognitive distortions.

Statement: "{text}"
Context: {status}

Return ONLY valid JSON:
{{
  "distortions": [
    {{
      "type": "distortion_key",
      "phrase": "exact quote",
      "confidence": 0.0-1.0,
      "explanation": "why"
    }}
  ],
  "overall_severity": 0.0-1.0,
  "primary_distortion": "key or null"
}}"""
    
    try:
        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1000,
            temperature=0.3,
            messages=[{"role": "user", "content": prompt}]
        )
        
        response_text = message.content[0].text.strip()
        
        if "```json" in response_text:
            json_str = response_text.split("```json")[1].split("```")[0].strip()
        else:
            json_str = response_text
        
        result = json.loads(json_str)
        return result
    
    except Exception as e:
        print(f"Error: {e}")
        return {"distortions": [], "overall_severity": 0.0, "primary_distortion": None}

def main():
    
    try:
        df = pd.read_csv('test_dataset.csv')
    except FileNotFoundError:
        exit(1)
    
    results = []

    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Extracting"):
        

        distortion_data = extract_distortions(row['statement'], row['status'])
        
        result = {
            'unique_id': row['unique_id'],
            'statement': row['statement'],
            'mental_health_status': row['status'],
            'distortions': distortion_data['distortions'],
            'distortion_count': len(distortion_data['distortions']),
            'overall_severity': distortion_data['overall_severity'],
            'primary_distortion': distortion_data['primary_distortion']
        }
        
        results.append(result)
        

        time.sleep(0.5)
    

    df_results = pd.DataFrame(results)

    df_results.to_json('test_results.json', orient='records', indent=2)
    
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    print(f"Total statements: {len(df_results)}")
    print(f"Statements with distortions: {(df_results['distortion_count'] > 0).sum()}")
    print(f"Average distortions per statement: {df_results['distortion_count'].mean():.2f}")
    
    print("\nDistortion frequency:")
    distortion_counts = {}
    for _, row in df_results.iterrows():
        for dist in row['distortions']:
            dist_type = dist['type']
            distortion_counts[dist_type] = distortion_counts.get(dist_type, 0) + 1
    
    for dist_type, count in sorted(distortion_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {DISTORTIONS.get(dist_type, dist_type)}: {count}")
    
    print("\n" + "="*60)
    print("DETAILED RESULTS")
    print("="*60)
    
    for idx, row in df_results.iterrows():
        print(f"\n{idx+1}. \"{row['statement'][:60]}...\"")
        print(f"   Status: {row['mental_health_status']}")
        print(f"   Distortions: {row['distortion_count']}")
        print(f"   Severity: {row['overall_severity']:.2f}")
        
        if row['distortions']:
            for dist in row['distortions']:
                print(f"     - {DISTORTIONS.get(dist['type'], dist['type'])}")
                print(f"       \"{dist['phrase']}\"")
    

if __name__ == "__main__":
    main()