import pandas as pd
import anthropic
import json
from tqdm import tqdm
import time
from typing import List, Dict
import os
from dotenv import load_dotenv

load_dotenv()

client = anthropic.Anthropic(
    api_key=os.getenv("ANTHROPIC_API_KEY")
)


DISTORTIONS = {
    "all_or_nothing": {
        "name": "All-or-Nothing Thinking",
        "description": "Seeing things in black-and-white categories",
        "examples": ["always", "never", "completely", "totally"]
    },
    "overgeneralization": {
        "name": "Overgeneralization",
        "description": "Seeing a single negative event as a never-ending pattern",
        "examples": ["always happens", "never works", "every time"]
    },
    "mental_filter": {
        "name": "Mental Filter",
        "description": "Dwelling on negatives and ignoring positives",
        "examples": ["only the bad", "nothing good"]
    },
    "mind_reading": {
        "name": "Mind Reading",
        "description": "Assuming you know what others think without evidence",
        "examples": ["they think I'm", "everyone must think", "they probably hate"]
    },
    "fortune_telling": {
        "name": "Fortune Telling",
        "description": "Predicting negative outcomes without evidence",
        "examples": ["I know it will", "it's going to", "I'll definitely"]
    },
    "catastrophizing": {
        "name": "Catastrophizing",
        "description": "Expecting disaster or blowing things out of proportion",
        "examples": ["it will be terrible", "I'll never recover", "everything is ruined"]
    },
    "emotional_reasoning": {
        "name": "Emotional Reasoning",
        "description": "Assuming feelings reflect reality",
        "examples": ["I feel like", "I feel therefore I am"]
    },
    "should_statements": {
        "name": "Should Statements",
        "description": "Rigid rules about how you or others should behave",
        "examples": ["I should", "I must", "I have to"]
    },
    "labeling": {
        "name": "Labeling",
        "description": "Assigning global negative labels to yourself or others",
        "examples": ["I'm a failure", "I'm worthless", "I'm stupid"]
    },
    "personalization": {
        "name": "Personalization",
        "description": "Blaming yourself for things outside your control",
        "examples": ["it's my fault", "I caused this", "if only I had"]
    }
}

def extract_distortions(text: str, mental_health_status: str) -> Dict:

    prompt = f"""You are a cognitive behavioral therapy expert. Analyze this statement for cognitive distortions.

Statement: "{text}"
Context: This person is experiencing {mental_health_status}

For EACH distortion present:
1. Extract the EXACT phrase from the statement
2. Identify the distortion type
3. Rate confidence (0.0-1.0)
4. Explain WHY it's that distortion

Available distortion types:
{json.dumps({k: v["name"] for k, v in DISTORTIONS.items()}, indent=2)}

Respond with ONLY valid JSON in this exact format:
{{
  "distortions": [
    {{
      "type": "distortion_key",
      "phrase": "exact quote from statement",
      "confidence": 0.95,
      "explanation": "brief explanation"
    }}
  ],
  "overall_severity": 0.0-1.0,
  "primary_distortion": "most prominent distortion type or null"
}}

If NO distortions found, return: {{"distortions": [], "overall_severity": 0.0, "primary_distortion": null}}
"""
    
    try:
        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1500,
            temperature=0.3, 
            messages=[{"role": "user", "content": prompt}]
        )
        
        response_text = message.content[0].text.strip()
        
        if "```json" in response_text:
            json_str = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            json_str = response_text.split("```")[1].split("```")[0].strip()
        else:
            json_str = response_text
        
        result = json.loads(json_str)
        
        if "distortions" not in result:
            result["distortions"] = []
        if "overall_severity" not in result:
            result["overall_severity"] = 0.0
        if "primary_distortion" not in result:
            result["primary_distortion"] = None
            
        return result
    
    except json.JSONDecodeError as e:
        print(f"JSON parse error: {e}")
        print(f"Raw response: {response_text[:200]}")
        return {"distortions": [], "overall_severity": 0.0, "primary_distortion": None}
    
    except Exception as e:
        print(f"Error: {e}")
        return {"distortions": [], "overall_severity": 0.0, "primary_distortion": None}

def process_dataset(input_csv: str, output_json: str, sample_size: int = None):
   
    df = pd.read_csv(input_csv)
    
    df = df[df['status'].isin(['Depression', 'Anxiety', 'Stress'])]
    df = df[df['statement'].str.len() > 100]
    df = df.sample(n=500, random_state=42)
    
    results = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Extracting distortions"):
        statement = row['statement']
        status = row['status']
        
        distortion_data = extract_distortions(statement, status)
        
        result = {
            'unique_id': row.get('unique_id', idx),
            'statement': statement,
            'mental_health_status': status,
            'distortions': distortion_data['distortions'],
            'distortion_count': len(distortion_data['distortions']),
            'overall_severity': distortion_data['overall_severity'],
            'primary_distortion': distortion_data['primary_distortion']
        }
        
        results.append(result)
        

        if (idx + 1) % 50 == 0:
            temp_df = pd.DataFrame(results)
            temp_df.to_json(f'{output_json}.progress', orient='records', indent=2)
        
        time.sleep(0.5)
    
    df_labeled = pd.DataFrame(results)
    df_labeled.to_json(output_json, orient='records', indent=2)
    
    df_csv = df_labeled.copy()
    df_csv['distortions'] = df_csv['distortions'].apply(json.dumps)
    df_csv.to_csv(output_json.replace('.json', '.csv'), index=False)

    
    print("\nEXTRACTION STATISTICS:")
    print(f"Total statements: {len(df_labeled)}")
    print(f"Statements with distortions: {(df_labeled['distortion_count'] > 0).sum()}")
    print(f"Average distortions per statement: {df_labeled['distortion_count'].mean():.2f}")
    print(f"\nDistortion frequency:")
    
    distortion_counts = {}
    for _, row in df_labeled.iterrows():
        for dist in row['distortions']:
            dist_type = dist['type']
            distortion_counts[dist_type] = distortion_counts.get(dist_type, 0) + 1
    
    for dist_type, count in sorted(distortion_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {DISTORTIONS[dist_type]['name']}: {count}")
    
    return df_labeled

def quick_analysis(df: pd.DataFrame):
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    plt.figure(figsize=(10, 6))
    df['distortion_count'].hist(bins=20)
    plt.xlabel('Number of Distortions')
    plt.ylabel('Frequency')
    plt.title('Distribution of Distortions per Statement')
    plt.savefig('distortion_distribution.png')
    
    plt.figure(figsize=(12, 6))
    status_dist = df.groupby('mental_health_status')['distortion_count'].mean().sort_values()
    status_dist.plot(kind='barh')
    plt.xlabel('Average Distortion Count')
    plt.title('Average Distortions by Mental Health Status')
    plt.tight_layout()
    plt.savefig('distortions_by_status.png')
    

if __name__ == "__main__":
    INPUT_FILE = "kaggle_mental_health.csv"
    OUTPUT_FILE = "dataset_with_distortions.json"

    
    df_labeled = process_dataset(
        input_csv=INPUT_FILE,
        output_json=OUTPUT_FILE,
        sample_size=100
    )

    quick_analysis(df_labeled)