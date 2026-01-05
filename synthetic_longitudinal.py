import pandas as pd
import anthropic
import json
from datetime import datetime, timedelta
import random
from tqdm import tqdm
import time
import os
from dotenv import load_dotenv

load_dotenv()

client = anthropic.Anthropic(
    api_key=os.getenv("ANTHROPIC_API_KEY")
)


JOURNEY_TEMPLATES = {
    "improving": {
        "trajectory": "gradual_improvement",
        "duration_weeks": 12,
        "intervention": "CBT thought records",
        "distortion_change": -0.6
    },
    "plateauing": {
        "trajectory": "initial_improvement_then_stable",
        "duration_weeks": 16,
        "intervention": "therapy + medication",
        "distortion_change": -0.4
    },
    "worsening": {
        "trajectory": "gradual_decline",
        "duration_weeks": 8,
        "intervention": None,
        "distortion_change": 0.3  
    },
    "fluctuating": {
        "trajectory": "up_and_down",
        "duration_weeks": 20,
        "intervention": "inconsistent therapy",
        "distortion_change": -0.2
    },
    "rapid_improvement": {
        "trajectory": "breakthrough",
        "duration_weeks": 6,
        "intervention": "intensive therapy",
        "distortion_change": -0.7
    }
}

def generate_user_profile(base_statement: dict) -> dict:

    age = random.randint(18, 65)
    
    profile = {
        "user_id": f"user_{random.randint(10000, 99999)}",
        "age": age,
        "initial_status": base_statement['mental_health_status'],
        "initial_severity": base_statement['overall_severity'],
        "primary_distortion": base_statement['primary_distortion'],
        "journey_type": random.choice(list(JOURNEY_TEMPLATES.keys())),
        "base_statement": base_statement['statement'],
        "base_distortions": base_statement['distortions']
    }
    
    return profile

def generate_journal_entry(
    user_profile: dict,
    week: int,
    previous_entries: list
) -> dict:

    journey = JOURNEY_TEMPLATES[user_profile['journey_type']]
    progress_pct = week / journey['duration_weeks']

    initial_severity = user_profile['initial_severity']
    severity_change = journey['distortion_change'] * progress_pct
    current_severity = max(0.1, min(1.0, initial_severity + severity_change))
    
    current_severity += random.gauss(0, 0.1)
    current_severity = max(0.1, min(1.0, current_severity))
    
    context = ""
    if previous_entries:
        recent = previous_entries[-2:] if len(previous_entries) >= 2 else previous_entries
        context = f"Previous entries: {[e['entry_text'][:100] for e in recent]}"
    
    prompt = f"""Generate a realistic mental health journal entry for this person:

Profile:
- Mental health status: {user_profile['initial_status']}
- Primary distortion pattern: {user_profile['primary_distortion']}
- Journey type: {user_profile['journey_type']}
- Week {week} of {journey['duration_weeks']} (progress: {progress_pct*100:.0f}%)
- Current severity: {current_severity:.2f} (0=none, 1=severe)
- Intervention: {journey['intervention']}

{context}

Base their struggles on: "{user_profile['base_statement']}"

Requirements:
1. Write 3-5 sentences in first person
2. Reflect current severity level and trajectory
3. Show realistic progression (not linear - some setbacks)
4. Include specific situations/triggers
5. If intervention exists, mention it naturally
6. Match the tone/style of someone with {user_profile['initial_status']}

IMPORTANT: Include cognitive distortions appropriate to severity level.
At severity {current_severity:.2f}, expect {"many" if current_severity > 0.7 else "moderate" if current_severity > 0.4 else "few"} distortions.

Return ONLY the journal entry text, no preamble."""

    try:
        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=500,
            temperature=0.8,  # Higher for variety
            messages=[{"role": "user", "content": prompt}]
        )
        
        entry_text = message.content[0].text.strip()
        
        # Remove any quotes or markdown
        entry_text = entry_text.strip('"').strip('`').strip()
        
        return {
            "week": week,
            "entry_text": entry_text,
            "expected_severity": current_severity,
            "intervention_active": journey['intervention'] is not None
        }
        
    except Exception as e:
        print(f"Error generating entry: {e}")
        return None

def extract_distortions_from_entry(entry_text: str, expected_severity: float) -> dict:
    prompt = f"""Analyze this journal entry for cognitive distortions.

Entry: "{entry_text}"
Expected severity: {expected_severity:.2f}

Extract all distortions present with their exact phrases.

Return JSON:
{{
  "distortions": [
    {{
      "type": "distortion_key",
      "phrase": "exact quote",
      "confidence": 0.0-1.0
    }}
  ],
  "measured_severity": 0.0-1.0
}}"""

    try:
        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=800,
            temperature=0.3,
            messages=[{"role": "user", "content": prompt}]
        )
        
        response_text = message.content[0].text.strip()
        
        if "```json" in response_text:
            json_str = response_text.split("```json")[1].split("```")[0].strip()
        else:
            json_str = response_text
        
        return json.loads(json_str)
    
    except Exception as e:
        print(f"Error extracting distortions: {e}")
        return {"distortions": [], "measured_severity": expected_severity}

def generate_user_journey(base_statement: dict, entries_per_week: int = 2) -> dict:
    profile = generate_user_profile(base_statement)
    journey = JOURNEY_TEMPLATES[profile['journey_type']]
    
    entries = []
    
    for week in range(1, journey['duration_weeks'] + 1):

        for entry_num in range(entries_per_week):
            
            entry = generate_journal_entry(profile, week, entries)
            
            if entry:
                distortion_data = extract_distortions_from_entry(
                    entry['entry_text'],
                    entry['expected_severity']
                )
                
                days_offset = (week - 1) * 7 + (entry_num * 3) 
                timestamp = datetime.now() - timedelta(days=days_offset)

                complete_entry = {
                    **entry,
                    "timestamp": timestamp.isoformat(),
                    "distortions": distortion_data['distortions'],
                    "measured_severity": distortion_data['measured_severity'],
                    "distortion_count": len(distortion_data['distortions'])
                }
                
                entries.append(complete_entry)
                
                time.sleep(0.3) 
    
    return {
        "user_profile": profile,
        "journey_type": profile['journey_type'],
        "entries": entries,
        "total_entries": len(entries),
        "duration_weeks": journey['duration_weeks'],
        "initial_severity": profile['initial_severity'],
        "final_severity": entries[-1]['measured_severity'] if entries else 0,
        "improvement": profile['initial_severity'] - (entries[-1]['measured_severity'] if entries else 0)
    }

def generate_synthetic_cohort(
    base_statements_df: pd.DataFrame,
    num_users: int = 10,
    entries_per_week: int = 2
) -> list:

    sampled = base_statements_df.sample(n=num_users, random_state=42)
    
    all_journeys = []
    
    for idx, row in tqdm(sampled.iterrows(), total=num_users, desc="Generating journeys"):
    
        base_statement = row.to_dict()
    
        journey = generate_user_journey(base_statement, entries_per_week)
        all_journeys.append(journey)
        if (idx + 1) % 10 == 0:
            with open('synthetic_journeys_progress.json', 'w') as f:
                json.dump(all_journeys, f, indent=2)
    
    return all_journeys

def analyze_synthetic_cohort(journeys: list):
    
    journey_types = {}
    for j in journeys:
        jtype = j['journey_type']
        journey_types[jtype] = journey_types.get(jtype, 0) + 1
    
    print("\nJourney types:")
    for jtype, count in journey_types.items():
        print(f"  {jtype}: {count}")
    
 
    improvements = [j['improvement'] for j in journeys]
    print(f"\nImprovement statistics:")
    print(f"  Mean: {sum(improvements)/len(improvements):.2f}")
    print(f"  Users improved: {sum(1 for i in improvements if i > 0)}")
    print(f"  Users worsened: {sum(1 for i in improvements if i < 0)}")
    
    total_entries = sum(j['total_entries'] for j in journeys)
    print(f"\nTotal synthetic entries: {total_entries}")
    print(f"Average per user: {total_entries/len(journeys):.1f}")

if __name__ == "__main__":
    df_labeled = pd.read_json('dataset_with_distortions.json')
    
    df_with_distortions = df_labeled[df_labeled['distortion_count'] > 0]
    print(f"Found {len(df_with_distortions)} statements with distortions")
    
    journeys = generate_synthetic_cohort(
        base_statements_df=df_with_distortions,
        num_users=10, 
        entries_per_week=2
    )

    output_file = 'synthetic_longitudinal_data.json'
    with open(output_file, 'w') as f:
        json.dump(journeys, f, indent=2)

    analyze_synthetic_cohort(journeys)