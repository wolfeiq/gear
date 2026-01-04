import anthropic
import json
import random
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv

load_dotenv()

client = anthropic.Anthropic(
    api_key=os.getenv("ANTHROPIC_API_KEY")
)

TEST_JOURNEY = {
    "trajectory": "gradual_improvement",
    "duration_weeks": 4,
    "intervention": "CBT thought records",
    "distortion_change": -0.6
}

def generate_entry(week: int, initial_severity: float, journey: dict):
    """Generate one journal entry"""

    progress = week / journey['duration_weeks']
    severity_change = journey['distortion_change'] * progress
    current_severity = max(0.1, initial_severity + severity_change)
    
    prompt = f"""Generate a realistic mental health journal entry.

Week {week} of {journey['duration_weeks']}
Current severity: {current_severity:.2f} (0=good, 1=severe)
Trajectory: {journey['trajectory']}
Using: {journey['intervention']}

Write 3-4 sentences in first person about struggles with anxiety.
Include cognitive distortions appropriate to severity {current_severity:.2f}.
Show {"improvement" if journey['distortion_change'] < 0 else "decline"}.

Return ONLY the journal entry text."""

    try:
        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=300,
            temperature=0.8,
            messages=[{"role": "user", "content": prompt}]
        )
        
        return message.content[0].text.strip().strip('"')
    
    except Exception as e:
        print(f"Error: {e}")
        return None

def extract_distortions_quick(text: str):
    """Quick distortion extraction"""
    
    prompt = f"""Extract cognitive distortions from: "{text}"

Return JSON:
{{
  "distortions": ["type1", "type2"],
  "severity": 0.0-1.0
}}"""

    try:
        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=200,
            temperature=0.3,
            messages=[{"role": "user", "content": prompt}]
        )
        
        response = message.content[0].text.strip()
        
        if "```json" in response:
            json_str = response.split("```json")[1].split("```")[0].strip()
        else:
            json_str = response
        
        return json.loads(json_str)
    
    except:
        return {"distortions": [], "severity": 0.5}

def test_single_journey():

    
    initial_severity = 0.8
    entries = []
    
    for week in range(1, TEST_JOURNEY['duration_weeks'] + 1):

        entry_text = generate_entry(week, initial_severity, TEST_JOURNEY)
        
        if not entry_text:
            print("failed")
            continue
    

        distortions = extract_distortions_quick(entry_text)
        
        print(f"   Distortions: {len(distortions['distortions'])}")
        print(f"   Severity: {distortions['severity']:.2f}")

        entry = {
            'week': week,
            'text': entry_text,
            'distortions': distortions['distortions'],
            'severity': distortions['severity'],
            'timestamp': (datetime.now() - timedelta(weeks=TEST_JOURNEY['duration_weeks']-week)).isoformat()
        }
        
        entries.append(entry)
 
    
    if len(entries) >= 2:
        initial = entries[0]['severity']
        final = entries[-1]['severity']
        change = final - initial
        
        print(f"Initial severity: {initial:.2f}")
        print(f"Final severity: {final:.2f}")
        print(f"Change: {change:+.2f} ({abs(change/initial)*100:.1f}%)")
        
        if change < 0:
            print("✅ Improvement trajectory detected")
        else:
            print("⚠️  Worsening trajectory detected")
    

    for entry in entries:
        bar = "█" * int(entry['severity'] * 20)
        print(f"  Week {entry['week']}: {bar} {entry['severity']:.2f}")
    
    result = {
        'journey_type': TEST_JOURNEY['trajectory'],
        'entries': entries,
        'metadata': {
            'initial_severity': entries[0]['severity'] if entries else 0,
            'final_severity': entries[-1]['severity'] if entries else 0
        }
    }
    
    with open('test_journey.json', 'w') as f:
        json.dump(result, f, indent=2)

    
    return result

if __name__ == "__main__":
    
    test_single_journey()