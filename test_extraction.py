import anthropic
import json
import os
from dotenv import load_dotenv

load_dotenv()

client = anthropic.Anthropic(
    api_key=os.getenv("ANTHROPIC_API_KEY")
)


TEST_STATEMENT = "I always mess everything up. I'm such a failure. Everyone probably thinks I'm incompetent."
TEST_STATUS = "Depression"

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

def test_extraction(text: str, status: str):
    print(f"\n{'='*60}")
    print("TEST: Distortion Extraction")
    print(f"{'='*60}")
    print(f"\nStatement: \"{text}\"")
    print(f"Status: {status}")
    
    prompt = f"""You are a cognitive behavioral therapy expert. Analyze this statement for cognitive distortions.

Statement: "{text}"
Context: This person is experiencing {status}

For EACH distortion present:
1. Extract the EXACT phrase from the statement
2. Identify the distortion type
3. Rate confidence (0.0-1.0)
4. Explain WHY it's that distortion

Available distortion types:
{json.dumps(DISTORTIONS, indent=2)}

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
        
        print("\n" + "="*60)
        print("PARSED RESULTS:")
        print("="*60)
        print(f"Distortions found: {len(result['distortions'])}")
        print(f"Overall severity: {result['overall_severity']}")
        print(f"Primary distortion: {result['primary_distortion']}")
        
        print("\nDetailed breakdown:")
        for i, dist in enumerate(result['distortions'], 1):
            print(f"\n{i}. {DISTORTIONS.get(dist['type'], dist['type'])}")
            print(f"   Phrase: \"{dist['phrase']}\"")
            print(f"   Confidence: {dist['confidence']:.2f}")
            print(f"   Explanation: {dist['explanation']}")
        
        return result
        
    except json.JSONDecodeError as e:
        return None
    
    except Exception as e:
        return None

if __name__ == "__main__":
    result = test_extraction(TEST_STATEMENT, TEST_STATUS)
    
    if result:
        print("\ntested")

    else:
        print("\nFailed :(")