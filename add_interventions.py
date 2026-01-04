import json
import random

INTERVENTIONS = {
    "thought_records": {
        "name": "Thought Records",
        "targets": ["all_or_nothing_thinking", "catastrophizing", "overgeneralization"],
        "effectiveness": 0.75
    },
    "behavioral_experiments": {
        "name": "Behavioral Experiments",
        "targets": ["fortune_telling", "mind_reading", "catastrophizing"],
        "effectiveness": 0.82
    },
    "examine_evidence": {
        "name": "Examining Evidence",
        "targets": ["mind_reading", "overgeneralization", "jumping_to_conclusions"],
        "effectiveness": 0.78
    },
    "cognitive_restructuring": {
        "name": "Cognitive Restructuring",
        "targets": ["labeling", "should_statements", "emotional_reasoning"],
        "effectiveness": 0.71
    },
    "mindfulness": {
        "name": "Mindfulness Practice",
        "targets": ["rumination", "emotional_reasoning", "mental_filter"],
        "effectiveness": 0.68
    },
    "exposure_therapy": {
        "name": "Exposure Therapy",
        "targets": ["catastrophizing", "fortune_telling"],
        "effectiveness": 0.85
    }
}

JOURNEY_TO_INTERVENTIONS = {
    "improving": ["thought_records", "examine_evidence"],
    "plateauing": ["cognitive_restructuring", "mindfulness"],
    "worsening": [],
    "fluctuating": ["thought_records"],
    "rapid_improvement": ["behavioral_experiments", "exposure_therapy", "examine_evidence"]
}

def get_matching_interventions(distortions, available_interventions):
    """Find which interventions target these distortions"""
    matches = {}
    
    for intervention_key in available_interventions:
        intervention = INTERVENTIONS[intervention_key]
        
        # Count how many distortions this intervention targets
        overlap = len(set(distortions) & set(intervention['targets']))
        if overlap > 0:
            matches[intervention_key] = overlap
    
    return matches

def assign_interventions_to_entry(entry, journey_interventions):
    """Assign interventions to a single entry"""
    
    # Get distortions in this entry
    distortions = [d['type'] for d in entry['distortions']]
    
    # Find which assigned interventions match these distortions
    matching = get_matching_interventions(distortions, journey_interventions)
    
    # Randomly select 1-2 interventions that match
    if matching:
        interventions_used = random.sample(
            list(matching.keys()), 
            min(random.randint(1, 2), len(matching))
        )
    else:
        interventions_used = []
    
    return interventions_used

def add_interventions_to_data(input_file, output_file):
    """Add interventions to existing synthetic data"""
    
    print("📂 Loading existing data...")
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    print(f"Found {len(data)} user journeys\n")
    
    for journey in data:
        journey_type = journey['journey_type']
        assigned_interventions = JOURNEY_TO_INTERVENTIONS[journey_type]
        
        print(f"Processing {journey['user_profile']['user_id']} ({journey_type})")
        print(f"  Assigned interventions: {assigned_interventions}")
        
        # Add to user profile
        journey['user_profile']['assigned_interventions'] = assigned_interventions
        
        # Add to each entry
        for entry in journey['entries']:
            interventions_used = assign_interventions_to_entry(
                entry, 
                assigned_interventions
            )
            entry['interventions_used'] = interventions_used
            entry['intervention_active'] = len(interventions_used) > 0
        
        # Calculate intervention effectiveness
        entries_with_interventions = [
            e for e in journey['entries'] 
            if e['intervention_active']
        ]
        
        journey['intervention_stats'] = {
            'total_interventions_used': sum(
                len(e['interventions_used']) for e in journey['entries']
            ),
            'entries_with_interventions': len(entries_with_interventions),
            'percentage': len(entries_with_interventions) / len(journey['entries']) * 100
        }
    
    # Save enhanced data
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"\n✅ Saved enhanced data to {output_file}")
    
    # Statistics
    print("\n📊 INTERVENTION STATISTICS:")
    for journey in data:
        stats = journey['intervention_stats']
        print(f"\n{journey['user_profile']['user_id']}:")
        print(f"  Journey: {journey['journey_type']}")
        print(f"  Total intervention uses: {stats['total_interventions_used']}")
        print(f"  Entries with interventions: {stats['entries_with_interventions']}/{len(journey['entries'])}")
        print(f"  Improvement: {journey['improvement']:.3f}")

if __name__ == "__main__":
    add_interventions_to_data(
        'synthetic_longitudinal_data.json',
        'synthetic_longitudinal_with_interventions.json'
    )
    
    print("\n✅ Complete!")
    print("\nNext: Rebuild graphs with intervention-enhanced data")