import pandas as pd

test_data = [
    {
        'unique_id': 1,
        'statement': "I always mess everything up. I'm such a failure. Everyone probably thinks I'm incompetent.",
        'status': 'Depression'
    },
    {
        'unique_id': 2,
        'statement': "I feel anxious about the presentation tomorrow. What if I forget everything?",
        'status': 'Anxiety'
    },
    {
        'unique_id': 3,
        'statement': "Had a good day at work. Finished the project on time and got positive feedback.",
        'status': 'Normal'
    },
    {
        'unique_id': 4,
        'statement': "I can't believe I made that mistake. Now everything is ruined. My career is over.",
        'status': 'Stress'
    },
    {
        'unique_id': 5,
        'statement': "I should be better at this by now. I must work harder or I'll never succeed.",
        'status': 'Anxiety'
    }
]

df = pd.DataFrame(test_data)
df.to_csv('test_dataset.csv', index=False)