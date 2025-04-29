import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Generate synthetic dataset
def generate_livestock_data(num_samples=10000):
    np.random.seed(42)

    data = {
        'body_temperature': np.random.normal(38.5, 0.5, num_samples),
        'heart_rate': np.random.randint(40, 120, num_samples),
        'respiration_rate': np.random.randint(12, 36, num_samples),
        'activity_level': np.random.uniform(0, 1, num_samples),
        'feeding_frequency': np.random.randint(1, 5, num_samples),
        'environment_temp': np.random.uniform(15, 40, num_samples),
        'vocalization_freq': np.random.poisson(3, num_samples)
    }

    df = pd.DataFrame(data)

    # Create stress level based on features
    df['stress_level'] = np.where(
        (df['body_temperature'] > 39.5) |
        (df['heart_rate'] > 100) |
        (df['respiration_rate'] > 30) |
        (df['environment_temp'] > 35) |
        (df['vocalization_freq'] > 5),
        1, 0  # 1 = Stressed, 0 = Normal
    )

    # Add noise
    df['stress_level'] = df['stress_level'].apply(lambda x: x if np.random.random() > 0.1 else 1-x)

    return df

# Generate and save dataset
dataset = generate_livestock_data(10000)
dataset.to_csv('livestock_stress_data.csv', index=False)
print("Dataset generated successfully!")