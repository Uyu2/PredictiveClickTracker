import pandas as pd
import numpy as np
from faker import Faker
from datetime import datetime, timedelta

class DataGenerator:
    def __init__(self, n_samples=1000):
        self.fake = Faker()
        self.n_samples = n_samples
        
    def generate_data(self):
        # Set random seed for reproducibility
        np.random.seed(42)
        
        # Generate timestamps over last 30 days
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        timestamps = [self.fake.date_time_between(start_date=start_date, end_date=end_date) 
                     for _ in range(self.n_samples)]
        
        # Common search terms
        search_terms = ['product', 'pricing', 'support', 'features', 'contact', 
                       'demo', 'login', 'signup', 'help', 'documentation']
        
        data = {
            'timestamp': timestamps,
            'time_on_screen': np.random.exponential(scale=300, size=self.n_samples),  # seconds
            'exited_screen': np.random.choice([0, 1], size=self.n_samples, p=[0.7, 0.3]),
            'search_count': np.random.poisson(lam=2, size=self.n_samples),
            'search_term': [np.random.choice(search_terms) for _ in range(self.n_samples)],
            'device_type': np.random.choice(['desktop', 'mobile', 'tablet'], 
                                          size=self.n_samples, p=[0.6, 0.3, 0.1]),
            'browser': np.random.choice(['chrome', 'firefox', 'safari', 'edge'], 
                                      size=self.n_samples, p=[0.5, 0.2, 0.2, 0.1]),
            'referrer': np.random.choice(['direct', 'search', 'social', 'email'], 
                                       size=self.n_samples, p=[0.4, 0.3, 0.2, 0.1])
        }
        
        # Generate click-through rate based on features
        base_prob = 0.3
        device_effect = {'desktop': 0.1, 'mobile': -0.05, 'tablet': 0}
        browser_effect = {'chrome': 0.05, 'firefox': 0, 'safari': 0, 'edge': -0.05}
        
        click_probs = np.array([
            base_prob +
            device_effect[dev] +
            browser_effect[br] +
            (-0.1 if ex else 0.1) +  # Exit effect
            min(0.2, ts/1000)  # Time on screen effect
            for dev, br, ex, ts in zip(
                data['device_type'],
                data['browser'],
                data['exited_screen'],
                data['time_on_screen']
            )
        ])
        
        data['clicked'] = np.random.binomial(n=1, p=np.clip(click_probs, 0, 1))
        
        return pd.DataFrame(data)
