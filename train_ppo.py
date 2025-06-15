# train_ppo.py

import pandas as pd
import os

from drl_env import DogeTradingEnv
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from sb3_contrib.ppo_recurrent import RecurrentPPO
from sb3_contrib.ppo_recurrent.policies import MlpLstmPolicy

# Load dataset
df = pd.read_csv("doge_1m_ohlcv.csv", parse_dates=["date"])
df = df.sort_values("date").reset_index(drop=True)

# Wrap env
def make_env():
    return DogeTradingEnv(df)

env = DummyVecEnv([make_env])

# Initialize model with LSTM-based policy
model = RecurrentPPO(
    policy=MlpLstmPolicy,
    env=env,
    verbose=1,
    n_steps=128,
    batch_size=64,
    learning_rate=2.5e-4,
    gamma=0.99,
    gae_lambda=0.95,
    ent_coef=0.01,
    clip_range=0.2,
    tensorboard_log="./tensorboard/"
)

# Train the model
model.learn(total_timesteps=200_000)

# Save the model
os.makedirs("models", exist_ok=True)
model.save("models/ppo_doge_recurrent")

print("✅ Training complete. Model saved to models/ppo_doge_recurrent")
