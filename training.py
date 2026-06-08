import os
import json
import gymnasium as gym
from collections import deque
import numpy as np
import random
import torch
import time
from dqnagent import DQNAgent

MAX_EPISODES = 500
MAX_STEPS = 1000
RECORD_FREQ = max(1, MAX_EPISODES // 5)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run_experiment(hyperparams, experiment_name):
    env = gym.make("LunarLander-v3", render_mode="rgb_array")
    env = gym.wrappers.RecordVideo(
        env,
        video_folder=f"records/{experiment_name}",
        episode_trigger=lambda episode: episode % RECORD_FREQ == 0,
        disable_logger=True,
    )
    env.action_space.seed(SEED)
    env.observation_space.seed(SEED)
    input_size = env.observation_space.shape[0]
    output_size = env.action_space.n

    agent = DQNAgent(input_size, output_size, **hyperparams)

    rewards_history = []
    moving_avg_history = []
    loss_history = []
    length_history = []
    epsilon_history = []

    recent_rewards = deque(maxlen=50)
    best_avg_reward = -float("inf")

    os.makedirs("models", exist_ok=True)
    os.makedirs("metrics", exist_ok=True)

    state, _ = env.reset(seed=SEED)

    start_time = time.perf_counter()
    for episode in range(MAX_EPISODES):
        total_reward = 0.0
        done = False
        step = 0
        episode_losses = []

        while not done and step < MAX_STEPS:
            action = agent.act(state)
            result = env.step(action)

            next_state, reward, terminated, truncated, _ = result
            done = terminated or truncated

            agent.memorize(state, action, reward, next_state, done)
            total_reward += float(reward)
            state = next_state
            loss = agent.replay()
            if loss is not None:
                episode_losses.append(loss)
            step += 1

        agent.decay_epsilon()
        rewards_history.append(total_reward)
        recent_rewards.append(total_reward)
        length_history.append(step)
        epsilon_history.append(agent.epsilon)
        avg_loss = float(np.mean(episode_losses)) if len(episode_losses) > 0 else 0.0
        loss_history.append(avg_loss)

        current_avg = float(np.mean(recent_rewards))
        moving_avg_history.append(current_avg)

        if len(recent_rewards) >= 50 and current_avg > best_avg_reward:
            best_avg_reward = current_avg
            torch.save(agent.model.state_dict(), f"models/{experiment_name}_best.pth")

        if episode % 10 == 0:
            print(
                f"Epizod {episode}/{MAX_EPISODES}, Nagroda: {total_reward:.2f}, Średnia (50): {current_avg:.2f}, Epsilon: {agent.epsilon:.3f}"
            )
        state, _ = env.reset()

    end_time = time.perf_counter()
    print(f"Całkowity czas treningu: {end_time - start_time:.2f} sekund")

    torch.save(agent.model.state_dict(), f"models/{experiment_name}_final.pth")

    metrics = {
        "rewards_history": rewards_history,
        "moving_avg_history": moving_avg_history,
        "loss_history": loss_history,
        "length_history": length_history,
        "epsilon_history": epsilon_history,
        "hyperparameters": hyperparams,
        "training_time": end_time - start_time,
    }
    with open(f"metrics/{experiment_name}.json", "w") as f:
        json.dump(metrics, f)

    env.close()
    return rewards_history


if __name__ == "__main__":
    base_params = {
        "lr": 1e-4,
        "batch_size": 64,
        "epsilon_decay": 0.995,
        "target_update_steps": 1000,
        "hidden_sizes": (128, 128),
        "gamma": 0.99,
    }

    experiments = {
        "lr": [5e-4, 1e-3],
        "epsilon_decay": [0.99, 0.985],
        "target_update_steps": [500, 2000],
        "hidden_sizes": [(64, 64)],
        "batch_size": [32, 128],
        "gamma": [0.95, 0.999],
    }

    print("--- Running Baseline ---")
    run_experiment(base_params, "baseline")

    for param_name, param_values in experiments.items():
        for val in param_values:
            current_params = base_params.copy()
            current_params[param_name] = val

            val_str = "_".join(map(str, val)) if isinstance(val, tuple) else str(val)
            experiment_name = f"{param_name}_{val_str}"

            print(f"\n--- Running Experiment: {experiment_name} ---")
            run_experiment(current_params, experiment_name)
