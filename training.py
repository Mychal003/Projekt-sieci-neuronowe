import os
import json
import gymnasium as gym
from collections import deque
import sys
import numpy as np
import random
import torch
import time
import wandb
from dqnagent import DQNAgent

MAX_EPISODES = 500
MAX_STEPS = 1000
RECORD_FREQ = max(1, MAX_EPISODES // 5)


def train(custom_config=None, run_name=None, tags=None):
    if custom_config is not None:
        wandb.init(
            project="dqn-lunar-lander", config=custom_config, name=run_name, tags=tags
        )
    else:
        wandb.init()

    config = wandb.config
    experiment_name = wandb.run.name

    seed = config.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    hyperparams = {
        "lr": config.lr,
        "batch_size": config.batch_size,
        "epsilon_decay": config.epsilon_decay,
        "target_update_steps": config.target_update_steps,
        "hidden_sizes": tuple(config.hidden_sizes),
        "gamma": config.gamma,
    }

    env = gym.make("LunarLander-v3", render_mode="rgb_array")
    env = gym.wrappers.RecordVideo(
        env,
        video_folder=f"records/{experiment_name}",
        episode_trigger=lambda episode: episode % RECORD_FREQ == 0,
        disable_logger=True,
    )
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
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

    state, _ = env.reset(seed=seed)

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

        wandb.log(
            {
                "episode": episode,
                "total_reward": total_reward,
                "moving_avg_reward": current_avg,
                "loss": avg_loss,
                "epsilon": agent.epsilon,
                "steps": step,
            }
        )

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
    wandb.finish()
    return rewards_history


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--agent":
        sweep_id = sys.argv[2]
        wandb.agent(sweep_id, function=train, project="dqn-lunar-lander")
        sys.exit(0)

    baseline_config = {
        "lr": 1e-4,
        "batch_size": 64,
        "epsilon_decay": 0.995,
        "target_update_steps": 1000,
        "hidden_sizes": [128, 128],
        "gamma": 0.99,
        "seed": 42,
    }

    print("--- Uruchamiam konfigurację bazową (Baseline) ---")
    train(custom_config=baseline_config, run_name="baseline_run", tags=["baseline"])

    sweep_config = {
        "method": "random",
        "metric": {"name": "moving_avg_reward", "goal": "maximize"},
        "parameters": {
            "lr": {"values": [1e-4, 5e-4, 1e-3]},
            "batch_size": {"values": [32, 64, 128]},
            "epsilon_decay": {"values": [0.99, 0.995, 0.999]},
            "target_update_steps": {"values": [500, 1000, 2000]},
            "hidden_sizes": {"values": [[64, 64], [128, 128], [256, 256]]},
            "gamma": {"values": [0.95, 0.99, 0.999]},
            "seed": {"value": 42},
        },
    }

    # Initiate the sweep on the W&B server
    sweep_id = wandb.sweep(sweep_config, project="dqn-lunar-lander")
    print(f"Sweep initiated! ID: {sweep_id}")
    print(
        "\nTo run agents in parallel, execute the following command multiple times in your terminal:"
    )
    print(f"uv run python training.py --agent {sweep_id} &")
