import os
import gymnasium as gym
import torch
from dqnagent import DQNAgent
from training import base_params, experiments_config

MAX_STEPS = 1000


def record_final_model(hyperparams, experiment_name, best_model_state: bool):
    model_path = (
        f"models/{experiment_name}_{'best'if best_model_state else 'final'}.pth"
    )
    if not os.path.exists(model_path):
        print(f"Nie znaleziono pliku modelu {model_path}. Pomijam.")
        return

    env = gym.make("LunarLander-v3", render_mode="rgb_array")
    env = gym.wrappers.RecordVideo(
        env,
        video_folder="records/final_tests",
        episode_trigger=lambda episode: True,  # Record every episode (we only run 1)
        disable_logger=True,
        name_prefix=f"{'best'if best_model_state else 'final'}_{experiment_name}",
    )

    input_size = env.observation_space.shape[0]
    output_size = env.action_space.n

    agent = DQNAgent(input_size, output_size, **hyperparams)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent.model.load_state_dict(torch.load(model_path, map_location=device))
    agent.model.eval()  # Set to evaluation mode
    agent.epsilon = -0.01  # Force completely greedy policy (no exploration)

    state, _ = env.reset(seed=42)
    done = False
    step = 0
    total_reward = 0.0

    while not done and step < MAX_STEPS:
        action = agent.act(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        total_reward += float(reward)
        done = terminated or truncated
        state = next_state
        step += 1

    env.close()
    print(
        f"Zakończono nagrywanie {experiment_name}. Całkowita nagroda: {total_reward:.2f}"
    )


if __name__ == "__main__":
    record_final_model(base_params, "baseline", False)
    record_final_model(base_params, "baseline", True)

    for param_name, param_values in experiments_config.items():
        for val in param_values:
            current_params = base_params.copy()
            current_params[param_name] = val
            val_str = "_".join(map(str, val)) if isinstance(val, tuple) else str(val)
            experiment_name = f"{param_name}_{val_str}"
            record_final_model(current_params, experiment_name, False)
            record_final_model(current_params, experiment_name, True)
