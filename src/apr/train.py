import argparse
from pathlib import Path
import yaml

from apr.env import WarehouseEnv
from apr.logger import RunLogger
from apr.agents import create_agent, list_available_agents

# ------------------------------------------------------------------ #

def run_episode(env, agent, training=True):
    state = env.reset()
    done = False
    total = 0
    while not done:
        action = agent.act(state, training=training)
        next_state, reward, done, _ = env.step(action)
        if training:
            agent.learn(state, action, reward, next_state, done)
        state = next_state
        total += reward
    return total

# ------------------------------------------------------------------ #

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, help="YAML with env/agent/train")
    args = p.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())

    # 1) Environment
    env = WarehouseEnv(**cfg["env"])

    # 2) Agent (using registry for clean algorithm lookup)
    algo = cfg["agent"]["algo"]
    try:
        agent = create_agent(
            algo,
            env.observation_space,
            env.action_space,
            alpha=cfg["agent"]["alpha"],
            gamma=cfg["agent"]["gamma"],
            epsilon=cfg["agent"]["epsilon"],
        )
    except ValueError as e:
        available = list_available_agents()
        print(f"Error: {e}")
        print(f"Available algorithms: {', '.join(available)}")
        return

    # 3) Logger
    logger = RunLogger(cfg, tag=algo)

    # 4) Training loop
    for ep in range(1, cfg["train"]["episodes"] + 1):
        reward = run_episode(env, agent, training=True)
        logger.log(episode=ep, reward=reward, epsilon=agent.epsilon)

        if ep % cfg["train"]["log_every"] == 0:
            ckpt = logger.dir / "checkpoints" / f"ckpt_ep{ep:05d}.pt"
            agent.save(ckpt)
            print(f"[{ep:>5}] R={reward:>6.1f}  ε={agent.epsilon:.3f}  → saved {ckpt.name}")

if __name__ == "__main__":
    main()