from gym.envs.registration import register

register(
    id='Dribble-v0',
    entry_point='gym_dribble.gym_dribble:DribbleEnv',
    max_episode_steps=1000,
    reward_threshold=99999,
)