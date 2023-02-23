from gym.envs.registration import register

# Register our environment
register(
    id="cuas-v0",
    entry_point="cuas.envs:CuasEnv",
)

register(
    id="cuas_single_agent-v0",
    entry_point="cuas.envs:CuasEnvSingleAgent",
)

register(
    id="cuas_multi_agent-v0",
    entry_point="cuas.envs:CuasEnvMultiAgent",
)

register(
    id="cuas_multi_agent-v1",
    entry_point="cuas.envs:CuasEnvMultiAgentV1",
)
