# Multi-Agent Counter UAS (CUAS) Using Reinforcement Learning

## Single Agent Pursuer Evader
![Single Agent Pursuer Evader](images/1v1_multiagent_ppo.gif)

## Multi Agent Pursuer Evader
![Multi Agent Purser Evader](images/4v1_multiagent_ppo.gif)

A Gym Environment for Counter UAS

conda install tensorflow-gpu


to train for 5 hours:
python train_test_cuas_multi_agent.py train --duration $(( 5*60*60 ))

to test:
```
python train_test_cuas_multi_agentpy test --checkpoint <checkpoint>
```

https://github.com/sisl/MADRL
https://github.com/ChanganVR/CADRL/blob/master/train.py


pip install -U "ray[tune,rllib,serve]"
pip install -U "ray[rllib]"
conda install -c conda-forge opencv


Issue with ray 1.13 and PyTorch 1.13: 
https://github.com/ray-project/ray/issues/26557

change line 760 in torch_policy.py

```python 
    def set_state(self, state: dict) -> None:
        # Set optimizer vars first.
        optimizer_vars = state.get("_optimizer_variables", None)
        if optimizer_vars:
            assert len(optimizer_vars) == len(self._optimizers)
            for o, s in zip(self._optimizers, optimizer_vars):
#fix
                for v in s["param_groups"]:
                    if "foreach" in v.keys():
                        v["foreach"] = False if v["foreach"] is None else v["foreach"]
                for v in s["state"].values():
                    if "momentum_buffer" in v.keys():
                        v["momentum_buffer"] = False if v["momentum_buffer"] is None else v["momentum_buffer"]
                optim_state_dict = convert_to_torch_tensor(s, device=self.device)
                o.load_state_dict(optim_state_dict)
        # Set exploration's state.
        if hasattr(self, "exploration") and "_exploration_state" in state:
            self.exploration.set_state(state=state["_exploration_state"])
        # Then the Policy's (NN) weights.
        super().set_state(state)
```

Add the lines below: 
```python
class MyPGTorchPolicy(MyPGTorchPolicyIntermediaryStep):

    @override(Policy)
    @DeveloperAPI
    def set_state(self, state: dict) -> None:
        # Set optimizer vars first.
        optimizer_vars = state.get("_optimizer_variables", None)
        if optimizer_vars:
            assert len(optimizer_vars) == len(self._optimizers)
            for o, s in zip(self._optimizers, optimizer_vars):

# Fix
                for v in s["param_groups"]:
                    if "foreach" in v.keys():
                        v["foreach"] = False if v["foreach"] is None else v["foreach"]
                for v in s["state"].values():
                    if "momentum_buffer" in v.keys():
                        v["momentum_buffer"] = False if v["momentum_buffer"] is None else v["momentum_buffer"]

                optim_state_dict = convert_to_torch_tensor(s, device=self.device)
                o.load_state_dict(optim_state_dict)
        # Set exploration's state.
        if hasattr(self, "exploration") and "_exploration_state" in state:
            self.exploration.set_state(state=state["_exploration_state"])
        # Then the Policy's (NN) weights.
        super().set_state(state)
    ```