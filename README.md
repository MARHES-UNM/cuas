# Multi-Agent Counter UAS (CUAS) Using Reinforcement Learning

![Multi Agent Purser Evader](images/cuas_deepset.gif)

A Gym Environment for Counter UAS

conda install tensorflow-gpu


To train model: 
```bash
python run_experiment.py train --duration <time in seconds>
```


To test model:
```bash
python run_experiment.py test --checkpoint <checkpoint>
```



https://github.com/sisl/MADRL
https://github.com/ChanganVR/CADRL/blob/master/train.py


pip install -U "ray[tune,rllib,serve]"
pip install -U "ray[rllib]"
conda install -c conda-forge opencv


Issue with ray 1.13 and PyTorch 1.13: 
https://github.com/ray-project/ray/issues/26781
https://github.com/ray-project/ray/issues/26557

change line 765 in torch_policy.py located at: `<your anaconda3 path>/envs/py3915/lib/python3.9/site-packages/ray/rllib/policy/torch_policy.py`


```python 
    def set_state(self, state: dict) -> None:
        # Set optimizer vars first.
        optimizer_vars = state.get("_optimizer_variables", None)
        if optimizer_vars:
            assert len(optimizer_vars) == len(self._optimizers)
            for o, s in zip(self._optimizers, optimizer_vars):
#start fix
                for v in s["param_groups"]:
                    if "foreach" in v.keys():
                        v["foreach"] = False if v["foreach"] is None else v["foreach"]
                for v in s["state"].values():
                    if "momentum_buffer" in v.keys():
                        v["momentum_buffer"] = False if v["momentum_buffer"] is None else v["momentum_buffer"]
#end fix
                optim_state_dict = convert_to_torch_tensor(s, device=self.device)
                o.load_state_dict(optim_state_dict)
        # Set exploration's state.
        if hasattr(self, "exploration") and "_exploration_state" in state:
            self.exploration.set_state(state=state["_exploration_state"])
        # Then the Policy's (NN) weights.
        super().set_state(state)
```