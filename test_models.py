### https://docs.python.org/3.9/library/concurrent.futures.html
### https://rsmith.home.xs4all.nl/programming/parallel-execution-with-python.html
### https://docs.python.org/3.9/library/concurrent.futures.html
import concurrent.futures
import os
import time
from functools import partial
import logging
import pathlib
import json
import subprocess
from datetime import datetime

max_num_episodes = 10
# max_num_cpus = 16
max_num_cpus = os.cpu_count() - 1

continue_experiment = False
# override = True

formatter = "%(asctime)s: %(name)s - %(levelname)s - <%(module)s:%(funcName)s:%(lineno)d> - %(message)s"
logging.basicConfig(
    # filename=os.path.join(app_log_path, log_file_name),
    format=formatter
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

PATH = pathlib.Path(__file__).parent.absolute().resolve()


branch_hash = (
        subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
        .strip()
        .decode()
    )

dir_timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")

log_dir = f"./results/test_results/exp_{dir_timestamp}_{branch_hash}"


if not os.path.exists(log_dir):
    os.makedirs(log_dir)

checkpoints = {
            'torch_fix': r"/home/prime/Documents/workspace/cuas/results/ppo/cuas_4v1o5_2023-01-08-01-47_6676de3/local_50e6_cpu_8/MyTrainer_cuas_multi_agent-v1_4v1o5_60b4f_00000_0_beta=0.0900,observation_type=local,use_safe_action=False,custom_model=TorchFixMo_2023-01-08_01-47-54/checkpoint_000747/checkpoint-747",
            "torch_fix_safe": r"/home/prime/Documents/workspace/cuas/results/ppo/cuas_4v1o5_2023-01-08-01-47_6676de3/local_50e6_cpu_8/MyTrainer_cuas_multi_agent-v1_4v1o5_60b4f_00001_1_beta=0.0900,observation_type=local,use_safe_action=True,custom_model=TorchFixMod_2023-01-08_01-47-59/checkpoint_000746/checkpoint-746",
            "deepset": r"/home/prime/Documents/workspace/cuas/results/ppo/cuas_4v1o5_2023-01-08-01-47_6676de3/local_50e6_cpu_8/MyTrainer_cuas_multi_agent-v1_4v1o5_60b4f_00002_2_beta=0.0900,observation_type=local,use_safe_action=False,custom_model=DeepsetMod_2023-01-08_08-13-36/checkpoint_000748/checkpoint-748",
            "deepset_safe": r"/home/prime/Documents/workspace/cuas/results/ppo/cuas_4v1o5_2023-01-08-01-47_6676de3/local_50e6_cpu_8/MyTrainer_cuas_multi_agent-v1_4v1o5_60b4f_00003_3_beta=0.0900,observation_type=local,use_safe_action=True,custom_model=DeepsetMode_2023-01-08_08-54-48/checkpoint_000747/checkpoint-747",
            "lstm": r"/home/prime/Documents/workspace/cuas/results/ppo/cuas_4v1o5_2023-02-02-08-31_5cc3baf/local_50e6_lstm/MyTrainer_cuas_multi_agent-v1_4v1o5_d9e04_00000_0_observation_type=local,use_safe_action=False,custom_model=LstmModel_2023-02-02_08-31-08/checkpoint_000748/checkpoint-748",
            "lstm_safe": r"/home/prime/Documents/workspace/cuas/results/ppo/cuas_4v1o5_2023-02-02-08-31_5cc3baf/local_50e6_lstm/MyTrainer_cuas_multi_agent-v1_4v1o5_d9e04_00001_1_observation_type=local,use_safe_action=True,custom_model=LstmModel_2023-02-02_08-31-14/checkpoint_000746/checkpoint-746"}
seeds = [5000, 0, 173, 93, 507, 1001]
# seeds = [5000, ]
num_obstacles = [2, 6, 10]
num_pursuers = [2, 4, 10]


seeds = [123]
num_obstacles = [5]
num_pursuers = [4]

exp_configs = []
idx = 0
for checkpoint_name, checkpoint in checkpoints.items():
    for seed in seeds:
        for obstacle in num_obstacles:
            for pursuer in num_pursuers:
                exp_configs.append({'checkpoint': checkpoint, 'seed': seed, 'obstacle': obstacle, 'pursuer': pursuer, 'exp_name': f'exp_{idx}_{checkpoint_name}_s{seed}_o{obstacle}_p{pursuer}'})
                idx += 1

def run_experiment(exp_config):
    # print(f"running experiment: {exp_config}")
    default_config=f"{PATH}/configs/sim_config.cfg"
    with open(default_config, "rt") as f:
            config = json.load(f)
            config['checkpoint'] = exp_config['checkpoint']
            config['test_config'] = {'seed': exp_config['seed']}
    
            config["restore_checkpoint"] = True
            config["update_environment"] = True
            config["env_config"]["observation_type"] = "local"
            config["test_env_config"] = {
                "observation_radius": 20,
                    "num_obstacles": exp_config['obstacle'],
                    "num_pursuers": exp_config["pursuer"]
                }

    output_folder = os.path.join(log_dir, exp_config['exp_name'])
    exp_file_config = os.path.join(output_folder, "exp_sim_config.cfg")
    fname = os.path.join(output_folder, "result.json")
    
    # test to make sure if result exist

    if continue_experiment:
        try:
            with open(fname, "r") as f:
                data = json.loads(f.readlines()[-1])
                print(data['config'])
            
        except Exception as e: 
            print(f"{fname}: error reading file. Will start experiment again just in case")
        else:
            print(f"{fname}: no issues loading data. Skipping this experiment")
            return 0
                
            
        
    config['fname'] = fname
    config['write_experiment'] = True

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    with open(exp_file_config, 'w') as f:
        json.dump(config, f)

    args = [
        'python', 'run_experiment.py', '--log_dir', f'{output_folder}', '--load_config', str(exp_file_config), "test", "--max_num_episodes", str(max_num_episodes), 
        "--checkpoint", config['checkpoint'],
        # "--video",
        "--no_render"
    ]

    rv = subprocess.call(args)
    # rv = subprocess.call(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
    print(f"{exp_config['exp_name']} done running.")

    return rv

starter = partial(run_experiment)
err_msg = 'running '

with concurrent.futures.ThreadPoolExecutor(max_workers=max_num_cpus) as executor:
    future_run_experiment = [executor.submit(starter, exp_config) for exp_config in exp_configs]
    # future_run_experiment = {executor.submit(run_experiment, exp_config) exp_config for exp_config in exp_configs }
    for future in concurrent.futures.as_completed(future_run_experiment):
        rv = future.result()
        
        
        