from time import time
import torch
from torch.optim import Adam
from cuas.models.constraint_model import ConstraintModel
from cuas.utils.replay_buffer import ReplayBuffer
from cuas.utils.list import for_each
import numpy as np
from datetime import datetime
import os
from qpsolvers import solve_qp, solve_ls
import scipy as sp
from cuas.utils.tensorboard import TensorBoard
from scipy.linalg import block_diag
import gym
import ray.tune as tune
import logging
import random

PATH = os.path.dirname(os.path.abspath(__file__))

logger = logging.getLogger(__name__)


class SafetyLayer:
    def __init__(self, env, config={}):
        self._env = env
        self._config = config

        self._parse_config()
        random.seed(self._seed)
        np.random.seed(self._seed)
        torch.manual_seed(self._seed)
        # torch.use_deterministic_algorithms(True)

        self._total_action_dim = self._env.action_space[0].shape[0]

        orig_space = getattr(
            self._env.observation_space[0],
            "original_space",
            self._env.observation_space[0],
        )
        assert (
            isinstance(orig_space, gym.spaces.Dict)
            and "raw_obs" in orig_space.spaces
            and "constraints" in orig_space.spaces
        )
        self._num_constraints = self._env.num_constraints
        self._num_agents = self._env.num_agents

        self._total_obs_dim = orig_space["raw_obs"].shape[0]
        self._total_c_dim = self._num_constraints * self._num_agents

        self._initialize_constraint_models()
        self._optimizers = [Adam(x.parameters(), lr=self._lr) for x in self._models]
        self._loss_function = torch.nn.MSELoss()
        self._replay_buffer = ReplayBuffer(self._replay_buffer_size)

        self._train_global_step = 0
        self._eval_global_step = 0
        self.num_corrective_actions = 0
        self.num_infeasible_solver = 0

        # use gpu if available
        # https://wandb.ai/wandb/common-ml-errors/reports/How-To-Use-GPU-with-PyTorch---VmlldzozMzAxMDk
        # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if torch.cuda.is_available():
            logger.debug("using cuda")
            self._device = torch.device("cuda")
            self._cuda()
        else:
            logger.debug("using cpu")
            self._device = torch.device("cpu")

    def _parse_config(self):
        self._replay_buffer_size = self._config.get("replay_buffer_size", 1000000)
        self._episode_length = self._config.get("episode_length", 400)
        self._lr = self._config.get("lr", 0.0001)
        self._batch_size = self._config.get("batch_size", 256)
        self._num_eval_steps = self._config.get("num_eval_step", 1500)
        self._num_steps_per_epoch = self._config.get("num_steps_per_epoch", 6000)
        self._num_epochs = self._config.get("num_epochs", 25)
        self._report_tune = self._config.get("report_tune", False)
        self._seed = self._config.get("seed", 123)

    def _as_tensor(self, ndarray, requires_grad=False):
        tensor = torch.Tensor(ndarray)
        tensor.requires_grad = requires_grad
        tensor = tensor.to(self._device)
        return tensor

    def _cuda(self):
        for_each(lambda x: x.cuda(), self._models)

    def _eval_mode(self):
        for_each(lambda x: x.eval(), self._models)

    def _train_mode(self):
        for_each(lambda x: x.train(), self._models)

    def _initialize_constraint_models(self):
        self._models = [
            ConstraintModel(self._total_obs_dim, self._total_action_dim)
            for _ in range(self._total_c_dim)
        ]

    def _sample_steps(self, num_steps):
        episode_length = 0

        env_obs = self._env.reset()

        for _ in range(num_steps):
            # constraints = env_obs["constraints"]
            # obs = env_obs["raw_obs"]
            # TODO: maybe we should unscale the action here. I'm assuming this will be taken care of by the network.
            actions = self._env.action_space.sample()
            env_obs_next, _, done, _ = self._env.step(actions)
            for key, action in actions.items():
                actions[key] = self._env._unscale_action(action)
            # actions = self._env.unscale_action(actions)
            # constraints_next = env_obs_next["constraints"]

            actions = np.concatenate([v for k, v in sorted(actions.items())], 0)
            new_action = np.reshape(actions, [self._num_agents, -1])

            constraints = np.concatenate(
                [v["constraints"] for k, v in sorted(env_obs.items())], 0
            )
            obs = np.concatenate([v["raw_obs"] for k, v in sorted(env_obs.items())], 0)
            constraints_next = np.concatenate(
                [v["constraints"] for k, v in sorted(env_obs_next.items())], 0
            )

            thetas = np.concatenate(
                [v["raw_obs"] for k, v in sorted(env_obs.items())], 0
            )
            thetas = np.reshape(obs, [self._num_agents, -1])
            thetas = thetas[:, 2]

            si_actions = []
            for i in range(self._num_agents):
                si_actions.append(np.dot(self._env.get_rl(thetas[i]), new_action[i]))

            si_actions = np.squeeze(np.reshape(np.array(si_actions), -1))
            self._replay_buffer.add(
                {
                    "actions": si_actions,
                    "observations": obs,
                    "c": constraints,
                    "c_next": constraints_next,
                }
            )

            env_obs = env_obs_next
            episode_length += 1

            if done["__all__"] or (episode_length == self._episode_length):
                obs = self._env.reset()
                episode_length = 0

    def _evaluate_batch(self, batch):
        observations = self._as_tensor(batch["observations"])
        actions = self._as_tensor(batch["actions"])
        c = self._as_tensor(batch["c"])
        c_next = self._as_tensor(batch["c_next"])

        observations = observations.reshape(-1, self._num_agents, self._total_obs_dim)
        actions = actions.reshape(-1, self._num_agents, self._total_action_dim)

        # c_next_predicted = []
        losses = []
        constraint_idx = 0
        for i in range(self._num_agents):
            obs = observations[:, i, :]
            action = actions[:, i, :]
            for j in range(self._num_constraints):
                gi = self._models[constraint_idx](obs)

                # TODO: remove bmm to make deterministic
                ci = c[:, constraint_idx] + torch.bmm(
                    gi.view(gi.shape[0], 1, -1), action.view(action.shape[0], -1, 1)
                ).view(-1)

                losses.append(self._loss_function(c_next[:, constraint_idx], ci))

                constraint_idx += 1

        return losses

    def _update_batch(self, batch):
        batch = self._replay_buffer.sample(self._batch_size)

        # update critic
        for_each(lambda x: x.zero_grad(), self._optimizers)
        losses = self._evaluate_batch(batch)
        for_each(lambda x: x.backward(), losses)
        for_each(lambda x: x.step(), self._optimizers)

        return np.asarray([x.item() for x in losses])

    def evaluate(self):
        # sample steps
        self._sample_steps(self._num_eval_steps)

        self._eval_mode()

        # compute losses
        losses = [
            list(map(lambda x: x.item(), self._evaluate_batch(batch)))
            for batch in self._replay_buffer.get_sequential(self._batch_size)
        ]

        losses = np.mean(np.concatenate(losses).reshape(-1, self._total_c_dim), axis=0)

        self._replay_buffer.clear()

        self._eval_global_step += 1
        self._train_mode()

        print(f"Validation completed, average loss {losses}")

        return losses

    def load_layer(self, checkpoint_dir):
        if checkpoint_dir is None or not os.path.exists(checkpoint_dir):
            logger.error("Please provide a valid checkpoint dir")
            return
        for index, model in enumerate(self._models):
            checkpoint_path = os.path.join(checkpoint_dir, f"constraint_{index}")
            model_state, _ = torch.load(
                checkpoint_path,
                map_location=torch.device("cpu"),
            )
            model.load_state_dict(model_state)
            model.eval()

    def get_hard_safe_action(self, obs, actions, c):
        """Use QP solver to calculate corrective action
        # https://scaron.info/blog/conversion-from-least-squares-to-quadratic-programming.html
        # https://scaron.info/blog/quadratic-programming-in-python.html
        """
        self._eval_mode()

        actions = np.concatenate(
            # [self._env._unscale_action(v) for k, v in sorted(actions.items())], 0
            [v for k, v in sorted(actions.items())],
            0,
        )
        c = np.concatenate([v for k, v in sorted(c.items())], 0)
        obs = np.concatenate([v for k, v in sorted(obs.items())], 0)

        obs = np.reshape(obs, [self._num_agents, -1])

        # (1) Problem Variables
        # Problem specific constants
        I = np.eye(self._total_action_dim * self._num_agents)
        ones = np.ones(self._total_action_dim * self._num_agents)

        # get values of G
        # G = np.zeros([self._total_c_dim, self._total_action_dim])
        # for i, x in enumerate(self._models):
        #     G[i, :] = x(self._as_tensor(obs)).data.cpu().numpy()

        G = []
        constraint_idx = 0
        for i in range(self._num_agents):
            gi = []
            for j in range(self._num_constraints):
                g = (
                    self._models[constraint_idx](self._as_tensor(obs[i]))
                    .data.cpu()
                    .numpy()
                )
                gi.append(g)
                constraint_idx += 1
            G.append(np.array(gi))

        G = block_diag(*tuple(G))

        # G = np.array([gi.data.cpu().numpy()])

        # G = np.array([x(self._as_tensor(obs)).data.cpu().numpy() for x in self._models])

        # (2) convert to QP form
        # cost function
        P = np.eye(self._total_action_dim * self._num_agents)
        q = -np.dot(P.T, actions)

        # constraint
        # A = np.concatenate([G, I, -I])
        A = np.concatenate([G])
        # ub = np.concatenate([-c, ones * 1.3, ones * 1.3])
        # ub = np.concatenate([-c - self._env._constraint_slack])
        ub = np.concatenate([-c])

        # # # https://courses.csail.mit.edu/6.867/wiki/images/a/a7/Qp-cvxopt.pdf
        # P = np.array([[1, 0], [0, 0]], dtype=np.float64)
        # q = np.array([3, 4], dtype=np.float64)
        # G = np.array([[-1, 0], [0, -1], [-1, -3], [2, 5], [3, 4]], dtype=np.float64)
        # h = np.array([0, 0, -15, 100, 80], dtype=np.float64)
        try:
            x = solve_qp(
                P.astype(np.float64),
                q.astype(np.float64),
                A.astype(np.float64),
                ub.astype(np.float64),
                None,
                None,
                None,
                None,
                # lb.astype(np.float64),
                # ub.astype(np.float64),
                solver="quadprog",
                # solver="osqp",
                # solver="cvxopt",
            )
        except Exception as e:
            print(f"exception: {e}")
            x = actions

        if x is None:
            self.num_infeasible_solver += 1
            # print(f"infeasible solver: {self.num_infeasible_solver}")
            x = actions
            # x = np.reshape(x, (self._num_agents, self._env.action_space[0].shape[0]))
            # actions = {i: x[i] for i in range(self._num_agents)}
            # return actions
            # return actions

        if np.linalg.norm(actions - x) > 1e-3:
            self.num_corrective_actions += 1
            # print(f"corrective action: {self.num_corrective_actions}")

        # x = actions
        x = np.reshape(x, (self._num_agents, self._env.action_space[0].shape[0]))
        actions = {i: x[i] for i in range(self._num_agents)}
        # actions = {i: np.clip(x[i], -1, 1) for i in range(self._num_agents)}

        return actions

    def get_soft_safe_action(self, obs, actions, c):
        """Use QP solver to calculate corrective action
        # https://scaron.info/blog/conversion-from-least-squares-to-quadratic-programming.html
        # https://scaron.info/blog/quadratic-programming-in-python.html
        """
        self._eval_mode()

        actions = np.concatenate([v for k, v in sorted(actions.items())], 0)
        c = np.concatenate([v for k, v in sorted(c.items())], 0)
        obs = np.concatenate([v for k, v in sorted(obs.items())], 0)

        obs = np.reshape(obs, [self._num_agents, -1])
        # (1) Create solver as a global variable
        l1_penalty = 1000

        # (1) Problem Variables
        # Problem specific constants
        I = np.eye(
            self._total_action_dim * self._num_agents,
            self._total_action_dim * self._num_agents,
        )
        Z = np.zeros(
            [
                self._total_action_dim * self._num_agents,
                self._total_action_dim * self._num_agents,
            ]
        )
        ones = np.ones(self._total_action_dim * self._num_agents)
        zeros = np.zeros(self._total_action_dim * self._num_agents)

        # get values of G
        # G = np.zeros([self._total_c_dim, self._total_action_dim])
        # for i in range(8):
        #     G[i, :] = self._models[i](self._as_tensor(obs)).data.cpu().numpy()
        # for i, x in enumerate(self._models):
        # G[i, :] = x(self._as_tensor(obs)).data.cpu().numpy()

        # G = np.array([x(self._as_tensor(obs)).data.cpu().numpy() for x in self._models])

        G = []
        constraint_idx = 0
        for i in range(self._num_agents):
            gi = []
            for j in range(self._num_constraints):
                g = (
                    self._models[constraint_idx](self._as_tensor(obs[i]))
                    .data.cpu()
                    .numpy()
                )
                gi.append(g)
                constraint_idx += 1
            G.append(np.array(gi))

        G = block_diag(*tuple(G))

        # (2) convert to QP form
        # cost function
        P = sp.linalg.block_diag(I, Z + I * 0.000001, Z + I * 0.000001)
        # q = -np.dot(P.T, actions)
        q = np.concatenate([-actions, ones, zeros])

        # Constraints
        A = np.vstack(
            (
                np.concatenate(
                    [
                        G,
                        np.zeros(
                            [
                                self._total_c_dim,
                                self._total_action_dim * self._num_agents,
                            ]
                        ),
                        -np.eye(
                            self._total_c_dim, self._total_action_dim * self._num_agents
                        ),
                    ],
                    axis=1,
                ),
                np.concatenate([Z, Z, -I], axis=1),
                np.concatenate([Z, -I, l1_penalty * I], axis=1),
                np.concatenate([Z, -I, -l1_penalty * I], axis=1),
            )
        )

        ub = np.concatenate((-c, zeros, zeros, zeros))
        # ub = np.concatenate((-c + self._env._constraint_slack, zeros, zeros, zeros))
        # vub = np.ones(self._total_action_dim)
        # vub[::2] = vub[::2] * 1
        # vlb = -vub

        # # https://courses.csail.mit.edu/6.867/wiki/images/a/a7/Qp-cvxopt.pdf
        # P = np.array([[1, 0], [0, 0]], dtype=np.float64)
        # q = np.array([3, 4], dtype=np.float64)
        # A = np.array([[-1, 0], [0, -1], [-1, -3], [2, 5], [3, 4]], dtype=np.float64)
        # ub = np.array([0, 0, -15, 100, 80], dtype=np.float64)

        try:
            x = solve_qp(
                P.astype(np.float64),
                q.astype(np.float64),
                A.astype(np.float64),
                ub.astype(np.float64),
                # vub.astype(np.float64),
                # vlb.astype(np.float64),
                None,
                None,
                None,
                None,
                solver="quadprog",
                # solver="quadprog",
                # solver="osqp"
                # solver="cvxopt",
            )
        except Exception as e:
            # print(f"error running solver: {e}")
            self.num_infeasible_solver += 1
            x = actions

        if x is None:
            self.num_infeasible_solver += 1
            # print(f"infeasible solver: {self.num_infeasible_solver}")
            x = actions
            # return actions
        else:
            x = x[: self._total_action_dim * self._num_agents]

        if np.linalg.norm(actions - x) > 1e-3:
            self.num_corrective_actions += 1
            # print(f"corrective action: {self.num_corrective_actions}")

        x = np.reshape(x, (self._num_agents, self._env.action_space[0].shape[0]))

        # x = np.split(x, self._num_agents)
        # x = [np.clip(np.squeeze(action), -1, 1) for action in x]
        actions = {i: np.clip(x[i], -1, 1) for i in range(self._num_agents)}
        # actions = {i: action[i] for i in range(self._num_agents)}
        return actions

    def get_safe_actions(self, observations, actions):

        self._eval_mode()
        for key, action in actions.items():
            actions[key] = self._env._unscale_action(action)

        actions = np.concatenate([v for k, v in sorted(actions.items())], 0)
        actions = np.reshape(actions, [self._num_agents, -1])
        c = np.concatenate(
            [v["constraints"] for k, v in sorted(observations.items())], 0
        )
        obs = np.concatenate([v["raw_obs"] for k, v in sorted(observations.items())], 0)
        thetas = np.concatenate(
            [v["raw_obs"] for k, v in sorted(observations.items())], 0
        )
        thetas = np.reshape(obs, [self._num_agents, -1])
        thetas = thetas[:, 2]

        obs = np.reshape(obs, [self._num_agents, -1])

        # G = []
        multipliers = []
        g = []
        corrections = []
        constraint_idx = 0
        for i in range(self._num_agents):
            action = actions[i]
            multipliers = []
            g = []
            for j in range(self._num_constraints):
                gi = (
                    self._models[constraint_idx](self._as_tensor(obs[i]))
                    .data.cpu()
                    .numpy()
                )

                multipliers.append(
                    (
                        np.dot(gi, np.dot(self._env.get_rl(thetas[i]), action))
                        # np.dot(gi, action)
                        + c[constraint_idx]
                        + self._env._constraint_slack
                    )
                    / np.dot(gi, gi)
                )
                g.append(gi)
                constraint_idx += 1

            multipliers = [np.clip(x, 0, np.inf) for x in multipliers]
            correction = np.max(multipliers) * g[np.argmax(multipliers)]
            corrections.append(correction)
            # G.append(np.array(gi))

        # G = block_diag(*tuple(G))

        # multipliers = (np.dot(G, actions) + c) / np.dot(G.T, G)

        # g = [x(self._as_tensor(obs).view(1, -1)) for x in self._models]

        # g = [x.data.cpu().numpy().reshape(-1) for x in g]

        # multipliers = [
        #     (np.dot(g_i, actions) + c_i) / np.dot(g_i, g_i) for g_i, c_i in zip(g, c)
        # ]

        # multipliers = [np.clip(x, 0, np.inf) for x in multipliers]
        #
        # correction = np.max(multipliers) * g[np.argmax(multipliers)]

        corrections = np.array(corrections)
        action_new = actions - corrections

        action_new = np.reshape(
            action_new, (self._num_agents, self._env.action_space[0].shape[0])
        )
        action_new = {i: action_new[i] for i in range(self._num_agents)}
        # action_new = {i: np.clip(x[i], -1, 1) for i in range(self._num_agents)}

        return action_new

    def get_safe_action(self, obs, action, c):

        # get values of G
        self._eval_mode()

        g = [x(self._as_tensor(obs).view(1, -1)) for x in self._models]

        # find lagrange multipliers
        g = [x.data.cpu().numpy().reshape(-1) for x in g]

        multipliers = [
            (np.dot(g_i, action) + c_i) / np.dot(g_i, g_i) for g_i, c_i in zip(g, c)
        ]

        multipliers = [np.clip(x, 0, np.inf) for x in multipliers]

        # calculate correction
        correction = np.max(multipliers) * g[np.argmax(multipliers)]

        action_new = action - correction

        if np.linalg.norm(action - action_new) > 1e-3:
            self.num_corrective_actions += 1
            # print(f"using corrective action: {self.num_corrective_actions}")

        action_new = np.clip(action_new, -1, 1)

        return action_new

    def train(self):
        """Train Step"""
        start_time = time()

        print("==========================================================")
        print("Initializing constraint model training...")
        print("----------------------------------------------------------")
        print(f"Start time: {datetime.fromtimestamp(start_time)}")
        print("==========================================================")

        for epoch in range(self._num_epochs):
            # sample episodes from whole epoch
            self._sample_steps(self._num_steps_per_epoch)

            # update from memory
            losses = np.mean(
                np.concatenate(
                    [
                        self._update_batch(batch)
                        for batch in self._replay_buffer.get_sequential(
                            self._batch_size
                        )
                    ]
                ).reshape(-1, self._total_c_dim),
                axis=0,
            )

            self._replay_buffer.clear()
            self._train_global_step += 1

            print(f"Finished epoch {epoch} with losses: {losses}.")

            if self._report_tune:
                results = {"training_iteration": self._train_global_step}
                for i in range(self._total_c_dim):
                    results[f"constraint_{i}_loss"] = losses[i]

                if ((epoch + 1) % 5 == 0) or (epoch == 0):
                    print("Running validation ...")
                    val_losses = self.evaluate()

                    with tune.checkpoint_dir(epoch) as checkpoint_dir:
                        for i in range(self._total_c_dim):
                            results[f"constraint_{i}_val_loss"] = val_losses[i]
                            path = os.path.join(checkpoint_dir, f"constraint_{i}")
                            torch.save(
                                (
                                    self._models[i].state_dict(),
                                    self._optimizers[i].state_dict(),
                                ),
                                path,
                            )

                tune.report(**results)

        print("==========================================================")
        print(
            f"Finished training constraint model. Time spent: {(time() - start_time) // 1} secs"
        )
        print("==========================================================")
