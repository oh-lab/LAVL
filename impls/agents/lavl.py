import copy
from typing import Any

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import ml_collections
import optax
from utils.encoders import GCEncoder, encoder_modules
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import GCActor, GCDiscreteActor, LANValue, MLP, LengthNormalize, Identity


class LAVLAgent(flax.struct.PyTreeNode):
    """
    Latent Alignment Value Learning (LAVL) agent.
    """
    rng: Any
    network: Any
    ema_v_mean: Any
    config: Any = nonpytree_field()

    @staticmethod
    def expectile_loss(adv, diff, expectile):
        """Compute the expectile loss."""
        weight = jnp.where(adv >= 0, expectile, (1 - expectile))
        return weight * (diff**2)

    @staticmethod
    def length_normalize(x):
        return x / jnp.linalg.norm(x, axis=-1, keepdims=True) * jnp.sqrt(x.shape[-1])

    def value_loss(self, batch, grad_params):
        next_v_t = -self.network.select('target_value')(batch['next_observations'], batch['value_goals'])
        q = batch['rewards'] + self.config['discount'] * batch['masks'] * next_v_t

        v_t = -self.network.select('target_value')(batch['observations'], batch['value_goals'])
        adv = q - v_t

        v = -self.network.select('value')(batch['observations'], batch['value_goals'], params=grad_params)

        value_loss = self.expectile_loss(adv, q - v, self.config['expectile']).mean()

        threshold = 1.0 - (1.0 - self.config['discount']) * jax.lax.stop_gradient(self.ema_v_mean)
        v_random = -self.network.select('value')(batch['observations'], batch['random_goals'], params=grad_params)
        v_next_random = -self.network.select('value')(batch['next_observations'], batch['random_goals'], params=grad_params)
        local_smoothness_loss = jnp.mean(0.1 *nn.softplus(10 * ((v_next_random - v_random)**2 - threshold**2)))

        total_loss =  value_loss + self.config['smoothness_weight'] * local_smoothness_loss
        
        return total_loss, {
            'value_loss': value_loss,
            'smoothness_loss': local_smoothness_loss,
            'v_mean': v.mean(),
            'v_max': v.max(),
            'v_min': v.min(),
        }

    def high_actor_loss(self, batch, grad_params, rng=None):
        target_reps = self.network.select('rep')(batch['high_actor_targets'], params=grad_params)
        if not self.config['high_actor_rep_grad']:
            target_reps = jax.lax.stop_gradient(target_reps)

        v = -self.network.select('value')(batch['observations'], batch['high_actor_goals'])
        nv = -self.network.select('value')(batch['high_actor_targets'], batch['high_actor_goals'])
        adv = nv - v

        exp_a = jnp.exp(adv * self.config['high_alpha'])
        exp_a = jnp.minimum(exp_a, 100.0)
        if self.config['normalize_adv']:
            exp_a = exp_a / (jnp.mean(exp_a) + 1e-3)

        dist = self.network.select('high_actor')(batch['observations'], batch['high_actor_goals'], params=grad_params)
        log_prob = dist.log_prob(target_reps)

        actor_loss = -(exp_a * log_prob).mean()

        actor_info = {
            'actor_loss': actor_loss,
            'adv': adv.mean(),
            'exp_a': exp_a.mean(),
            'bc_log_prob': log_prob.mean(),
        }
        if not self.config['discrete']:
            actor_info.update(
                {
                    'mse': jnp.mean((dist.mode() - target_reps) ** 2),
                    'std': jnp.mean(dist.scale_diag),
                }
            )
        return actor_loss, actor_info

    def low_actor_loss(self, batch, grad_params, rng=None):
        goal_reps = self.network.select('rep')(batch['low_actor_goals'], params=grad_params)
        if not self.config['low_actor_rep_grad']:
            goal_reps = jax.lax.stop_gradient(goal_reps)

        v = -self.network.select('value')(batch['observations'], batch['low_actor_goals'])
        nv = -self.network.select('value')(batch['next_observations'], batch['low_actor_goals'])
        adv = nv - v

        exp_a = jnp.exp(adv * self.config['low_alpha'])
        exp_a = jnp.minimum(exp_a, 100.0)
        if self.config['normalize_adv']:
            exp_a = exp_a / (jnp.mean(exp_a) + 1e-3)

        dist = self.network.select('low_actor')(batch['observations'], goal_reps, goal_encoded=True, params=grad_params)
        log_prob = dist.log_prob(batch['actions'])

        actor_loss = -(exp_a * log_prob).mean()

        actor_info = {
            'actor_loss': actor_loss,
            'adv': adv.mean(),
            'exp_a': exp_a.mean(),
            'bc_log_prob': log_prob.mean(),
        }
        if not self.config['discrete']:
            actor_info.update(
                {
                    'mse': jnp.mean((dist.mode() - batch['actions']) ** 2),
                    'std': jnp.mean(dist.scale_diag),
                }
            )
        return actor_loss, actor_info
        
       
    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        """Compute the total loss."""
        info = {}
        rng = rng if rng is not None else self.rng

        value_loss, value_info = self.value_loss(batch, grad_params)
        for k, v in value_info.items():
            info[f'value/{k}'] = v

        rng, high_actor_rng = jax.random.split(rng)
        high_actor_loss, high_actor_info = self.high_actor_loss(batch, grad_params, high_actor_rng)
        for k, v in high_actor_info.items():
            info[f'high_actor/{k}'] = v

        rng, low_actor_rng = jax.random.split(rng)
        low_actor_loss, low_actor_info = self.low_actor_loss(batch, grad_params, low_actor_rng)
        for k, v in low_actor_info.items():
            info[f'low_actor/{k}'] = v

        loss = value_loss + high_actor_loss + low_actor_loss
        return loss, info

    def target_update(self, network, module_name):
        """Update the target network."""
        new_target_params = jax.tree_util.tree_map(
            lambda p, tp: p * self.config['tau'] + tp * (1 - self.config['tau']),
            self.network.params[f'modules_{module_name}'],
            self.network.params[f'modules_target_{module_name}'],
        )
        network.params[f'modules_target_{module_name}'] = new_target_params

    @jax.jit
    def update(self, batch):
        """Update the agent and return a new agent with information dictionary."""
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params, rng=rng)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        self.target_update(new_network, 'value')
        self.target_update(new_network, 'rep')
        new_ema_v_mean = (1 - self.config['tau']) * self.ema_v_mean + self.config['tau'] * jax.lax.stop_gradient(info['value/v_mean'])
        info['ema_v_mean'] = new_ema_v_mean

        return self.replace(network=new_network, rng=new_rng, ema_v_mean=new_ema_v_mean), info

    @jax.jit
    def sample_actions(
        self,
        observations,
        goals=None,
        seed=None,
        temperature=1.0,
    ):
        """Sample actions from the actor."""
        high_seed, low_seed = jax.random.split(seed)

        high_dist = self.network.select('high_actor')(observations, goals, temperature=temperature)
        goal_reps = high_dist.sample(seed=high_seed)
        goal_reps = self.length_normalize(goal_reps)

        low_dist = self.network.select('low_actor')(observations, goal_reps, goal_encoded=True, temperature=temperature)
        actions = low_dist.sample(seed=low_seed)

        if not self.config['discrete']:
            actions = jnp.clip(actions, -1, 1)
        return actions
        
    @jax.jit
    def compute_value(self, observations, goals):
        """Compute the value of (observations, goals)."""
        return -self.network.select('value')(observations, goals)

    @classmethod
    def create(
        cls,
        seed,
        ex_observations,
        ex_actions,
        config,
    ):
        """Create a new agent.

        Args:
            seed: Random seed.
            ex_observations: Example observations.
            ex_actions: Example batch of actions. In discrete-action MDPs, this should contain the maximum action value.
            config: Configuration dictionary.
        """
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)

        ex_goals = ex_observations
        ex_goal_reps = np.zeros((ex_observations.shape[0], config['rep_dim']), dtype=np.float32)
        if config['discrete']:
            action_dim = ex_actions.max() + 1
        else:
            action_dim = ex_actions.shape[-1]
        obs_dim = ex_observations.shape[-1]
        config['obs_dim'] = obs_dim
        config['action_dim'] = action_dim

        # Define representation networks and encoders.
        if config['encoder'] is not None:
            encoder_module = encoder_modules[config['encoder']]
            rep_seq = [encoder_module()]
        else:
            rep_seq = []
        rep_seq.append(
            MLP(
                hidden_dims=(*config['value_hidden_dims'], config['rep_dim']),
                activate_final=False,
                layer_norm=config['layer_norm'],
            )
        )
        rep_seq.append(LengthNormalize())
        rep_def = nn.Sequential(rep_seq)
        target_rep_def = copy.deepcopy(rep_def)

        encoders = dict()
        if config['encoder'] is not None:
            encoders['state'] = encoder_module()
            encoders['target_state'] = encoder_module()
            encoders['high_actor'] = GCEncoder(state_encoder=encoder_module(), goal_encoder=encoder_module())
            encoders['low_actor'] = GCEncoder(state_encoder=encoder_module(), goal_encoder=rep_def)
        else:
            encoders['low_actor'] = GCEncoder(state_encoder=Identity(), goal_encoder=rep_def)

        value_def = LANValue(
            hidden_dims=config['value_hidden_dims'],
            latent_dim=config['latent_dim'],
            layer_norm=config['layer_norm'],
            state_encoder=encoders.get('state'),
            goal_encoder=rep_def
        )
        target_value_def = LANValue(
            hidden_dims=config['value_hidden_dims'],
            latent_dim=config['latent_dim'],
            layer_norm=config['layer_norm'],
            state_encoder=encoders.get('target_state'),
            goal_encoder=target_rep_def
        )

        # Define actor networks.
        high_actor_def = GCActor(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=config['rep_dim'],
            state_dependent_std=False,
            const_std=config['const_std'],
            gc_encoder=encoders.get('high_actor'),
        )

        if config['discrete']:
            low_actor_def = GCDiscreteActor(
                hidden_dims=config['actor_hidden_dims'],
                action_dim=action_dim,
                gc_encoder=encoders.get('low_actor'),
            )
        else:
            low_actor_def = GCActor(
                hidden_dims=config['actor_hidden_dims'],
                action_dim=action_dim,
                state_dependent_std=False,
                const_std=config['const_std'],
                gc_encoder=encoders.get('low_actor'),
            )

        network_info = dict(
            rep=(rep_def, (ex_observations,)),
            target_rep=(target_rep_def, (ex_observations,)),
            value=(value_def, (ex_observations, ex_goals)),
            target_value=(target_value_def, (ex_observations, ex_goals)),
            high_actor=(high_actor_def, (ex_observations, ex_goals)),
            low_actor=(low_actor_def, (ex_observations, ex_goals))
        )

        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        # network_tx = optax.adam(learning_rate=config['lr'])
        network_tx = optax.chain(
            optax.clip_by_global_norm(1000.0),
            optax.adam(learning_rate=config['lr']),
        )

        network_params = network_def.init(init_rng, **network_args)['params']
        network = TrainState.create(network_def, network_params, tx=network_tx)

        params = network_params
        params['modules_target_value'] = params['modules_value']
        params['modules_target_rep'] = params['modules_rep']

        return cls(rng, network=network, config=flax.core.FrozenDict(**config),
                   ema_v_mean=-1.0/(1-config['discount']))

def get_config():
    config = ml_collections.ConfigDict(
        dict(
            # Agent hyperparameters.
            agent_name='lavl',  # Agent name.
            lr=3e-4,  # Learning rate.
            batch_size=1024,  # Batch size.
            actor_hidden_dims=(512, 512, 512),  # Actor network hidden dimensions.
            value_hidden_dims=(512, 512, 512),  # Value network hidden dimensions.
            latent_dim=64,  # Latent dimension for the quasimetric value function.
            rep_dim=10, # Goal representation dimension.
            layer_norm=True,  # Whether to use layer normalization.
            discount=0.999,  # Discount factor.
            tau=0.005,  # Target network update rate.
            expectile=0.9,  # IQL expectile.
            smoothness_weight=0.0,  # Weight for the smoothness regularization term.
            high_alpha=3.0,  # AWR temperature for the high-level actor.
            low_alpha=3.0,  # AWR temperature for the low-level actor.
            const_std=True,  # Whether to use constant standard deviation for the low-level actor.
            discrete=False,  # Whether the action space is discrete.
            normalize_adv=True,  # Whether to normalize advantages when computing actor loss.
            high_actor_rep_grad=False,  # Whether high-actor gradients flow to goal representation.
            low_actor_rep_grad=False,  # Whether low-actor gradients flow to goal representation.
            encoder=ml_collections.config_dict.placeholder(str),  # Visual encoder name (None, 'impala_small', etc.).
            # Dataset hyperparameters.
            dataset_class='HGCDataset',  # Dataset class name.
            subgoal_steps=25,  # Number of steps between subgoals.
            value_p_curgoal=0.2,  # Probability of using the current state as the value goal.
            value_p_trajgoal=0.5,  # Probability of using a future state in the same trajectory as the value goal.
            value_p_randomgoal=0.3,  # Probability of using a random state as the value goal.
            value_geom_sample=True,  # Whether to use geometric sampling for future value goals.
            actor_p_curgoal=0.0,  # Probability of using the current state as the actor goal.
            actor_p_trajgoal=0.5,  # Probability of using a future state in the same trajectory as the actor goal.
            actor_p_randomgoal=0.5,  # Probability of using a random state as the actor goal.
            actor_geom_sample=False,  # Whether to use geometric sampling for future actor goals.
            gc_negative=True,  # Whether to use '0 if s == g else -1' (True) or '1 if s == g else 0' (False) as reward.
            p_aug=0.0,  # Probability of applying image augmentation.
            frame_stack=ml_collections.config_dict.placeholder(int),  # Number of frames to stack.
        )
    )
    return config
