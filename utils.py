from dataclasses import dataclass
import itertools
import ray
from ray.actor import ActorHandle
from asyncio import Event
from typing import Tuple
from tqdm import tqdm
from copy import deepcopy
import hashlib
import json


def get_hash(config):
    # Convert the dictionary to a JSON string
    json_data = json.dumps(config, sort_keys=True)  # sort_keys ensures consistent order

    # Generate the hash (using SHA256 as an example)
    hash_object = hashlib.sha256(json_data.encode('utf-8'))
    hash_value = hash_object.hexdigest()
    return hash_value


@dataclass
class Config:
    ssm_type: str
    initA_real: str
    initA_imag: str
    discretizationA: str
    discretizationB: str
    param_A_imag: str
    A_imag_using_weight_decay: str
    dt_is_selective: str
    channel_sharing: str
    deterministic: bool
    pscan: bool
    d_model: int
    d_state: int
    n_layers: int
    n_categories: int
    lag: int
    extra: int
    batch_size: int
    epoch_size: int
    epochs: int
    lr: float
    stop_on_loss: float
    seed: int
    comment: str
    bias: bool


def experiments(kwargs):
    # Extract argument names and their corresponding value lists
    arg_names = [k[0] for k in kwargs]
    value_lists = [k[1] for k in kwargs]

    # Iterate over the Cartesian product of value lists
    for values in itertools.product(*value_lists):
        # Yield a dictionary mapping argument names to values
        yield dict(zip(arg_names, values))


def override_config(config, args):
    # return a new config with the values in args updated
    config = deepcopy(config)
    for arg in args:
        key, value = arg.split('=', 1)
        value = convert_type(value)
        keys = key.split('.')
        sub_config = config
        for k in keys[:-1]:
            sub_config = sub_config.setdefault(k, {})
        sub_config[keys[-1]] = value
    return config


def convert_type(value):
    # Convert to integer
    try:
        return int(value)
    except ValueError:
        pass

    # Convert to boolean
    if value.lower() in ['true', 'false']:
        return value.lower() == 'true'

    # Return string if no conversion possible
    return value


@ray.remote
class ProgressBarActor:
    counter: int
    delta: int
    event: Event

    def __init__(self) -> None:
        self.counter = 0
        self.delta = 0
        self.event = Event()

    def update(self, num_items_completed: int = 1) -> None:
        """Updates the ProgressBar with the incremental
        number of items that were just completed.
        """
        self.counter += num_items_completed
        self.delta += num_items_completed
        self.event.set()

    async def wait_for_update(self) -> Tuple[int, int]:
        """Blocking call.

        Waits until somebody calls `update`, then returns a tuple of
        the number of updates since the last call to
        `wait_for_update`, and the total number of completed items.
        """
        await self.event.wait()
        self.event.clear()
        saved_delta = self.delta
        self.delta = 0
        return saved_delta, self.counter

    def get_counter(self) -> int:
        """
        Returns the total number of complete items.
        """
        return self.counter


class ProgressBar:
    """
    Usage example:
    >> pb = ProgressBar()
    >> futures = [somefunc.remote(someinput, pb.actor) for someinput in someinputs]
    >> pb.set_total(len(futures))
    >> pb.print_until_done()
    """
    progress_actor: ActorHandle
    total: int
    description: str
    pbar: tqdm

    def __init__(self, description: str = ""):
        # Ray actors don't seem to play nice with mypy, generating
        # a spurious warning for the following line,
        # which we need to suppress. The code is fine.
        self.progress_actor = ProgressBarActor.remote()  # type: ignore
        self.description = description

    def set_total(self, total: int):
        self.total = total

    @property
    def actor(self) -> ActorHandle:
        """Returns a reference to the remote `ProgressBarActor`.

        When you complete tasks, call `update` on the actor.
        """
        return self.progress_actor

    def print_until_done(self) -> None:
        """Blocking call.

        Do this after starting a series of remote Ray tasks, to which you've
        passed the actor handle. Each of them calls `update` on the actor.
        When the progress meter reaches 100%, this method returns.
        """
        pbar = tqdm(desc=self.description, total=self.total,
                    smoothing=0)
        while True:
            delta, counter = ray.get(self.actor.wait_for_update.remote())
            pbar.update(delta)
            if counter >= self.total:
                pbar.close()
                return
