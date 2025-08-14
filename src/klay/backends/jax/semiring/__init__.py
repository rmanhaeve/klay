from klay.backends.jax.semiring.log import log_sum_layer, encode_input_log
from klay.backends.jax.semiring.real import sum_layer, prod_layer, encode_input_real


def get_semiring(name: str):
    if name == 'real':
        return sum_layer, prod_layer
    elif name == 'log':
        return log_sum_layer, sum_layer
    else:
        raise ValueError(f"Unknown semiring {name}")


def encode_input(name: str):
    return {'log': encode_input_log, 'real': encode_input_real}[name]
