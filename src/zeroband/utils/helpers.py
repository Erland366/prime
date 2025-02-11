import sys
import wandb
import os

from zeroband.utils.logging import get_logger
from huggingface_hub import login, whoami

def float_to_e_formatting(number: float) -> str:
    return f"{number:.1e}"

def is_debug_py():
    return "debugpy" in sys.modules

def login_hf(environ_name: str = "HF_TOKEN", token: str | None = None):
    logger = get_logger()
    if token is None:
        token = os.getenv(environ_name)
        logger.debug(f"Use token from environment variable {environ_name}")
    login(token=token)
    print(f"Hugging Face user: {whoami()['name']}, full name: {whoami()['fullname']}")


def login_wandb(environ_name: str = "WANDB_API_KEY", token: str | None = None, **kwargs):
    logger = get_logger()

    if token is None:
        token = os.getenv(environ_name)
        logger.debug(f"Use token from environment variable {environ_name}")
    wandb.login(key=token, **kwargs)