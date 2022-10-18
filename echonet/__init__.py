"""
The echonet package contains code for loading echocardiogram videos, and
functions for training and testing segmentation and ejection fraction
prediction models.
"""

import click

from echonet.__version__ import __version__
from echonet.config import CONFIG as config
import echonet.datasets as datasets
import echonet.utils as utils
import echonet.models as models
import echonet.segmentation as segmentation

@click.group()
def main():
    """Entry point for command line interface."""


del click

main.add_command(utils.seg_cycle.run)
main.add_command(utils.vidsegin_teachstd_kd.run)
main.add_command(utils.video_segin.run)


__all__ = ["__version__", "config", "datasets", "main", "utils", "models", "segmentation"]
