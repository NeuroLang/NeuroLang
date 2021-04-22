"""Configuration module for Relational Algebra Sets.
# TODO : Maybe move this to a root dir for Neurolang ?
"""
import configparser
import logging
import os

LOG = logging.getLogger(__name__)

config = configparser.ConfigParser()
config_file = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "config.ini"
)
LOG.info(f"Reading configuration file for Neurolang: {config_file}")
config.read(config_file)
LOG.info(f"Read config file with sections {config.sections()}")

