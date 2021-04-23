import configparser
import logging
import os
import sys

LOG = logging.getLogger(__name__)

config = configparser.ConfigParser()

config_dirs = [os.path.join(sys.prefix, 'config'), os.path.dirname(os.path.realpath(__file__))]
for d in config_dirs:
    config_file = os.path.join(d, "config.ini")
    if os.path.isfile(config_file):
        LOG.info(f"Reading configuration file for Neurolang: {config_file}")
        config.read(config_file)
        LOG.info(f"Read config file with sections {config.sections()}")
        break
