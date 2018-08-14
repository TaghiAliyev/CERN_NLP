#!/usr/bin/env bash
# First put in usual system-wide required programs



# Use requirements_generic.txt for python packages that should always be there


# Dynamic entry point for the NLP Model. This will also install and create local requirements.txt file
ENTRYPOINT["./init.sh"]