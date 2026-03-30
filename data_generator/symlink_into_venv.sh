#!/bin/bash

# VIRTUAL_ENV holds the environment location.


# check if that is set, if it is not, quit with a message
if [ -z "$VIRTUAL_ENV" ]; then
    echo "No virtual environment activated. Please activate one before running this script."
    exit 1
fi

# For now, lets assume debug.
RELEASE_TYPE=debug

NAME_IN_CARGO=$(grep -Po "(?<=name ?= ?\")([^\"]+)" Cargo.toml -m 1)

# cpython-313-x86_64-linux-gnu
PYTHONVERSION_WITHDOT=$(python3 -c "import platform; t=platform.python_version_tuple();print(f'{t[0]}.{t[1]}')")
PYTHONVERSION_CONCAT=$(python3 -c "import platform; t=platform.python_version_tuple();print(f'{t[0]}{t[1]}')")

PACKAGE_NAME=${NAME_IN_CARGO}
PACKAGE_DIR=${VIRTUAL_ENV}/lib/python${PYTHONVERSION_WITHDOT}/site-packages/${PACKAGE_NAME}
mkdir -p ${PACKAGE_DIR}


TARGETLIBPATH=$(realpath ./target/${RELEASE_TYPE}/lib${PACKAGE_NAME}.so)

ln -s ${TARGETLIBPATH}  ${PACKAGE_DIR}/${PACKAGE_NAME}.cpython-${PYTHONVERSION_CONCAT}-x86_64-linux-gnu.so

read -r -d '' INIT_FILE_PAYLOAD <<EOF
from .${PACKAGE_NAME} import *

__doc__ = ${PACKAGE_NAME}.__doc__
if hasattr(${PACKAGE_NAME}, "__all__"):
    __all__ = ${PACKAGE_NAME}.__all__
EOF


echo "${INIT_FILE_PAYLOAD}" >  ${PACKAGE_DIR}/__init__.py
