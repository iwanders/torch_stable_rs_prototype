#!/bin/bash -xe

DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )

valgrind -s --suppressions="${DIR}"/combined.supp --suppressions="${DIR}"/rust_generic.supp $*
