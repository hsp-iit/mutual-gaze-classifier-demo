#!/bin/bash
set -e

echo "[ -r /usr/share/bash-completion/bash_completion   ] && . /usr/share/bash-completion/bash_completion" >> /root/.bashrc


if [ -z "$(which setup_robotology_tdd.sh)" ] ; then
    echo "File setup_robotology_tdd.sh not found."
    exit 1
fi

source setup_robotology_tdd.sh

# If a CMD is passed, execute it
exec "$@"

