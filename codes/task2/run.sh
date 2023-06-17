# eval "$(conda shell.bash hook)"
# conda activate deep-learning

_CONDA_DEFAULT_ENV="${CONDA_DEFAULT_ENV:-base}"

__conda_setup="$('/opt/pytorch/miniconda/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    # this the path I've observed during docker RUN
    eval "$__conda_setup"
else
    # Not sure what triggers this branch
    if [ -f "/opt/pytorch/miniconda/etc/profile.d/conda.sh" ]; then
        . "/opt/pytorch/miniconda/etc/profile.d/conda.sh"
    else
        export PATH="/opt/pytorch/miniconda/bin:$PATH"
    fi
fi
unset __conda_setup
# Restore our "indended" default env
conda activate "${_CONDA_DEFAULT_ENV}"
# This just logs the output to stderr for debugging. 
>&2 echo "ENTRYPOINT: CONDA_DEFAULT_ENV=${CONDA_DEFAULT_ENV}"

exec "$@"

# tail -f /dev/null