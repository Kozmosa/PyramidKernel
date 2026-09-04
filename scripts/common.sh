#!/usr/bin/env bash
set -euo pipefail

# Shared helpers for stable long-running experiment launches.

pk_repo_root() {
  local script_path=$1
  local script_dir
  script_dir=$(cd $(dirname $script_path) && pwd)
  cd $script_dir/.. && pwd
}

pk_abs_path() {
  local repo_root=$1
  local raw_path=$2
  if [[ $raw_path = /* ]]; then
    printf '%s\n' $raw_path
  else
    printf '%s\n' $repo_root/$raw_path
  fi
}

pk_prepare_output_dir() {
  local output_dir=$1
  local allow_reuse=${2:-0}
  if [[ -d $output_dir ]]; then
    if [[ $allow_reuse != 1 ]] && find $output_dir -mindepth 1 -print -quit | grep -q .; then
      echo [PyramidKernel] Output directory is not empty: $output_dir >&2
      echo [PyramidKernel] Use a fresh output directory, or pass --reuse-output-dir if you really want to reuse it. >&2
      return 1
    fi
  fi
  mkdir -p $output_dir
}

pk_write_launch_script() {
  local launch_path=$1
  local repo_root=$2
  local command_array_name=$3
  local -n command_ref=$command_array_name
  local forwarded_vars=(
    PYRAMIDKERNEL_CONDA_PREFIX
    PYRAMIDKERNEL_CONDA_ENV
    PYRAMIDKERNEL_CONDA_SH
    PYRAMIDKERNEL_ENV_FILE
    CUDA_VISIBLE_DEVICES
    CUDALLM_DEVICE
    CUDALLM_MODEL_PATH
    OPENAI_API_KEY
    OPENAI_BASE_URL
    OPENAI_MODEL
    CLAUDE_API_KEY
    CLAUDE_BASE_URL
    CLAUDE_MODEL
    ANTHROPIC_API_KEY
    ANTHROPIC_BASE_URL
    ANTHROPIC_MODEL
    GEMINI_API_KEY
    GEMINI_BASE_URL
    GEMINI_MODEL
  )
  {
    cat <<SCRIPT_HEAD
#!/usr/bin/env bash
set -euo pipefail

cd $(printf '%q' $repo_root)
SCRIPT_HEAD

    local python_bin_dir=""
    if [[ ${#command_ref[@]} -gt 0 && ${command_ref[0]} == */* ]]; then
      python_bin_dir=$(cd $(dirname ${command_ref[0]}) && pwd)
    fi

    local var_name
    for var_name in ${forwarded_vars[@]}; do
      if [[ -n ${!var_name:-} ]]; then
        printf 'export %s=%q\n' $var_name ${!var_name}
      fi
    done

    if [[ -n $python_bin_dir ]]; then
      printf 'export PATH=%q:$PATH\n' $python_bin_dir
    fi

    cat <<'''SCRIPT_BODY'''

if [[ -n ${PYRAMIDKERNEL_CONDA_PREFIX:-} || -n ${PYRAMIDKERNEL_CONDA_ENV:-} ]]; then
  CONDA_SH=${PYRAMIDKERNEL_CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}
  if [[ ! -f $CONDA_SH ]]; then
    echo [PyramidKernel] Could not find conda.sh at: $CONDA_SH >&2
    exit 1
  fi
  # shellcheck disable=SC1090
  source $CONDA_SH
  if [[ -n ${PYRAMIDKERNEL_CONDA_PREFIX:-} ]]; then
    conda activate ${PYRAMIDKERNEL_CONDA_PREFIX}
  else
    conda activate ${PYRAMIDKERNEL_CONDA_ENV}
  fi
fi

export PYTHONUNBUFFERED=${PYTHONUNBUFFERED:-1}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

resolve_cuda_home() {
  if [[ -n ${CUDA_HOME:-} && -x $CUDA_HOME/bin/nvcc ]]; then
    return 0
  fi
  if [[ -n ${CUDA_HOME:-} && ! -x $CUDA_HOME/bin/nvcc ]]; then
    echo [PyramidKernel] Ignoring CUDA_HOME without nvcc: $CUDA_HOME >&2
    unset CUDA_HOME
  fi
  if [[ -n ${CONDA_PREFIX:-} && -x $CONDA_PREFIX/bin/nvcc ]]; then
    export CUDA_HOME=$CONDA_PREFIX
  elif command -v nvcc >/dev/null 2>&1; then
    NVCC_PATH=$(command -v nvcc)
    export CUDA_HOME=$(cd $(dirname $NVCC_PATH)/.. && pwd)
  elif [[ -x /usr/local/cuda/bin/nvcc ]]; then
    export CUDA_HOME=/usr/local/cuda
  else
    for candidate in /usr/local/cuda-*/bin/nvcc; do
      if [[ -x $candidate ]]; then
        export CUDA_HOME=$(cd $(dirname $candidate)/.. && pwd)
        break
      fi
    done
  fi
}
resolve_cuda_home
if [[ -z ${CUDA_HOME:-} || ! -x $CUDA_HOME/bin/nvcc ]]; then
  echo [PyramidKernel] Could not locate a CUDA toolkit with nvcc. Set CUDA_HOME to a valid CUDA toolkit root. >&2
  exit 1
fi

if [[ -n ${CUDA_HOME:-} ]]; then
  if [[ -n ${CONDA_PREFIX:-} ]]; then
    export PATH=$CONDA_PREFIX/bin:$CUDA_HOME/bin:$PATH
  else
    export PATH=$CUDA_HOME/bin:$PATH
  fi
  if [[ -n ${LD_LIBRARY_PATH:-} ]]; then
    export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$CUDA_HOME/lib:$LD_LIBRARY_PATH
  else
    export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$CUDA_HOME/lib
  fi
fi

COMMAND=(
SCRIPT_BODY

    local item
    for item in ${command_ref[@]}; do
      printf '  %q\n' $item
    done

    cat <<'''SCRIPT_TAIL'''
)

echo [PyramidKernel] repo: __REPO_ROOT__
printf '[PyramidKernel] command:'
printf ' %q' ${COMMAND[@]}
printf '\n'
echo [PyramidKernel] started_at: $(date '+%Y-%m-%d %H:%M:%S')

echo [PyramidKernel] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}
echo [PyramidKernel] CUDALLM_DEVICE=${CUDALLM_DEVICE:-<unset>}
echo [PyramidKernel] PYRAMIDKERNEL_CONDA_ENV=${PYRAMIDKERNEL_CONDA_ENV:-<unset>}
echo [PyramidKernel] PYRAMIDKERNEL_ENV_FILE=${PYRAMIDKERNEL_ENV_FILE:-<unset>}
echo [PyramidKernel] CUDA_HOME=${CUDA_HOME:-<unset>}
echo [PyramidKernel] NVCC_PATH=$(command -v nvcc || printf '%s' '<missing>')

exec ${COMMAND[@]}
SCRIPT_TAIL
  } | sed "s|__REPO_ROOT__|$repo_root|g" > $launch_path
  chmod +x $launch_path
}

pk_unique_tmux_session() {
  local base_name=$1
  local session_name=$base_name
  if ! command -v tmux >/dev/null 2>&1; then
    echo [PyramidKernel] tmux is required for --detach mode. >&2
    return 1
  fi
  if tmux has-session -t $session_name 2>/dev/null; then
    session_name=${base_name}-$(date +%H%M%S)
  fi
  printf '%s\n' $session_name
}
