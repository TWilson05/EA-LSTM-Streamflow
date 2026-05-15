#!/bin/sh
set -e

PYTHON_VERSION="3.10.11"
ENV_NAME="ealstm-postprocessing-py-3-10-11"
KERNEL_NAME="ealstm-postprocessing-py-3-10-11"

echo "Initializing pyenv..."

# Ensure pyenv exists
if ! command -v pyenv >/dev/null 2>&1; then
  echo "ERROR: pyenv is not installed or not in PATH."
  exit 1
fi

# POSIX-compatible pyenv init
eval "$(pyenv init -)"

# Initialize pyenv-virtualenv if installed
if pyenv commands | grep -q '^virtualenv-init$'; then
  eval "$(pyenv virtualenv-init -)"
fi

echo ""
echo "Installed Jupyter kernels:"
jupyter kernelspec list || true

echo ""
echo "Removing Jupyter kernel..."

if jupyter kernelspec list 2>/dev/null | grep -q "${KERNEL_NAME}"; then
  jupyter kernelspec uninstall -f "${KERNEL_NAME}" || true
  echo "Removed kernel ${KERNEL_NAME}"
else
  echo "Kernel ${KERNEL_NAME} not found."
fi

echo ""
echo "Removing pyenv virtualenv..."

if pyenv versions --bare | grep -q "^${ENV_NAME}\$"; then
  pyenv uninstall -f "${ENV_NAME}"
  echo "Removed virtualenv ${ENV_NAME}"
else
  echo "Virtualenv ${ENV_NAME} not found."
fi

echo ""
echo "Removing local .python-version..."

if [ -f ".python-version" ]; then
  rm -f .python-version
  echo "Removed .python-version"
else
  echo ".python-version not found."
fi

echo ""
echo "Installed pyenv versions now:"
pyenv versions

echo ""
echo "Base Python interpreter ${PYTHON_VERSION} was NOT removed."
echo ""
echo "To also remove it manually:"
echo "  pyenv uninstall ${PYTHON_VERSION}"
echo ""
echo "Cleanup complete."
