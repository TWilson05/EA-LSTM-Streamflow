#!/bin/sh
set -e

PYTHON_VERSION="3.10.11"
ENV_NAME="ealstm-postprocessing-py-3-10-11"
KERNEL_NAME="ealstm-postprocessing-py-3-10-11"
KERNEL_DISPLAY_NAME="EA-LSTM Postprocessing (Py 3.10.11)"

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
else
  echo "ERROR: pyenv-virtualenv is required."
  exit 1
fi

echo "Checking Python installation..."

# Install base Python if missing
if ! pyenv versions --bare | grep -q "^${PYTHON_VERSION}\$"; then
  echo "Installing Python ${PYTHON_VERSION}..."
  pyenv install "${PYTHON_VERSION}"
else
  echo "Python ${PYTHON_VERSION} already installed."
fi

echo "Checking virtualenv..."

# Create virtualenv if missing
if ! pyenv versions --bare | grep -q "^${ENV_NAME}\$"; then
  echo "Creating virtualenv ${ENV_NAME}..."
  pyenv virtualenv "${PYTHON_VERSION}" "${ENV_NAME}"
else
  echo "Virtualenv ${ENV_NAME} already exists."
fi

echo "Setting local pyenv environment..."

# Create/update local .python-version
pyenv local "${ENV_NAME}"

echo "Activating environment..."

# Activate shell environment
pyenv shell "${ENV_NAME}"

echo "Python executable:"
which python

python --version

echo "Upgrading pip..."
pip install --upgrade pip

# Install requirements
if [ -f "postprocessing_requirements.txt" ]; then
  echo "Installing dependencies from postprocessing_requirements.txt..."
  pip install -r postprocessing_requirements.txt
else
  echo "No postprocessing_requirements.txt found."
fi

# Ensure Jupyter kernel support
echo "Installing ipykernel..."
pip install ipykernel

# Register kernel if missing
if ! jupyter kernelspec list | grep -q "${KERNEL_NAME}"; then
  echo "Registering Jupyter kernel..."

  python -m ipykernel install \
    --user \
    --name "${KERNEL_NAME}" \
    --display-name "${KERNEL_DISPLAY_NAME}"
else
  echo "Jupyter kernel already exists."
fi

echo ""
echo "Setup complete!"
echo ""
echo "Environment name:"
echo "  ${ENV_NAME}"
echo ""
echo "Jupyter kernel:"
echo "  ${KERNEL_DISPLAY_NAME}"
echo ""
echo "To activate manually later:"
echo "  pyenv activate ${ENV_NAME}"