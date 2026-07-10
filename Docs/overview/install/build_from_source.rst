:orphan:

.. _build-from-source:

####################
Building from source
####################

This page describes how to install AIMET from source in a uv environment and within docker container.

You can also use a virtual environment (venv), provided your system has the required Python version and necessary dependencies that aren't available via pip, such as CUDA and cuDNN.

UV environment
=================

Install uv
----------

Following https://docs.astral.sh/uv/getting-started/installation/ to isntall UV

On Linux/MacOS, you can run the following command to install UV:

.. code-block:: bash

    curl -LsSf https://astral.sh/uv/install.sh | sh

Create a new uv environment with Python 3.10+
-----------------------------------------------

AIMET requires Python 3.10 or newer. An example of environment setup with Python 3.10 is shown
below; substitute any supported Python version (3.10 through 3.13) via ``--python=<version>``:

.. code-block:: bash

    # Create new uv environment with Python 3.10 (or newer, e.g. 3.11/3.12/3.13)
    uv venv --python=3.10 aimet-dev

    # Activate the environment
    . aimet-dev/bin/activate

NVIDIA CUDA support
-------------------

Skip the following step, if you don't want to compile with CUDA support or already have CUDA installed.

Here, we show how to install CUDA Toolkit 12.1 on Ubuntu 22.04 (also applicable to Ubuntu 22.04 and above; replace ``ubuntu2204`` in the repo URL with your Ubuntu release, e.g. ``ubuntu2404``).
You can find instructions for other versions and platforms in NVIDIA's documentation: https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html

.. code-block:: bash
    # Download and install the CUDA repository keyring
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
    sudo dpkg -i cuda-keyring_1.0-1_all.deb

    # Update package lists
    sudo apt update

    # Install CUDA Toolkit 12.1
    sudo apt install -y cuda-toolkit-12-1 libcudnn9-cuda-12 libcudnn9-dev-cuda-12

    # Add CUDA to PATH (add these lines to ~/.bashrc)
    export PATH=/usr/local/cuda-12.1/bin:$PATH
    export LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64:$LD_LIBRARY_PATH
    # Add the above lines into your ~/.bashrc or ~/.zshrc file to make the changes permanent

    # Verify installation
    nvcc --version

Set environment variables to build desired AIMET wheel
------------------------------------------------------

General Toggles

- GPU build: -DENABLE_CUDA=ON
- CPU-only build: -DENABLE_CUDA=OFF
- Build C++ tests: -DENABLE_TESTS=ON
- Skip building C++ tests: -DENABLE_TESTS=OFF

Variant-specific Toggles

.. list-table::
   :widths: 8 40
   :header-rows: 1

   * - Variant
     - CMake flags
   * - aimet-onnx
     - -DENABLE_ONNX=ON -DENABLE_TORCH=OFF
   * - aimet-torch
     - -DENABLE_TORCH=ON -DENABLE_ONNX=OFF
   * - Docs
     - -DENABLE_ONNX=ON -DENABLE_TORCH=ON -DENABLE_CUDA=OFF

.. code-block:: bash

    # Example: Build for aimet-onnx with GPU
    export 'CMAKE_ARGS=-DENABLE_CUDA=ON -DENABLE_ONNX=ON -DENABLE_TORCH=OFF -DENABLE_TESTS=OFF'
    export 'SKBUILD_BUILD_TARGETS=all'

Compile and install pip package dependencies
--------------------------------------------

.. code-block:: bash

    # cd to AIMET root directory
    cd aimet/

    # Compile requirements from pyproject.toml with constraints
    uv pip compile pyproject.toml --extra=dev --extra=test --output-file=/tmp/requirements.txt

    # Install the compiled dependencies
    uv pip install -r /tmp/requirements.txt

Build AIMET wheel and run unit tests
------------------------------------

.. code-block:: bash

    # Build AIMET wheel
    python3 -m build --wheel --no-isolation .

    # Install the built wheel
    pip install dist/aimet*.whl

    # Run unit tests (ONNX)
    cd TrainingExtensions/onnx/test/python
    pytest

Build AIMET documentation
-------------------------

.. code-block:: bash

    # cd to AIMET root directory
    cd aimet/

    # Example: Build for Documentation Only
    export 'CMAKE_ARGS=-DENABLE_ONNX=ON -DENABLE_TORCH=ON -DENABLE_CUDA=OFF -DENABLE_TESTS=OFF'
    export 'SKBUILD_BUILD_TARGETS=all;doc'

    # Pin torch, onnxruntime versions
    echo "onnxruntime==1.22.0" >> /tmp/constraints.txt
    echo "torch==2.1.2" >> /tmp/constraints.txt

    # Compile requirements from pyproject.toml with constraints
    uv pip compile pyproject.toml -v --constraint=/tmp/constraints.txt --extra=dev --extra=test,docs --output-file=/tmp/requirements.txt

    # Install the compiled dependencies
    python3 -m pip install -r /tmp/requirements.txt

    # Build AIMET docs (aimet/build/Docs/index.html)
    python3 -m build --wheel --no-isolation .

Docker environment
==================

Build and run docker container locally
--------------------------------------

Docker build argument examples for AIMET Variants.

.. list-table::
   :widths: 8 40
   :header-rows: 1

   * - Variant
     - Build args
   * - aimet-onnx
     - VER_PYTHON=3.10 VER_ONNXRUNTIME=1.22.0 VER_CUDA=12.1.0
   * - aimet-torch
     - VER_PYTHON=3.10 VER_TORCH=2.1.2 VER_CUDA=12.1.1

.. code-block:: bash

    # cd to AIMET root directory
    cd aimet

    # Example: Build docker image for aimet-onnx with GPU
    docker buildx build --build-arg VER_PYTHON=3.10 --build-arg VER_ONNXRUNTIME=1.22.0 --build-arg VER_CUDA=12.1.0 -t onnx-gpu:1.0 -f Jenkins/fast-release/Dockerfile.ci .

    # Run the container
    docker run -it -v /local/mnt/workspace:/local/mnt/workspace/ --gpus all --user root onnx-gpu:1.0

    # Set up the conda environment inside the container
    . ${VIRTUAL_ENV}/bin/activate

Set environment variables to build desired AIMET wheel
------------------------------------------------------

General Toggles

- GPU build: -DENABLE_CUDA=ON
- CPU-only build: -DENABLE_CUDA=OFF
- Build C++ tests: -DENABLE_TESTS=ON
- Skip build C++ tests: -DENABLE_TESTS=OFF

Variant-specific Toggles

.. list-table::
   :widths: 8 40
   :header-rows: 1

   * - Variant
     - CMake flags
   * - aimet-onnx
     - -DENABLE_ONNX=ON -DENABLE_TORCH=OFF
   * - aimet-torch
     - -DENABLE_TORCH=ON -DENABLE_ONNX=OFF

.. code-block:: bash

    # Example: Build for aimet-onnx with GPU
    export 'CMAKE_ARGS=-DENABLE_CUDA=ON -DENABLE_ONNX=ON -DENABLE_TORCH=OFF -DENABLE_TESTS=OFF'
    export 'SKBUILD_BUILD_TARGETS=all'

Build AIMET wheel and run unit tests
------------------------------------

.. code-block:: bash

    # Build AIMET wheel
    python3 -m build --wheel --no-isolation .

    # Install the built wheel
    pip install dist/aimet*.whl

    # Run unit tests (ONNX)
    cd TrainingExtensions/onnx/test/python/
    pytest
