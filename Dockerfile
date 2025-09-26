#
# Stage 1: Builder
#
# This stage prepares the application environment. It installs all necessary
# build dependencies, including the 'uv' package manager, and then installs
# the Python packages into a virtual environment.
#
FROM python:3.13-slim AS builder

# ARG TARGETARCH is automatically set by Docker to the architecture of the
# build machine (e.g., 'amd64', 'arm64').
ARG TARGETARCH

# Set a non-interactive frontend for Debian's package manager to prevent
# prompts from blocking the build process.
ENV DEBIAN_FRONTEND=noninteractive

# Install build dependencies.
# - build-essential: Provides compilers (gcc, etc.) needed for Python packages
#   that contain C extensions.
# - libsndfile1-dev: Development libraries for reading/writing audio files,
#   required by the 'soundfile' dependency.
# - curl: Utility for downloading dependencies.
RUN apt-get update && apt-get install -y \
    build-essential \
    libsndfile1-dev \
    curl

# Download and install uv, a high-performance Python package manager.
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# Set the working directory for the application source code.
WORKDIR /app

# Copy the project's dependency definition file into the builder stage. This
# leverages Docker's layer caching, so dependencies are only re-installed if
# this file changes.
COPY pyproject.toml .

# Create a virtual environment and install dependencies using uv. This is the
# equivalent of a compilation step, preparing a self-contained environment.
RUN uv venv && uv sync

# Copy the entire build context (source code, data files, etc.) into the
# builder stage.
COPY . .

# Install the project itself into the virtual environment in editable mode.
# This creates the 'kero' command-line entrypoint. The --no-deps flag is used
# because 'uv sync' already installed all dependencies.
RUN uv pip install . --no-deps

#
# Stage 2: Final Image
#
# This stage creates the final, minimal production image. It copies the
# virtual environment with all its dependencies from the builder stage and
# includes only the necessary runtime system libraries, resulting in a smaller
# and more secure image.
#
FROM python:3.13-slim

# Set a non-interactive frontend for Debian's package manager.
ENV DEBIAN_FRONTEND=noninteractive

# Install runtime dependencies and clean up the apt cache to minimize image size.
# - libsndfile1: Runtime library for audio file I/O.
# - libgl1-mesa-glx & libglib2.0-0: Runtime libraries required for headless
#   operation of image processing packages like OpenCV and Matplotlib.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory for the application.
WORKDIR /app

# Copy the virtual environment (containing all Python packages and the 'kero'
# executable) from the builder stage into the final image.
COPY --from=builder /app/.venv ./.venv

# Copy the application source code and data required at runtime.
COPY --from=builder /app/kero.py .
COPY --from=builder /app/data ./data

# Add the virtual environment's bin directory to the system's PATH. This allows
# the 'kero' command to be executed directly.
ENV PATH="/app/.venv/bin:$PATH"

# Set the container's entrypoint to run the application executable.
ENTRYPOINT ["kero"]

# Set the default command to display the application's help message. This is
# executed when the container is run without any additional arguments.
CMD ["--help"]
