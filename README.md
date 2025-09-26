# kero

A command-line instrument for quantitative analysis of audio convolution and frequency domain image filtering.

```
Usage: kero [OPTIONS] COMMAND [ARGS]...

  kero - A command-line instrument for quantitative analysis of audio
  convolution and frequency domain image filtering.

Options:
  --help  Show this message and exit.

Commands:
  all     Process both audio and image analysis.
  audio   Process audio convolution in time and frequency domains.
  image   Process image filtering in frequency domain.
```

## Architectural Rationale

The system's architecture is predicated on a deterministic and hermetic build environment, leveraging `uv`. The entire toolchain is encapsulated within a multi-stage Docker container, ensuring a consistent and isolated context for both compilation and execution, thereby eliminating environmental variance in analytical workflows.

## Prerequisites

- A functional Docker daemon.

## Build Procedure

The build process is entirely self-contained. From the root of the source tree, the build sequence is initiated by the following command:

```bash
docker build -t kero .
```

This directive orchestrates a sequenced, multi-stage build, resulting in a minimal runtime image containing the `kero` executable.

## Operational Execution

Execution of the `kero` executable is performed via the `docker run` command. A volume must be mapped from the host system to the container's `/app/output` directory to facilitate file I/O for output artifacts. For validation and baseline testing, the container includes sample audio and image files (`AudioBanheiro.wav`, `AudioBeethoven.wav`, `image1.jpg`, `image4.png`).

### Audio Analysis

To perform audio convolution analysis, execute the following:

```bash
docker run --rm \
  -v "$(pwd)/output:/app/output" \
  kero audio \
  --input data/AudioBeethoven.wav \
  --impulse data/AudioBanheiro.wav \
  --output-dir output
```

The `-v "$(pwd)/output:/app/output"` directive establishes a bind mount between the host's `output` directory and the container's `/app/output` mount point. Input files are addressed via their path in the container's virtual filesystem (e.g., `data/AudioBeethoven.wav`). The resulting convoluted audio files, analysis plots, and a detailed report will be deposited in the `output` directory on the host filesystem.

### Image Analysis

To perform frequency domain image filtering, execute the following:

```bash
docker run --rm \
  -v "$(pwd)/output:/app/output" \
  kero image \
  --image1 data/image1.jpg \
  --image2 data/image4.png \
  --output-dir output \
  --cutoff 80.0
```

The `--cutoff` parameter controls the cutoff frequency for the Gaussian filter, accepting a floating-point value; if this argument is omitted, the system defaults to a value of 80.0. The resulting processed images, frequency spectra, and a detailed analysis report will be deposited in the `output` directory on the host filesystem.

### Combined Analysis

To execute both audio and image analyses simultaneously, use the `all` command:

```bash
docker run --rm \
  -v "$(pwd)/output:/app/output" \
  kero all \
  --audio-input data/AudioBeethoven.wav \
  --audio-impulse data/AudioBanheiro.wav \
  --image1 data/image1.jpg \
  --image2 data/image4.png \
  --output-dir output \
  --cutoff 80.0
```

This command will run both the audio and image processing pipelines, generating all associated output artifacts within the `output` directory on the host system.
