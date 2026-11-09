# HARPOR in one image: the seven envs (setup/install.sh), Blender, Node.js and the Codex CLI for the agent, and the
# docker CLI that starts the agent's and Blender's sibling containers (controller.container). The weights, caches and
# jobs are not inside: they live in the data folder, mounted at /data (containers/harpor-docker).
#
# Build it from the repo root on a Linux x86_64 machine. No GPU is needed; it takes about 2 h, 60 GB and plenty of RAM:
#
#   docker build -f containers/Dockerfile -t harpor:0.1 .
#
# TORCH_CUDA_ARCH_LIST sets the GPUs that the CUDA extensions are built for: Ampere (8.0, 8.6), Ada (8.9) and Hopper
# (9.0). The pinned torch versions build for nothing newer. HARPOR_IMAGE is the tag the image gives its sibling
# containers; build with --build-arg HARPOR_IMAGE=... when you tag it otherwise (harpor-docker also passes it).
FROM ubuntu:22.04

ARG TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0"
ARG MAX_JOBS=16
ARG HARPOR_IMAGE=harpor:0.1

# The X and GL libraries are those of containers/harpor-agent.def and of VoMP's install_env.sh (Blender and bpy).
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      ca-certificates curl git bzip2 xz-utils procps \
      libx11-6 libxi6 libxxf86vm1 libxfixes3 libxrender1 libgl1 libegl1 libxkbcommon0 \
      libsm6 libice6 libwayland-client0 libdecor-0-0 libvulkan1 libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# The envs, from the same scripts as a machine without docker. Only setup/ is copied first, so a change to the code
# does not build the envs again. The pip and micromamba caches go, to keep the image small.
ENV HARPOR_ROOT=/opt/harpor
COPY setup/ /opt/harpor/code/setup/
RUN HARPOR_SKIP_AGENT=1 HARPOR_SKIP_WEIGHTS=1 TORCH_CUDA_ARCH_LIST="$TORCH_CUDA_ARCH_LIST" MAX_JOBS="$MAX_JOBS" \
      bash /opt/harpor/code/setup/install.sh \
    && rm -rf /opt/harpor/.cache /opt/harpor/micromamba/pkgs

# The agent's tools, at the versions of containers/harpor-agent.def, and the docker CLI.
RUN curl -fsSL https://nodejs.org/dist/v24.21.0/node-v24.21.0-linux-x64.tar.xz | tar -xJ -C /opt \
    && ln -s /opt/node-v24.21.0-linux-x64 /opt/node \
    && PATH=/opt/node/bin:$PATH npm install -g --prefix /opt/codex @openai/codex@0.162.1 \
    && curl -fsSL https://download.blender.org/release/Blender5.2/blender-5.2.2-linux-x64.tar.xz | tar -xJ -C /opt \
    && ln -s /opt/blender-5.2.2-linux-x64 /opt/blender \
    && curl -fsSL https://download.docker.com/linux/static/stable/x86_64/docker-27.5.1.tgz \
      | tar -xz -C /usr/local/bin --strip-components=1 docker/docker

# What the stages write goes to the data folder through these links: caches (Hugging Face, torch, the nvdiffrast
# build), the weights that setup/weights.sh puts inside the clones, and the jobs. /opt/gator/bin is where the agent's
# containers find the helper client, which Apptainer binds there.
RUN ln -s /data/cache /opt/harpor/.cache \
    && rm -rf /opt/harpor/sam-3d-objects/checkpoints && ln -s /data/weights/sam-3d-objects /opt/harpor/sam-3d-objects/checkpoints \
    && rm -rf /opt/harpor/VoMP/weights && ln -s /data/weights/vomp /opt/harpor/VoMP/weights \
    && ln -s /data/runs /opt/harpor/webapp_runs \
    && mkdir -p /opt/gator && ln -s /opt/harpor/code/harpor/agent/bin /opt/gator/bin

COPY . /opt/harpor/code/

ENV HARPOR_DATA=/data WEBAPP_RUNS=/data/runs TMPDIR=/data/tmp HOME=/data/home \
    HARPOR_SANDBOX=docker:${HARPOR_IMAGE} \
    NVIDIA_VISIBLE_DEVICES=all NVIDIA_DRIVER_CAPABILITIES=all \
    PATH=/opt/codex/bin:/opt/node/bin:/opt/blender:$PATH
WORKDIR /opt/harpor/code
ENTRYPOINT ["/opt/harpor/code/containers/entrypoint.sh"]
CMD ["serve"]
