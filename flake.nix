{
  description = "diffusion-ebm: THRML joint sampler for masked diffusion LMs";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachSystem [ "x86_64-linux" "aarch64-linux" ] (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config.allowUnfree = true;  # NVIDIA libraries
        };

        # PyPI CUDA wheels (torch +cuXX, jax[cuda12]) expect a Linux FHS
        # layout (libs in /usr/lib) and a discoverable libcuda.so.  buildFHSEnv
        # gives us the layout; LD_LIBRARY_PATH points at the kernel-matched
        # NVIDIA user-space libs that NixOS exposes at /run/opengl-driver/lib.
        fhs = pkgs.buildFHSEnv {
          name = "diffusion-ebm";
          targetPkgs = p: with p; [
            python312
            uv
            git

            # C/C++ runtime libs PyPI wheels link against.
            stdenv.cc.cc
            stdenv.cc.cc.lib
            zlib
            glib
            libGL

            # Build toolchain for flash-attn (nvcc + ninja).
            cudaPackages_12.cudatoolkit
            ninja
          ];
          profile = ''
            export LD_LIBRARY_PATH="/run/opengl-driver/lib:$LD_LIBRARY_PATH"

            # The unwrapped binutils `ld` that PyTorch's cpp_extension invokes
            # at link time has no default search path for glibc's start-files
            # (crti.o, crtn.o) under buildFHSEnv. Point LIBRARY_PATH at glibc
            # and the gcc runtime so flash-attn's final .so link succeeds.
            export LIBRARY_PATH="${pkgs.glibc}/lib:${pkgs.stdenv.cc.cc.lib}/lib:$LIBRARY_PATH"

            # Triton's NVIDIA backend shells out to /sbin/ldconfig to discover
            # libcuda.so. NixOS doesn't ship ldconfig in /sbin, so the call
            # fails with FileNotFoundError. The env override below is checked
            # first by triton.runtime.driver and skips the ldconfig probe.
            export TRITON_LIBCUDA_PATH="/run/opengl-driver/lib"

            # nvcc lives under cudatoolkit; flash-attn's setup.py reads CUDA_HOME.
            export CUDA_HOME="${pkgs.cudaPackages_12.cudatoolkit}"
            export CUDA_PATH="$CUDA_HOME"
            export PATH="$CUDA_HOME/bin:$PATH"

            # flash-attn defaults to building for *every* recent compute capability.
            # Restrict to the local GPU (RTX 3000 Ada = sm_89) to cut build time
            # from ~40min to ~10min.  Override on other hardware:
            #   export TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.9;9.0"
            export TORCH_CUDA_ARCH_LIST="8.9"

            # Limit parallel compile jobs — each flash-attn translation unit can
            # consume ~10GB RAM.  Bump if you have >32GB.
            export MAX_JOBS="2"

            export UV_PROJECT_ENVIRONMENT=".venv"
          '';
          runScript = "bash";
        };
      in {
        devShells.default = fhs.env;

        # Convenience aliases.
        devShells.fhs = fhs.env;
      });
}
