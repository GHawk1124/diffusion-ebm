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
        # `nix run .#tour` — open the marimo project tour in a browser.
        # Resolves the project root via git so the command works from any
        # subdirectory of the checkout.
        tourScript = pkgs.writeShellScript "diffusion-ebm-tour" ''
          set -euo pipefail
          ROOT="$(${pkgs.git}/bin/git rev-parse --show-toplevel 2>/dev/null || pwd)"
          cd "$ROOT"
          exec ${fhs}/bin/diffusion-ebm -c \
            "uv run marimo edit notebooks/08_tour_marimo.py"
        '';

        # `nix run .#build-apptainer-image` — produce a ready-to-run .sif from
        # container/diffusion-ebm.def.  Uses --fakeroot so the build doesn't
        # need root, but does require subuid/subgid mapping for $USER (NixOS
        # users.users.<name>.subUidRanges / .subGidRanges in configuration.nix
        # if not already configured).  ~15–20 min on first build because
        # flash-attn is compiled inside the image for sm_80/8.9/9.0.
        buildApptainerScript = pkgs.writeShellApplication {
          name = "diffusion-ebm-build-apptainer";
          runtimeInputs = [ pkgs.apptainer pkgs.git pkgs.coreutils ];
          text = ''
            set -euo pipefail

            ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
            cd "$ROOT"

            OUTPUT="''${1:-diffusion-ebm.sif}"
            DEF="$ROOT/container/diffusion-ebm.def"

            if [ ! -f "$DEF" ]; then
              echo "error: missing $DEF" >&2
              exit 1
            fi

            # NixOS-specific apptainer build flags:
            #  --fakeroot                 enable appears-as-root mode (umbrella).
            #  --ignore-subuid            don't go through newuidmap/newgidmap
            #                             (NixOS lacks the setuid helpers); fall
            #                             back to root-mapped userns.
            #  --ignore-fakeroot-command  don't inject the host's `fakeroot`
            #                             binary into the container — on NixOS
            #                             it's nix-store-linked and can't load
            #                             its libs once the Ubuntu rootfs is in
            #                             place.  We don't actually need it:
            #                             inside the userns we appear as UID 0
            #                             so apt-get / pip / etc. work without
            #                             the LD_PRELOAD shim.
            APPTAINER_BUILD_ARGS=(--fakeroot --ignore-subuid --ignore-fakeroot-command --force)

            echo ":: writing $OUTPUT (expect ~15-20 min for the first build)"
            apptainer build "''${APPTAINER_BUILD_ARGS[@]}" "$OUTPUT" "$DEF"

            echo
            echo ":: image built: $OUTPUT"
            SIZE=$(du -h "$OUTPUT" | cut -f1)
            echo ":: size: $SIZE"
            echo
            echo ":: smoke-test locally (needs an NVIDIA GPU on the host):"
            echo "     nix run .#apptainer -- run --nv $OUTPUT python notebooks/00_smoke.py"
            echo ":: transfer to cluster:"
            echo "     scp $OUTPUT <cluster>:~/"
          '';
        };
      in {
        devShells.default = fhs.env;

        # Convenience aliases.
        devShells.fhs = fhs.env;

        apps.tour = {
          type = "app";
          program = toString tourScript;
        };

        apps.build-apptainer-image = {
          type = "app";
          program = "${buildApptainerScript}/bin/diffusion-ebm-build-apptainer";
        };

        # `nix run .#apptainer -- <args>` — direct apptainer pass-through so
        # the user can `apptainer run --nv ./diffusion-ebm.sif ...`,
        # `apptainer shell --nv ...`, `apptainer inspect ...` etc. without
        # putting apptainer into the FHS dev shell (where it would conflict
        # with the FHS chroot machinery).
        apps.apptainer = {
          type = "app";
          program = "${pkgs.apptainer}/bin/apptainer";
        };

        # `nix build .#apptainer-def` — exposes the def file as a flake
        # output for reproducibility / CI consumption.
        packages.apptainer-def = pkgs.runCommand "diffusion-ebm.def" { } ''
          cp ${./container/diffusion-ebm.def} $out
        '';
      });
}
