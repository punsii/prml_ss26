{
  description = "reflection-removal";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs?ref=nixos-unstable";
    treefmt-nix = {
      inputs.nixpkgs.follows = "nixpkgs";
      url = "github:numtide/treefmt-nix";
    };
  };

  outputs =
    {
      self,
      nixpkgs,
      treefmt-nix,
      ...
    }@inputs:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs {
        inherit system;
        config = {
          allowUnfree = true;
        };
      };
      python3 = pkgs.python3;
      treefmtEval = treefmt-nix.lib.evalModule pkgs {
        projectRootFile = "flake.nix";
        programs = {
          black.enable = true;
          isort.enable = true;
          prettier.enable = true;
          nixfmt.enable = true;
        };
      };
      pythonEnv = python3.withPackages (
        ps: with ps; [
          # core
          numpy
          pillow
          tqdm

          # opencv
          opencv4

          # deep learning
          torch
          torchvision

          # utilities
          matplotlib
          streamlit
        ]
      );

      reflectionRemovalSrc = pkgs.fetchFromGitHub {
        owner = "Devashi-Choudhary";
        repo = "Reflection-Removal-Techniques-Review";
        rev = "f5e1c040c33c62e8311c3efa5e318be0d1162f74";
        hash = "sha256-/EZDDZjHCRNiIdSRC5D8YDTWsVe92HHnUDIR9AqpLFE=";
      };

      test-specular-diffuse = pkgs.runCommand "test-specular-diffuse" { buildInputs = [ pythonEnv ]; } ''
        cp ${./src/specular_diffuse.py} specular_diffuse.py
        # Use one of the convex optimization sample images as test input
        cp ${reflectionRemovalSrc}/Convex_Optimization/input/toy_example.jpg test_input.jpg
        python specular_diffuse.py test_input.jpg -o output --iterations 10
        test -f output/test_input_diffuse.png
        test -f output/test_input_specular.png
        mkdir -p $out
        cp output/*.png $out/
      '';

      runStreamlit = pkgs.writeShellScriptBin "run-streamlit" ''
        cd "$(${pkgs.git}/bin/git rev-parse --show-toplevel)"
        ${pythonEnv}/bin/streamlit run src/app.py
      '';

      runStreamlitService = pkgs.writeShellScriptBin "run-streamlit-service" ''
        exec ${pythonEnv}/bin/streamlit run ${./src}/app.py "$@"
      '';

      runClaheBmw = pkgs.writeShellScriptBin "run-clahe-bmw" ''
        INPUT="data/BMW_25/Rohdaten/Erste Bearbeitungsstufe 10-17ym/_DSC1090.JPG"
        if [ ! -f "$INPUT" ]; then
          echo "ERROR: Input not found: $INPUT"
          echo "Run this app from the repo root directory."
          exit 1
        fi
        OUTDIR=$(mktemp -d)
        ${pythonEnv}/bin/python ${./src/clahe.py} "$INPUT" -o "$OUTDIR"
        echo "Showing: original | CLAHE"
        ${pkgs.feh}/bin/feh \
          --montage \
          --thumb-width 600 \
          --thumb-height 600 \
          --limit-width 1400 \
          "$INPUT" \
          "$OUTDIR/_DSC1090_clahe.png"
      '';

      runRetinexBmw = pkgs.writeShellScriptBin "run-retinex-bmw" ''
        INPUT="data/BMW_25/Rohdaten/Erste Bearbeitungsstufe 10-17ym/_DSC1090.JPG"
        if [ ! -f "$INPUT" ]; then
          echo "ERROR: Input not found: $INPUT"
          echo "Run this app from the repo root directory."
          exit 1
        fi
        OUTDIR=$(mktemp -d)
        ${pythonEnv}/bin/python ${./src/retinex.py} "$INPUT" -o "$OUTDIR"
        echo "Showing: original | Retinex"
        ${pkgs.feh}/bin/feh \
          --montage \
          --thumb-width 600 \
          --thumb-height 600 \
          --limit-width 1400 \
          "$INPUT" \
          "$OUTDIR/_DSC1090_retinex.png"
      '';

      runHomomorphicBmw = pkgs.writeShellScriptBin "run-homomorphic-bmw" ''
        INPUT="data/BMW_25/Rohdaten/Erste Bearbeitungsstufe 10-17ym/_DSC1090.JPG"
        if [ ! -f "$INPUT" ]; then
          echo "ERROR: Input not found: $INPUT"
          echo "Run this app from the repo root directory."
          exit 1
        fi
        OUTDIR=$(mktemp -d)
        ${pythonEnv}/bin/python ${./src/homomorphic.py} "$INPUT" -o "$OUTDIR"
        echo "Showing: original | Homomorphic"
        ${pkgs.feh}/bin/feh \
          --montage \
          --thumb-width 600 \
          --thumb-height 600 \
          --limit-width 1400 \
          "$INPUT" \
          "$OUTDIR/_DSC1090_homomorphic.png"
      '';

      runSpecularBmw = pkgs.writeShellScriptBin "run-specular-bmw" ''
        INPUT="data/BMW_25/Rohdaten/Erste Bearbeitungsstufe 10-17ym/_DSC1090.JPG"
        if [ ! -f "$INPUT" ]; then
          echo "ERROR: Input not found: $INPUT"
          echo "Run this app from the repo root directory."
          exit 1
        fi
        OUTDIR=$(mktemp -d)
        ${pythonEnv}/bin/python ${./src/specular_diffuse.py} "$INPUT" -o "$OUTDIR"
        echo "Showing: original | diffuse | specular"
        ${pkgs.feh}/bin/feh \
          --montage \
          --thumb-width 500 \
          --thumb-height 500 \
          --limit-width 1600 \
          "$INPUT" \
          "$OUTDIR/_DSC1090_diffuse.png" \
          "$OUTDIR/_DSC1090_specular.png"
      '';

    in
    {
      checks.${system} = {
        devShell = self.devShells.${system}.default;
        inherit test-specular-diffuse;

        # NixOS module evaluation verified via `nix eval .#nixosModules.reflectionRemoval`
      };

      apps.${system} = {
        runStreamlit = {
          type = "app";
          program = "${runStreamlit}/bin/run-streamlit";
        };
        runStreamlitService = {
          type = "app";
          program = "${runStreamlitService}/bin/run-streamlit-service";
        };
        runClaheBmw = {
          type = "app";
          program = "${runClaheBmw}/bin/run-clahe-bmw";
        };
        runRetinexBmw = {
          type = "app";
          program = "${runRetinexBmw}/bin/run-retinex-bmw";
        };
        runHomomorphicBmw = {
          type = "app";
          program = "${runHomomorphicBmw}/bin/run-homomorphic-bmw";
        };
        runSpecularBmw = {
          type = "app";
          program = "${runSpecularBmw}/bin/run-specular-bmw";
        };
      };

      devShells.${system}.default = pkgs.mkShell {
        packages = [
          treefmtEval.config.build.wrapper
          pythonEnv
          pkgs.nil
          pkgs.pyright
        ];

        REFLECTION_REMOVAL_SRC = reflectionRemovalSrc;
      };

      formatter.${system} = treefmtEval.config.build.wrapper;

      nixosModules = {
        reflectionRemoval = ./nix/nixosModules/reflectionRemoval.nix;
      };
    };
}
