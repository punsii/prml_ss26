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
    }:
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
        ]
      );

      reflectionRemovalSrc = pkgs.fetchFromGitHub {
        owner = "Devashi-Choudhary";
        repo = "Reflection-Removal-Techniques-Review";
        rev = "f5e1c040c33c62e8311c3efa5e318be0d1162f74";
        hash = "sha256-/EZDDZjHCRNiIdSRC5D8YDTWsVe92HHnUDIR9AqpLFE=";
      };

      test-averaging = pkgs.runCommand "test-averaging" { buildInputs = [ pythonEnv ]; } ''
        cp -r ${reflectionRemovalSrc}/Averaging/* .
        chmod -R u+w .
        # Patch out cv2.imshow/waitKey (no display in sandbox)
        sed -i 's/cv2.imshow.*/#/' Averaging.py
        sed -i 's/cv2.waitKey.*/#/' Averaging.py
        python Averaging.py -i 5_images_lowers
        test -f Average.png
        mkdir -p $out
        cp Average.png $out/
      '';

      test-ica = pkgs.runCommand "test-ica" { buildInputs = [ pythonEnv ]; } ''
        cp -r ${reflectionRemovalSrc}/ICA/* .
        chmod -R u+w .
        python ICA.py -i1 1.png -i2 2.png
        test -f try-A1.png
        test -f try-B2.png
        mkdir -p $out
        cp try-A1.png try-B2.png $out/
      '';

      showAveraging = pkgs.writeShellScriptBin "show-averaging" ''
        echo "Averaging: before (5 input images) → after"
        echo "Inputs: ${reflectionRemovalSrc}/Averaging/5_images_lowers/"
        echo "Output: ${test-averaging}/Average.png"
        ${pkgs.feh}/bin/feh \
          --montage \
          --thumb-width 400 \
          --thumb-height 300 \
          --limit-width 2000 \
          ${reflectionRemovalSrc}/Averaging/5_images_lowers/* \
          ${test-averaging}/Average.png
      '';

      showIca = pkgs.writeShellScriptBin "show-ica" ''
        echo "ICA: 2 input images → 2 separated layers"
        echo "Inputs: ${reflectionRemovalSrc}/ICA/1.png, ${reflectionRemovalSrc}/ICA/2.png"
        echo "Outputs: ${test-ica}/try-A1.png, ${test-ica}/try-B2.png"
        ${pkgs.feh}/bin/feh \
          --montage \
          --thumb-width 400 \
          --thumb-height 400 \
          --limit-width 1600 \
          ${reflectionRemovalSrc}/ICA/1.png \
          ${reflectionRemovalSrc}/ICA/2.png \
          ${test-ica}/try-A1.png \
          ${test-ica}/try-B2.png
      '';
    in
    {
      checks.${system} = {
        devShell = self.devShells.${system}.default;
        inherit test-averaging test-ica;
        show-averaging = showAveraging;
        show-ica = showIca;
      };

      apps.${system} = {
        showAveraging = {
          type = "app";
          program = "${showAveraging}/bin/show-averaging";
        };
        showIca = {
          type = "app";
          program = "${showIca}/bin/show-ica";
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
    };
}
