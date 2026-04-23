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
        ps:
        with ps;
        [
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
    in
    {
      checks.${system} = {
        devShell = self.devShells.${system}.default;
      };

      devShells.${system}.default = pkgs.mkShell {
        packages = [
          treefmtEval.config.build.wrapper
          pythonEnv
          pkgs.nil
          pkgs.pyright
        ];
      };

      formatter.${system} = treefmtEval.config.build.wrapper;
    };
}
