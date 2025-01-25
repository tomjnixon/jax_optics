{
  inputs.nixpkgs.url = "nixpkgs/nixos-unstable";
  inputs.flake-utils.url = "github:numtide/flake-utils";

  outputs =
    {
      self,
      nixpkgs,
      flake-utils,
    }:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
        python = pkgs.python3;
      in
      rec {
        packages.jax_optics = python.pkgs.buildPythonPackage rec {
          pname = "jax_optics";
          version = "0.0.1";
          pyproject = true;
          src = ./.;

          dependencies = [
            python.pkgs.jax
            python.pkgs.jaxlib
          ];

          build-system = [
            python.pkgs.setuptools
          ];

          nativeCheckInputs = [
            python.pkgs.pytestCheckHook
          ];
        };

        packages.default = packages.jax_optics;

        devShells.default = packages.jax_optics.overridePythonAttrs (old: {
          nativeBuildInputs = [
            python.pkgs.jaxopt
            python.pkgs.flake8
            python.pkgs.black
            python.pkgs.isort
            (pkgs.jupyter.override { python3 = python; })
            pkgs.nixfmt-rfc-style
          ];
        });
      }
    );
}
