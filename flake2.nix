{
  description = "Flake for Python torch with CUDA using cachix";

  # Configure Cachix to use the CUDA binary cache.
#   nixConfig = {
#     extra-substituters = [ "https://cuda-maintainers.cachix.org" ];
#     extra-trusted-public-keys = [
#       "cuda-maintainers.cachix.org-1:0dq3bujKpuEPMCX6U4WylrUDZ9JyUG0VpVZa7CNfq5E="
#     ];
#   };

  inputs = {
    # Use an appropriate nixpkgs channel; often nixos-unstable is a good choice.
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs = { self, nixpkgs }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs {
        inherit system;
        config = {
          allowUnfree = true;
        };
      };
    in {
      devShells.${system}.default = pkgs.mkShell {
        buildInputs = [
          # Create a Python with packages environment that includes torch-bin.
          (pkgs.python3.withPackages (ps: with ps; [ 
            pytorch-bin
            pip
            peft
            pandas
            datasets
            scikit-learn
            timm
            einops
            flask
            pillow
            transformers
            torchvision
        ]))
        ];

        # shellHook = ''
        #   echo "Using CUDA from ${pkgs.cudatoolkit}"
        #   export CUDA_PATH=${pkgs.cudatoolkit}
        #   export CUDA_HOME=${pkgs.cudatoolkit}
        # '';
      };
    };
}
