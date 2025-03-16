{
  description = "A very basic flake";

  nixConfig = {
    extra-substituters = [ "https://nix-community.cachix.org" "https://cuda-maintainers.cachix.org"];
    extra-trusted-public-keys = [
      "nix-community.cachix.org-1:mB9FSh9qf2dCimDSUo8Zy7bkq5CX+/rkCWyvRCYg3Fs="
      "cuda-maintainers.cachix.org-1:0dq3bujKpuEPMCX6U4WylrUDZ9JyUG0VpVZa7CNfq5E="
    ];
  };

  outputs =
    { self, nixpkgs }:
    {
      devShell.x86_64-linux =
        let
          # pkgs = nixpkgs.legacyPackages.x86_64-linux;
          pkgs = import nixpkgs { system = "x86_64-linux"; config.allowUnfree = true;};

        in
        pkgs.mkShell {
          buildInputs = with pkgs; [
            cudatoolkit
            (pkgs.python3.withPackages (python-pkgs: with python-pkgs; [
              # pip
              torch-bin
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
              # flash_attn
            ]))
          ];
          # shellHook = ''
          #   echo hi
          #   LD_LIBRARY_PATH=${pkgs.gcc.cc.lib}/lib/
          # '';
          # shellHook = ''
          #   echo "You are now using a NIX environment"
          #   export CUDA_PATH=${pkgs.cudatoolkit}
          #   echo $CUDA_PATH
          # '';
          # shellHook = ''
          #   export LD_LIBRARY_PATH="${pkgs.vulkan-loader}/lib"
          # '';
          shellHook = ''
    echo "You are now using a NIX environment"
    export CUDA_PATH=${pkgs.cudatoolkit}
    export CUDA_HOME=${pkgs.cudatoolkit}
    echo $CUDA_PATH
  '';
        };
    };
}
