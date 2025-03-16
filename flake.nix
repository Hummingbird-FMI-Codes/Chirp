{
  description = "A very basic flake";

    inputs = {
    # Use an appropriate nixpkgs channel; often nixos-unstable is a good choice.
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
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
            # cudatoolkit
            (pkgs.python3.withPackages (ps: with ps; [
              torch
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
              pydantic
            ]))
          ];
          # shellHook = ''
          #   echo hi
          #   LD_LIBRARY_PATH=${pkgs.gcc.cc.lib}/lib/
          # '';
        };
    };
}
