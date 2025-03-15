{
  description = "A very basic flake";

  outputs =
    { self, nixpkgs }:
    {
      devShell.x86_64-linux =
        let
          pkgs = nixpkgs.legacyPackages.x86_64-linux;
        in
        pkgs.mkShell {
          buildInputs = with pkgs; [
            (pkgs.python3.withPackages (python-pkgs: [
              python-pkgs.torch
              python-pkgs.peft
              python-pkgs.pandas
              python-pkgs.datasets
              python-pkgs.scikit-learn
            ]))
          ];
          # shellHook = ''
          #   echo hi
          #   LD_LIBRARY_PATH=${pkgs.gcc.cc.lib}/lib/
          # '';
        };
    };
}
