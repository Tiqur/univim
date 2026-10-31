{
  description = "UniVim: native Rust UI detector and Hyprland desktop navigation";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs = { self, nixpkgs }:
    let
      # Wayland dependencies are unconditional in Cargo.toml, even for the CLI.
      systems = [ "x86_64-linux" "aarch64-linux" ];
      forAllSystems = f: nixpkgs.lib.genAttrs systems (system: f nixpkgs.legacyPackages.${system});
      source = nixpkgs.lib.cleanSourceWith {
        src = ./.;
        filter = path: type:
          let name = builtins.baseNameOf path;
          in !(builtins.elem name [ "target" "result" ".git" "__pycache__" ])
            && nixpkgs.lib.cleanSourceFilter path type;
      };
    in {
      devShells = forAllSystems (pkgs: {
        default = pkgs.mkShell {
          src = source;
          packages = [
            (pkgs.python3.withPackages (ps: [ ps.opencv4 ps.numpy ]))
            pkgs.rustc
            pkgs.cargo
            pkgs.rustfmt
            pkgs.clippy
            pkgs.clang
            pkgs.llvmPackages.libclang
            pkgs.pkg-config
            pkgs.opencv4
            pkgs.wayland
            pkgs.libxkbcommon
            pkgs.wayland-utils
            pkgs.cmake
            pkgs.gnumake
            pkgs.hyperfine
            pkgs.jq
          ];

          LIBCLANG_PATH = "${pkgs.llvmPackages.libclang.lib}/lib";
          shellHook = ''
            export PKG_CONFIG_PATH="${pkgs.opencv4.cxxdev}/lib/pkgconfig''${PKG_CONFIG_PATH:+:$PKG_CONFIG_PATH}"
          '';
        };
      });

      packages = forAllSystems (pkgs: {
        default = pkgs.rustPlatform.buildRustPackage {
          pname = "univim";
          version = "0.1.0";
          src = source;
          cargoLock.lockFile = source + "/Cargo.lock";

          nativeBuildInputs = [ pkgs.pkg-config pkgs.clang ];
          buildInputs = [ pkgs.opencv4 pkgs.wayland pkgs.libxkbcommon ];

          LIBCLANG_PATH = "${pkgs.llvmPackages.libclang.lib}/lib";
          PKG_CONFIG_PATH = "${pkgs.opencv4.cxxdev}/lib/pkgconfig";

          meta = {
            mainProgram = "univim";
            platforms = systems;
          };
        };
      });
    };
}
