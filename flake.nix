{
  inputs = {
    # nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
    systems.url = "github:nix-systems/default";
  };

  outputs = { systems, nixpkgs, ... } @ inputs: let
    eachSystem = f: nixpkgs.lib.genAttrs (import systems) (system: f nixpkgs.legacyPackages.${system});
  in {
    devShells = eachSystem (pkgs: {
      default = pkgs.mkShell {
        buildInputs = with pkgs; [
          python310Packages.python
          python3Packages.pynput
        ];
        
        shellHook = ''
          export LD_LIBRARY_PATH=${pkgs.lib.makeLibraryPath [
            pkgs.fontconfig
            pkgs.freetype
            pkgs.xorg.libX11
            pkgs.xorg.libxcb
            pkgs.xorg.xcbutilwm
            pkgs.xorg.xcbutilimage
            pkgs.xorg.xcbutil
            pkgs.xorg.xcbutilkeysyms
            pkgs.xorg.xcbutilrenderutil
            pkgs.libxkbcommon
            pkgs.stdenv.cc.cc
            pkgs.dbus
            pkgs.libGL
            pkgs.glib
            pkgs.zlib
            pkgs.python3
            pkgs.python3Packages.pip
            pkgs.xorg.xrandr
            pkgs.xorg.libX11
            pkgs.xorg.libXrandr
          ]}

          export QT_PLUGIN_PATH=venv/lib/python3.10/site-packages/PyQt5/Qt5/plugins

          # Create a virtual environment if not already created
          if [ ! -d ".venv" ]; then
            python3 -m venv .venv
          fi

          # Activate the virtual environment
          source .venv/bin/activate

          # Install non-Nix packages using pip
          pip install --upgrade pip
          pip install PyQt5
          pip install opencv-python
          pip install ultralytics

          pip3 freeze > requirements.txt
        '';
      };
    });
  };
}

