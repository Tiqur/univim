{
  lib,
  pkgs,
  home-manager,
  univimModule,
  univimPackage,
}:
let
  base = {
    home.username = "tester";
    home.homeDirectory = "/home/tester";
    home.stateVersion = "25.11";
  };

  evaluateFor =
    stateVersion: modules:
    (home-manager.lib.homeManagerConfiguration {
      inherit pkgs;
      modules = [
        (
          base
          // {
            home = base.home // {
              stateVersion = stateVersion;
            };
          }
        )
        univimModule
      ]
      ++ modules;
    }).config;
  evaluate = evaluateFor "25.11";

  hyprlandBase = {
    enable = true;
    package = null;
    portalPackage = null;
    systemd.enable = true;
    extraConfig = "# test\n";
  };
  hyprland = extra: {
    wayland.windowManager.hyprland = hyprlandBase // { configType = "hyprlang"; } // extra;
  };
  hyprlandDefault = extra: {
    wayland.windowManager.hyprland = hyprlandBase // extra;
  };

  check =
    name: condition: if condition then null else throw "UniVim Home Manager test failed: ${name}";

  disabled = evaluate [ ];
  native = evaluate [
    (hyprland { })
    { services.univim.enable = true; }
  ];
  graphical = evaluate [
    { services.univim.enable = true; }
  ];
  uwsmlike = evaluate [
    (hyprland { systemd.enable = false; })
    { services.univim.enable = true; }
  ];
  customTarget = evaluate [
    {
      services.univim = {
        enable = true;
        sessionTarget = "univim-test.target";
      };
    }
  ];
  hyprlangBindings = evaluate [
    (hyprland { settings.bind = [ "SUPER, Q, exec, kitty" ]; })
    {
      services.univim = {
        enable = true;
        keybindings.enable = true;
      };
    }
  ];
  gestureBindings = evaluate [
    (hyprland { })
    {
      services.univim = {
        enable = true;
        keybindings.modifierGestures = true;
      };
    }
  ];
  luaBindings = evaluate [
    (hyprland { configType = "lua"; })
    {
      services.univim = {
        enable = true;
        keybindings = {
          enable = true;
          modifierGestures = true;
        };
      };
    }
  ];
  luaDefaultBindings = evaluateFor "26.05" [
    (hyprlandDefault { })
    {
      services.univim = {
        enable = true;
        keybindings.enable = true;
      };
    }
  ];
  allFlags = evaluate [
    {
      services.univim = {
        enable = true;
        output = "DP-3";
        turbo = true;
        interactiveOnly = true;
        noClick = true;
        flat = true;
        timeout = 42;
      };
    }
  ];

  nativeService = native.systemd.user.services.univim;
  graphicalService = graphical.systemd.user.services.univim;
  uwsmlikeService = uwsmlike.systemd.user.services.univim;
  customService = customTarget.systemd.user.services.univim;
  hyprlangValues = hyprlangBindings.wayland.windowManager.hyprland.settings;
  hyprlangText = hyprlangBindings.xdg.configFile."hypr/hyprland.conf".text;
  gestureValues = gestureBindings.wayland.windowManager.hyprland.settings;
  luaText = luaBindings.xdg.configFile."hypr/hyprland.lua".text;
  luaDefaultText = luaDefaultBindings.xdg.configFile."hypr/hyprland.lua".text;
  launcher = lib.head allFlags.systemd.user.services.univim.Service.ExecStart;

  checks = [
    (check "disabled by default" (
      !disabled.services.univim.enable && !(disabled.systemd.user.services ? univim)
    ))
    (check "disabled adds no package" (!(lib.elem univimPackage disabled.home.packages)))
    (check "installs the selected package" (lib.elem univimPackage native.home.packages))
    (check "native Home Manager target" (
      lib.elem "hyprland-session.target" nativeService.Unit.After
      && lib.elem "hyprland-session.target" nativeService.Unit.PartOf
      && lib.elem "hyprland-session.target" nativeService.Install.WantedBy
    ))
    (check "external graphical session target" (
      lib.elem "graphical-session.target" graphicalService.Unit.After
      && lib.elem "graphical-session.target" graphicalService.Install.WantedBy
    ))
    (check "disabled native integration falls back to graphical target" (
      lib.elem "graphical-session.target" uwsmlikeService.Unit.After
      && lib.elem "graphical-session.target" uwsmlikeService.Install.WantedBy
    ))
    (check "custom session target" (
      lib.elem "univim-test.target" customService.Unit.After
      && lib.elem "univim-test.target" customService.Install.WantedBy
    ))
    (check "service does not restart into a stale socket" (
      nativeService.Service.Restart == "no" && nativeService.Service.UMask == "0077"
    ))
    (check "existing bindings are preserved" (lib.elem "SUPER, Q, exec, kitty" hyprlangValues.bind))
    (check "Hyprlang hint and grid bindings" (
      lib.any (value: lib.hasInfix "univim-desktop hints" value) hyprlangValues.bind
      && lib.any (value: lib.hasInfix "univim-desktop grid" value) hyprlangValues.bind
      && lib.any (value: lib.hasInfix "SUPER, F" value) hyprlangValues.bind
      && lib.any (value: lib.hasInfix "SUPER, G" value) hyprlangValues.bind
    ))
    (check "Hyprlang modifier gestures are nonconsuming global binds" (
      lib.all (value: lib.hasInfix "global, univim:" value) gestureValues.bindn
      && builtins.length gestureValues.bindn == 4
      && !(gestureValues ? bind)
    ))
    (check "generated Hyprlang file contains merged bindings" (
      lib.hasInfix "SUPER, Q, exec, kitty" hyprlangText
      && lib.hasInfix "univim-desktop hints" hyprlangText
      && lib.hasInfix "univim-desktop grid" hyprlangText
    ))
    (check "Lua bindings use the Lua key syntax" (
      lib.hasInfix "SUPER + F" luaText
      && lib.hasInfix "SUPER + G" luaText
      && lib.hasInfix "univim-desktop hints" luaText
      && lib.hasInfix "univim-desktop grid" luaText
    ))
    (check "Lua gestures set non_consuming" (
      lib.hasInfix "[\"non_consuming\"] = true" luaText
      && lib.hasInfix "univim:shift-left" luaText
      && lib.hasInfix "univim:control-right" luaText
    ))
    (check "Lua is detected from the Home Manager default" (
      lib.hasInfix "SUPER + F" luaDefaultText && lib.hasInfix "univim-desktop hints" luaDefaultText
    ))
  ];

  forcedChecks = builtins.deepSeq checks (lib.all (result: result == null) checks);
  launcherCheck = pkgs.runCommand "univim-home-manager-launcher" { } ''
    ${pkgs.gnugrep}/bin/grep -F -- 'serve' ${launcher}
    ${pkgs.gnugrep}/bin/grep -F -- '--output' ${launcher}
    ${pkgs.gnugrep}/bin/grep -F -- 'DP-3' ${launcher}
    ${pkgs.gnugrep}/bin/grep -F -- '--turbo' ${launcher}
    ${pkgs.gnugrep}/bin/grep -F -- '--interactive-only' ${launcher}
    ${pkgs.gnugrep}/bin/grep -F -- '--no-click' ${launcher}
    ${pkgs.gnugrep}/bin/grep -F -- '--flat' ${launcher}
    ${pkgs.gnugrep}/bin/grep -F -- '--timeout 42' ${launcher}
    ${pkgs.coreutils}/bin/touch $out
  '';
in
if forcedChecks then launcherCheck else throw "UniVim Home Manager tests failed"
