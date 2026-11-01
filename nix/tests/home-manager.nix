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
        colorByType = true;
        timeout = 42;
      };
    }
  ];
  globalOptions = evaluate [
    (hyprland { })
    {
      services.univim = {
        enable = true;
        hyprland = {
          enable = true;
          hintsBind = "SUPER, semicolon";
          gridBind = "SUPER, G";
          doubleTapCapsLock = true;
          evdev = false;
        };
        settings = {
          minScore = 0.52;
          fastHalfRes = true;
          doubleTapMs = 240;
        };
      };
    }
  ];
  singleCaps = evaluate [
    (hyprland { })
    {
      services.univim = {
        enable = true;
        hyprland.enable = true;
        hyprland.singlePressCapsLock = true;
        hyprland.altActivation = "off";
        settings.pointerMotion = "ease_out";
        settings.pointerSpeed = 1800;
      };
    }
  ];
  singleCapsLua = evaluateFor "26.05" [
    (hyprland { configType = "lua"; })
    {
      services.univim = {
        enable = true;
        hyprland.enable = true;
        hyprland.singlePressCapsLock = true;
      };
    }
  ];
  singleAlt = evaluate [
    (hyprland { })
    {
      services.univim = {
        enable = true;
        hyprland.enable = true;
        hyprland.altActivation = "single";
      };
    }
  ];
  programsAlias = evaluate [
    {
      programs.univim.enable = true;
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
  globalValues = globalOptions.wayland.windowManager.hyprland.settings;
  globalLauncher = lib.head globalOptions.systemd.user.services.univim.Service.ExecStart;
  globalSettings = globalOptions.xdg.configFile."univim/settings.json".text;
  singleCapsValues = singleCaps.wayland.windowManager.hyprland.settings;
  singleCapsSettings = singleCaps.xdg.configFile."univim/settings.json".text;
  singleCapsLuaText = singleCapsLua.xdg.configFile."hypr/hyprland.lua".text;
  singleAltValues = singleAlt.wayland.windowManager.hyprland.settings;

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
    (check "global Hyprland hints and grid bindings are opt-in" (
      lib.elem "SUPER, semicolon, global, univim:hints" globalValues.bind
      && lib.elem "SUPER, G, global, univim:grid" globalValues.bind
    ))
    (check "double-tap CapsLock binding is non-consuming" (
      lib.elem ", CAPS, global, univim:caps-lock" globalValues.bindn
    ))
    (check "left Alt double-tap defaults to a non-consuming global binding" (
      lib.elem "ALT, Alt_L, global, univim:alt-left" globalValues.bindn
      && !lib.elem "ALT, Alt_R, global, univim:alt-left" globalValues.bindn
    ))
    (check "single left Alt uses the hints shortcut without intercepting Alt chords" (
      lib.elem "ALT, Alt_L, global, univim:hints" singleAltValues.bindn
      && !lib.elem "ALT, Alt_L, global, univim:alt-left" singleAltValues.bindn
    ))
    (check "settings configure the daemon and evdev" (
      lib.hasInfix "\"double_tap_ms\":240" globalSettings
      && lib.hasInfix "\"use_evdev\":false" globalSettings
      && lib.hasInfix "\"min_score\":0.52" globalSettings
    ))
    (check "opt-in single CapsLock generates consuming bindings and disables evdev gestures" (
      lib.any (value: lib.hasInfix ", code:66, exec," value && lib.hasInfix "univim-desktop hints" value) singleCapsValues.bind
      && lib.any (value: lib.hasInfix "SHIFT, code:66, exec," value && lib.hasInfix "univim-desktop grid" value) singleCapsValues.bind
      && lib.any (value: lib.hasInfix "ALT, code:66, exec," value && lib.hasInfix "univim-desktop debug-open" value) singleCapsValues.bind
      && singleCapsValues.input.kb_options == "caps:none"
      && !(singleCapsValues ? bindn)
      && lib.hasInfix "\"use_evdev\":false" singleCapsSettings
      && lib.hasInfix "\"pointer_motion\":\"ease_out\"" singleCapsSettings
      && lib.hasInfix "\"pointer_speed\":1800" singleCapsSettings
    ))
    (check "Hyprlang exec passes executable and arguments separately" (
      lib.elem ", code:66, exec, ${univimPackage}/bin/univim-desktop hints" singleCapsValues.bind
      && lib.elem "SHIFT, code:66, exec, ${univimPackage}/bin/univim-desktop grid" singleCapsValues.bind
      && lib.elem "ALT, code:66, exec, ${univimPackage}/bin/univim-desktop debug-open" singleCapsValues.bind
    ))
    (check "Lua single CapsLock bindings consume keys" (
      lib.hasInfix "code:66" singleCapsLuaText
      && lib.hasInfix "SHIFT + code:66" singleCapsLuaText
      && lib.hasInfix "ALT + code:66" singleCapsLuaText
      && lib.hasInfix "caps:none" singleCapsLuaText
      && lib.hasInfix "univim-desktop debug-open" singleCapsLuaText
    ))
    (check "programs.univim remains a working alias" programsAlias.services.univim.enable)
  ];

  forcedChecks = builtins.deepSeq checks (lib.all (result: result == null) checks);
  launcherCheck = pkgs.runCommand "univim-home-manager-launcher" { } ''
    ${pkgs.gnugrep}/bin/grep -F -- 'daemon' ${launcher}
    ${pkgs.gnugrep}/bin/grep -F -- '--min-score 0.45' ${launcher}
    ${pkgs.gnugrep}/bin/grep -F -- '--output' ${launcher}
    ${pkgs.gnugrep}/bin/grep -F -- 'DP-3' ${launcher}
    ${pkgs.gnugrep}/bin/grep -F -- '--fast-half-res' ${launcher}
    ${pkgs.gnugrep}/bin/grep -F -- '--no-text' ${launcher}
    ${pkgs.gnugrep}/bin/grep -F -- '--no-click' ${launcher}
    ${pkgs.gnugrep}/bin/grep -F -- '--flat' ${launcher}
    ${pkgs.gnugrep}/bin/grep -F -- '--timeout 42' ${launcher}
    ${pkgs.gnugrep}/bin/grep -F -- '--min-score 0.52' ${globalLauncher}
    ${pkgs.gnugrep}/bin/grep -F -- '--fast-half-res' ${globalLauncher}
    ${pkgs.coreutils}/bin/touch $out
  '';
in
if forcedChecks then launcherCheck else throw "UniVim Home Manager tests failed"
