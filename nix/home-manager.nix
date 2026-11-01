{ self }:
{
  config,
  lib,
  pkgs,
  ...
}:
let
  cfg = config.services.univim;
  bindings = cfg.keybindings;
  settings = cfg.settings;
  defaultPackage = self.packages.${pkgs.stdenv.hostPlatform.system}.default;
  desktop = "${cfg.package}/bin/univim-desktop";
  hyprlandEnabled = lib.attrByPath [ "wayland" "windowManager" "hyprland" "enable" ] false config;
  hyprlandSystemdEnabled = lib.attrByPath [
    "wayland"
    "windowManager"
    "hyprland"
    "systemd"
    "enable"
  ] false config;
  # Home Manager's Hyprland module imports the session environment before
  # starting this target. A graphical-session.target managed by another
  # session manager remains the portable fallback.
  defaultSessionTarget =
    if hyprlandEnabled && hyprlandSystemdEnabled then
      "hyprland-session.target"
    else
      "graphical-session.target";
  sessionTarget = if cfg.sessionTarget == null then defaultSessionTarget else cfg.sessionTarget;
  configType = lib.attrByPath [ "wayland" "windowManager" "hyprland" "configType" ] "hyprlang" config;
  luaConfiguration = configType == "lua";

  serviceArgs = [
    "${cfg.package}/bin/univim-desktop"
    "daemon"
    "--min-score"
    (toString settings.minScore)
  ]
  ++ lib.optionals (cfg.output != null) [
    "--output"
    cfg.output
  ]
  ++ lib.optional (settings.fastHalfRes || cfg.turbo) "--fast-half-res"
  ++ lib.optional (cfg.interactiveOnly || !settings.includeText) "--no-text"
  ++ lib.optional (!settings.collapseParagraphs) "--no-collapse-paragraphs"
  ++ lib.optional cfg.noClick "--no-click"
  ++ lib.optional cfg.flat "--flat"
  ++ [
    "--timeout"
    (toString cfg.timeout)
  ];
  serviceLauncher = pkgs.writeShellScript "univim-desktop-serve" ''
    exec ${lib.escapeShellArgs serviceArgs}
  '';

  runtimeSettingsJson = builtins.toJSON {
    detector = {
      min_score = settings.minScore;
      include_text = settings.includeText;
      fast_half_res = settings.fastHalfRes;
      collapse_paragraphs = settings.collapseParagraphs;
    };
    double_tap_ms = settings.doubleTapMs;
    cache_ttl_ms = settings.cacheTtlMs;
    spotlight_dim = settings.spotlightDim;
    use_evdev = cfg.hyprland.evdev && !cfg.hyprland.singlePressCapsLock;
    show_status_bar = true;
    pointer_motion = settings.pointerMotion;
    pointer_speed = settings.pointerSpeed;
  };

  globalBind = key: event: "${bindingKey key}, global, ${event}";
  luaGlobalBind = key: event: {
    _args = [
      (bindingKey key)
      (lib.generators.mkLuaInline "hl.dsp.global(${builtins.toJSON event})")
    ];
  };
  globalBindings =
    lib.optionals cfg.hyprland.enable (
      (lib.optionals (cfg.hyprland.hintsBind != null) [
        (if luaConfiguration then
          luaGlobalBind cfg.hyprland.hintsBind "univim:hints"
        else
          globalBind cfg.hyprland.hintsBind "univim:hints")
      ])
      ++ (lib.optionals (cfg.hyprland.gridBind != null) [
        (if luaConfiguration then
          luaGlobalBind cfg.hyprland.gridBind "univim:grid"
        else
          globalBind cfg.hyprland.gridBind "univim:grid")
      ])
    );
  doubleTapBindings =
    lib.optionals (cfg.hyprland.enable && cfg.hyprland.doubleTapCapsLock) (
      (lib.optional cfg.hyprland.doubleTapCapsLock (
        if luaConfiguration then
          luaGestureBind ", Caps_Lock" "hl.dsp.global(\"univim:caps-lock\")"
        else
          ", CAPS, global, univim:caps-lock"
      ))
    );
  singleCapsBindings = lib.optionals (cfg.hyprland.enable && cfg.hyprland.singlePressCapsLock) (
    if luaConfiguration then
      [
        (luaBind ", code:66" "hl.dsp.exec_cmd(${builtins.toJSON "${desktop} hints"})")
        (luaBind "SHIFT, code:66" "hl.dsp.exec_cmd(${builtins.toJSON "${desktop} grid"})")
        (luaBind "ALT, code:66" "hl.dsp.exec_cmd(${builtins.toJSON "${desktop} debug-open"})")
      ]
    else
      [
        (hyprlangBind ", code:66" "${desktop} hints")
        (hyprlangBind "SHIFT, code:66" "${desktop} grid")
        (hyprlangBind "ALT, code:66" "${desktop} debug-open")
      ]
  );

  bindingParts = value: map lib.trim (lib.splitString "," value);
  validBinding =
    value:
    let
      parts = bindingParts value;
    in
    builtins.length parts == 2 && builtins.elemAt parts 1 != "";
  bindingKey =
    value:
    let
      parts = bindingParts value;
      modifiers = builtins.elemAt parts 0;
      key = builtins.elemAt parts 1;
    in
    if !luaConfiguration then
      "${modifiers}, ${key}"
    else if modifiers == "" then
      key
    else
      "${modifiers} + ${key}";

  hyprlangBind = key: command: "${bindingKey key}, exec, ${lib.escapeShellArg command}";
  hyprlangGlobalBind = key: event: "${bindingKey key}, global, ${event}";
  luaBind = key: dispatcher: {
    _args = [
      (bindingKey key)
      (lib.generators.mkLuaInline dispatcher)
    ];
  };
  luaGestureBind = key: dispatcher: {
    _args = [
      (bindingKey key)
      (lib.generators.mkLuaInline dispatcher)
      { non_consuming = true; }
    ];
  };

  explicitBindings = lib.optionals bindings.enable (
    if luaConfiguration then
      [
        (luaBind bindings.hints "hl.dsp.exec_cmd(${builtins.toJSON "${desktop} hints"})")
        (luaBind bindings.grid "hl.dsp.exec_cmd(${builtins.toJSON "${desktop} grid"})")
      ]
    else
      [
        (hyprlangBind bindings.hints "${desktop} hints")
        (hyprlangBind bindings.grid "${desktop} grid")
      ]
  );
  gestureBindings = lib.optionals bindings.modifierGestures (
    if luaConfiguration then
      [
        (luaGestureBind "SHIFT, Shift_L" "hl.dsp.global(\"univim:shift-left\")")
        (luaGestureBind "SHIFT, Shift_R" "hl.dsp.global(\"univim:shift-right\")")
        (luaGestureBind "CTRL, Control_L" "hl.dsp.global(\"univim:control-left\")")
        (luaGestureBind "CTRL, Control_R" "hl.dsp.global(\"univim:control-right\")")
      ]
    else
      [
        (hyprlangGlobalBind "SHIFT, Shift_L" "univim:shift-left")
        (hyprlangGlobalBind "SHIFT, Shift_R" "univim:shift-right")
        (hyprlangGlobalBind "CTRL, Control_L" "univim:control-left")
        (hyprlangGlobalBind "CTRL, Control_R" "univim:control-right")
      ]
  );
in
{
  imports = [
    # Keep the newer `programs.univim` spelling working while preserving the
    # documented `services.univim` option tree.
    (lib.mkAliasOptionModule [ "programs" "univim" ] [ "services" "univim" ])
  ];

  options.services.univim = {
    enable = lib.mkEnableOption "UniVim desktop navigation";

    package = lib.mkOption {
      type = lib.types.package;
      default = defaultPackage;
      defaultText = lib.literalExpression "inputs.univim.packages.<system>.default";
      description = "UniVim package to install.";
    };

    output = lib.mkOption {
      type = lib.types.nullOr lib.types.str;
      default = null;
      description = "Hyprland output connector to use; null follows the focused monitor.";
    };

    turbo = lib.mkEnableOption "turbo detection";
    interactiveOnly = lib.mkEnableOption "interactive-only detection";
    noClick = lib.mkEnableOption "display without injecting clicks";
    flat = lib.mkEnableOption "flattened leaf-only hints";
    colorByType = lib.mkOption {
      type = lib.types.bool;
      default = true;
      description = "Deprecated compatibility option; the interactive overlay always colors by element type.";
    };

    timeout = lib.mkOption {
      type = lib.types.ints.between 1 3600;
      default = 30;
      description = "Selection timeout in seconds.";
    };

    sessionTarget = lib.mkOption {
      type = lib.types.nullOr lib.types.str;
      default = null;
      description = ''
        Systemd user target that owns the service lifetime. When null, use
        hyprland-session.target for Home Manager's native Hyprland systemd
        integration, otherwise graphical-session.target.
      '';
      example = "graphical-session.target";
    };

    settings = {
      minScore = lib.mkOption {
        type = lib.types.float;
        default = 0.45;
        description = "Minimum confidence score for detected UI targets.";
      };

      includeText = lib.mkOption {
        type = lib.types.bool;
        default = true;
        description = "Whether to include text block targets alongside UI controls.";
      };

      fastHalfRes = lib.mkOption {
        type = lib.types.bool;
        default = false;
        description = "Whether to run Stage 1-4 detection at half resolution.";
      };

      collapseParagraphs = lib.mkOption {
        type = lib.types.bool;
        default = true;
        description = "Whether to collapse multi-line paragraphs into one root hint with drill-down.";
      };

      doubleTapMs = lib.mkOption {
        type = lib.types.int;
        default = 260;
        description = "Maximum interval in milliseconds between CapsLock taps.";
      };

      cacheTtlMs = lib.mkOption {
        type = lib.types.int;
        default = 900;
        description = "Maximum lifetime in milliseconds of the speculative first-tap detection cache.";
      };

      spotlightDim = lib.mkOption {
        type = lib.types.bool;
        default = true;
        description = "Whether to dim the background screenshot so hint badges stand out.";
      };

      pointerMotion = lib.mkOption {
        type = lib.types.enum [ "instant" "linear" "ease_out" ];
        default = "instant";
        description = "Cursor travel after selecting a hint; ease_out and linear glide at pointerSpeed pixels per second.";
      };
      pointerSpeed = lib.mkOption {
        type = lib.types.ints.between 200 6000;
        default = 1600;
        description = "Cursor glide speed in logical pixels per second.";
      };
    };

    hyprland = {
      enable = lib.mkEnableOption "global UniVim Hyprland shortcuts";

      hintsBind = lib.mkOption {
        type = lib.types.nullOr lib.types.str;
        default = null;
        example = "SUPER, F";
        description = "Global hints binding in '<modifiers>, <key>' form; null disables it.";
      };

      gridBind = lib.mkOption {
        type = lib.types.nullOr lib.types.str;
        default = null;
        example = "SUPER, G";
        description = "Global grid binding in '<modifiers>, <key>' form; null disables it.";
      };

      doubleTapCapsLock = lib.mkOption {
        type = lib.types.bool;
        default = false;
        description = "Bind non-consuming CapsLock taps for instant hints on double-tap.";
      };

      singlePressCapsLock = lib.mkOption {
        type = lib.types.bool;
        default = false;
        description = "Opt-in CapsLock keycode bindings and caps:none XKB option: Caps for hints, Shift+Caps for grid, Alt+Caps for debug. Disables evdev gestures and normal CapsLock. Merge any other kb_options yourself.";
      };

      evdev = lib.mkOption {
        type = lib.types.bool;
        default = true;
        description = "Listen passively on /dev/input/event* for modifier-aware CapsLock gestures.";
      };
    };

    keybindings = {
      enable = lib.mkEnableOption "declarative hints and grid keybindings";
      hints = lib.mkOption {
        type = lib.types.str;
        default = "SUPER, F";
        description = "Hyprland binding in '<modifiers>, <key>' form for hints.";
      };
      grid = lib.mkOption {
        type = lib.types.str;
        default = "SUPER, G";
        description = "Hyprland binding in '<modifiers>, <key>' form for grid.";
      };
      modifierGestures = lib.mkEnableOption "double-Shift and double-Ctrl activation gestures";
    };
  };

  config = lib.mkIf cfg.enable {
    assertions = [
      {
        assertion =
          !(bindings.enable || bindings.modifierGestures || cfg.hyprland.enable)
          || hyprlandEnabled;
        message = "services.univim keybindings require wayland.windowManager.hyprland.enable.";
      }
      {
        assertion =
          !(bindings.enable || bindings.modifierGestures || cfg.hyprland.enable)
          || configType == "hyprlang"
          || configType == "lua";
        message = "services.univim supports Hyprland configType hyprlang or lua.";
      }
      {
        assertion = !bindings.enable || validBinding bindings.hints;
        message = "services.univim.keybindings.hints must have the form '<modifiers>, <key>'.";
      }
      {
        assertion = !bindings.enable || validBinding bindings.grid;
        message = "services.univim.keybindings.grid must have the form '<modifiers>, <key>'.";
      }
      {
        assertion =
          !cfg.hyprland.enable
          || cfg.hyprland.hintsBind == null
          || validBinding cfg.hyprland.hintsBind;
        message = "services.univim.hyprland.hintsBind must have the form '<modifiers>, <key>'.";
      }
      {
        assertion =
          !cfg.hyprland.enable
          || cfg.hyprland.gridBind == null
          || validBinding cfg.hyprland.gridBind;
        message = "services.univim.hyprland.gridBind must have the form '<modifiers>, <key>'.";
      }
      {
        assertion = !bindings.enable || bindings.hints != bindings.grid;
        message = "services.univim hints and grid bindings must differ.";
      }
      {
        assertion = !cfg.hyprland.singlePressCapsLock || (cfg.hyprland.enable && !cfg.hyprland.doubleTapCapsLock);
        message = "services.univim.hyprland.singlePressCapsLock requires hyprland.enable and conflicts with doubleTapCapsLock.";
      }
      {
        assertion = sessionTarget != "";
        message = "services.univim.sessionTarget must not be empty.";
      }
    ];

    home.packages = [ cfg.package ];
    xdg.configFile."univim/settings.json".text = runtimeSettingsJson;

    # XKB consumes the CapsLock action; binding consumption alone only prevents
    # client delivery, not the compositor's internal lock-state update.
    # An existing kb_options definition will conflict rather than be discarded.
    systemd.user.services.univim = {
      Unit = {
        Description = "UniVim desktop navigation";
        After = [ sessionTarget ];
        PartOf = [ sessionTarget ];
      };
      Service = {
        Type = "simple";
        ExecStart = "${serviceLauncher}";
        Restart = "no";
        UMask = "0077";
      };
      Install.WantedBy = [ sessionTarget ];
    };

    wayland.windowManager.hyprland.settings = lib.mkMerge [
      (lib.mkIf cfg.hyprland.singlePressCapsLock { input.kb_options = "caps:none"; })
      (lib.mkIf (hyprlandEnabled && (bindings.enable || bindings.modifierGestures || cfg.hyprland.enable))
        (
          if luaConfiguration then
            {
              bind = explicitBindings ++ gestureBindings ++ globalBindings ++ singleCapsBindings;
              bindn = doubleTapBindings;
            }
          else
            let
              base =
                lib.optionalAttrs bindings.enable { bind = explicitBindings; }
                // lib.optionalAttrs bindings.modifierGestures { bindn = gestureBindings; };
              bindList = (base.bind or [ ]) ++ globalBindings ++ singleCapsBindings;
              bindnList = (base.bindn or [ ]) ++ doubleTapBindings;
            in
            base
            // lib.optionalAttrs (bindList != [ ]) { bind = bindList; }
            // lib.optionalAttrs (bindnList != [ ]) { bindn = bindnList; }
        ))
    ];
  };
}
