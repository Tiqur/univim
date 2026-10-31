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
    "serve"
  ]
  ++ lib.optionals (cfg.output != null) [
    "--output"
    cfg.output
  ]
  ++ lib.optional cfg.turbo "--turbo"
  ++ lib.optional cfg.interactiveOnly "--interactive-only"
  ++ lib.optional cfg.noClick "--no-click"
  ++ lib.optional cfg.flat "--flat"
  ++ lib.optional cfg.colorByType "--color-by-type"
  ++ [
    "--timeout"
    (toString cfg.timeout)
  ];
  serviceLauncher = pkgs.writeShellScript "univim-desktop-serve" ''
    exec ${lib.escapeShellArgs serviceArgs}
  '';

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
    colorByType = lib.mkEnableOption "target and hint tag colors selected by element type";

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
        assertion = !(bindings.enable || bindings.modifierGestures) || hyprlandEnabled;
        message = "services.univim keybindings require wayland.windowManager.hyprland.enable.";
      }
      {
        assertion =
          !(bindings.enable || bindings.modifierGestures) || configType == "hyprlang" || configType == "lua";
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
        assertion = !bindings.enable || bindings.hints != bindings.grid;
        message = "services.univim hints and grid bindings must differ.";
      }
      {
        assertion = sessionTarget != "";
        message = "services.univim.sessionTarget must not be empty.";
      }
    ];

    home.packages = [ cfg.package ];

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

    wayland.windowManager.hyprland.settings =
      lib.mkIf (hyprlandEnabled && (bindings.enable || bindings.modifierGestures))
        (
          if luaConfiguration then
            {
              bind = explicitBindings ++ gestureBindings;
            }
          else
            lib.optionalAttrs bindings.enable { bind = explicitBindings; }
            // lib.optionalAttrs bindings.modifierGestures { bindn = gestureBindings; }
        );
  };
}
