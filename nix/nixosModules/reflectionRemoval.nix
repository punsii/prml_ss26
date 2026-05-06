{
  config,
  lib,
  pkgs,
  ...
}:

let
  cfg = config.punsi.reflectionRemoval;
in
{
  options.punsi.reflectionRemoval = {
    enable = lib.mkEnableOption "the reflection removal streamlit service with caddy reverse proxy";

    port = lib.mkOption {
      type = lib.types.port;
      default = 8927;
      description = "Local port for the streamlit service.";
    };

    dataDir = lib.mkOption {
      type = lib.types.str;
      description = "Path to the directory containing image data for the app.";
    };

    hostName = lib.mkOption {
      type = lib.types.str;
      example = "reflection-removal.example.com";
      description = "Public hostname served by Caddy as a reverse proxy to the streamlit service.";
    };

    acmeHost = lib.mkOption {
      type = lib.types.str;
      example = "example.com";
      description = "Name of the ACME certificate (security.acme.certs.<name>) to use for the vhost.";
    };
  };

  config = lib.mkIf cfg.enable {

    systemd.timers."reflection-removal-restart" = {
      wantedBy = [ "timers.target" ];
      timerConfig = {
        OnCalendar = "*-*-* 03:30:00";
        RandomizedDelaySec = "1800";
        Persistent = "true";
        Unit = "reflection-removal-restart.service";
      };
    };

    systemd.services."reflection-removal-restart" = {
      description = "Service for restarting the reflection removal streamlit app";
      serviceConfig = {
        Type = "oneshot";
        ExecStart = "${pkgs.systemd}/bin/systemctl restart reflection-removal-streamlit.service";
      };
    };

    systemd.services.reflection-removal-streamlit = {
      description = "Reflection Removal Streamlit App";
      wantedBy = [ "multi-user.target" ];
      wants = [ "network-online.target" ];
      after = [ "network-online.target" ];

      path = [
        pkgs.nix
        pkgs.git
        pkgs.openssh
      ];

      environment = {
        STREAMLIT_SERVER_PORT = toString cfg.port;
        STREAMLIT_SERVER_ADDRESS = "127.0.0.1";
        STREAMLIT_SERVER_HEADLESS = "true";
        STREAMLIT_BROWSER_GATHER_USAGE_STATS = "false";
        IMAGE_DIR = cfg.dataDir;
        HOME = "/var/lib/reflection-removal";
        XDG_CACHE_HOME = "/var/cache/reflection-removal";
      };

      unitConfig = {
        StartLimitIntervalSec = "10min";
        StartLimitBurst = 5;
      };

      serviceConfig = {
        ExecStart = "${pkgs.nix}/bin/nix run --refresh github:punsii/prml_ss26?ref=main#runStreamlitService";
        WorkingDirectory = cfg.dataDir;
        Restart = "always";
        RestartSec = "30s";
        DynamicUser = true;
        StateDirectory = "reflection-removal";
        CacheDirectory = "reflection-removal";
      };
    };

    services.caddy = {
      enable = true;
      virtualHosts.${cfg.hostName} = {
        useACMEHost = cfg.acmeHost;
        extraConfig = ''
          reverse_proxy 127.0.0.1:${toString cfg.port}
          encode gzip
        '';
      };
    };

    networking.firewall.allowedTCPPorts = [
      80
      443
    ];
  };
}
