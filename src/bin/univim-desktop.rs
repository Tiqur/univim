//! Persistent desktop service and small, permission-scoped control client.
#[cfg(target_os = "linux")]
mod linux {
    use clap::{Parser, Subcommand};
    use std::{
        fs,
        os::unix::{
            fs::{DirBuilderExt, MetadataExt, PermissionsExt},
            net::UnixDatagram,
        },
        path::{Path, PathBuf},
        time::Duration,
    };
    use univim::{
        Options,
        desktop::{Config, hyprland, wayland},
    };

    #[derive(Parser)]
    #[command(about = "Native Wayland navigation for Hyprland")]
    pub struct Args {
        #[command(subcommand)]
        command: Command,
    }
    #[derive(Subcommand)]
    enum Command {
        /// Enumerate capabilities and automatically discovered outputs. No capture/input.
        Check,
        /// Print opt-in Hyprland configuration; does not edit your config.
        Bindings,
        /// Run persistently without mapping any surface while idle.
        Serve {
            #[arg(long)]
            output: Option<String>,
            #[arg(long)]
            turbo: bool,
            #[arg(long)]
            interactive_only: bool,
            /// Display normally, but never inject clicks.
            #[arg(long)]
            no_click: bool,
            #[arg(long,default_value_t=30,value_parser=clap::value_parser!(u64).range(1..=3600))]
            timeout: u64,
        },
        Hints,
        Grid,
        Cancel,
        Status,
        Quit,
    }

    fn private_directory_metadata(is_dir: bool, owner: u32, mode: u32, effective_uid: u32) -> bool {
        is_dir && owner == effective_uid && mode & 0o077 == 0
    }

    fn directory() -> Result<PathBuf, Box<dyn std::error::Error>> {
        let runtime = std::env::var_os("XDG_RUNTIME_DIR").ok_or("XDG_RUNTIME_DIR is unset")?;
        let display = std::env::var("WAYLAND_DISPLAY").map_err(|_| "WAYLAND_DISPLAY is unset")?;
        if display.contains('/') || display.contains("..") {
            return Err("unsupported WAYLAND_DISPLAY".into());
        }
        let base = PathBuf::from(runtime);
        // SAFETY: geteuid takes no pointers and has no preconditions.
        let effective_uid = unsafe { nix::libc::geteuid() };
        let metadata = fs::symlink_metadata(&base)?;
        if !private_directory_metadata(
            metadata.is_dir(),
            metadata.uid(),
            metadata.permissions().mode(),
            effective_uid,
        ) {
            return Err("unsafe runtime base directory permissions/ownership".into());
        }
        let path = base.join(format!("univim-{display}"));
        match fs::DirBuilder::new().mode(0o700).create(&path) {
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {}
            Err(e) => return Err(e.into()),
        }
        let metadata = fs::symlink_metadata(&path)?;
        if !private_directory_metadata(
            metadata.is_dir(),
            metadata.uid(),
            metadata.permissions().mode(),
            effective_uid,
        ) {
            return Err("unsafe UniVim runtime directory permissions/ownership".into());
        }
        Ok(path)
    }
    struct SocketPath(PathBuf);
    impl Drop for SocketPath {
        fn drop(&mut self) {
            let _ = fs::remove_file(&self.0);
        }
    }

    fn bind_service(path: &Path) -> Result<UnixDatagram, Box<dyn std::error::Error>> {
        // Exclusive bind fails closed on live or stale sockets; never unlink
        // another daemon's address automatically.
        let socket = UnixDatagram::bind(path)?;
        fs::set_permissions(path, fs::Permissions::from_mode(0o600))?;
        Ok(socket)
    }

    pub fn main() -> Result<(), Box<dyn std::error::Error>> {
        let args = Args::parse();
        match args.command {
            Command::Check => {
                println!(
                    "{}",
                    serde_json::to_string_pretty(
                        &wayland::check().map_err(std::io::Error::other)?
                    )?
                );
                println!(
                    "Focused output: {:?}",
                    hyprland::selected_output(None).map_err(std::io::Error::other)?
                );
            }
            Command::Bindings => {
                println!(
                    "# Start once after your Wayland environment is ready.\nexec-once = univim-desktop serve\n# Nonconsuming modifier bindings preserve ordinary modifier events.\nbindn = SHIFT, Shift_L, global, univim:shift-left\nbindn = SHIFT, Shift_R, global, univim:shift-right\nbindn = CTRL, Control_L, global, univim:control-left\nbindn = CTRL, Control_R, global, univim:control-right\n# Anonymous modifier notifications cannot reliably reject intervening chord keys.\n# Optional explicit activation alternatives:\n# bind = SUPER, F, exec, univim-desktop hints\n# bind = SUPER, G, exec, univim-desktop grid"
                );
            }
            Command::Serve {
                output,
                turbo,
                interactive_only,
                no_click,
                timeout,
            } => {
                opencv::core::set_num_threads(1)?;
                let path = directory()?.join("control.sock");
                let socket=bind_service(&path).map_err(|e|format!("{e}; if a prior daemon crashed, remove {} only after verifying it is stopped",path.display()))?;
                let _cleanup = SocketPath(path);
                wayland::run(
                    Config {
                        output,
                        options: Options {
                            turbo,
                            include_text: !interactive_only,
                            ..Options::default()
                        },
                        no_click,
                        timeout: Duration::from_secs(timeout),
                    },
                    socket,
                )
                .map_err(std::io::Error::other)?;
            }
            command => {
                let directory = directory()?;
                let reply = directory.join(format!("client-{}.sock", std::process::id()));
                let socket = UnixDatagram::bind(&reply)?;
                let _cleanup = SocketPath(reply);
                socket.set_read_timeout(Some(Duration::from_secs(2)))?;
                socket.set_write_timeout(Some(Duration::from_secs(2)))?;
                let command = match command {
                    Command::Hints => "hints",
                    Command::Grid => "grid",
                    Command::Cancel => "cancel",
                    Command::Status => "status",
                    Command::Quit => "quit",
                    _ => unreachable!(),
                };
                socket
                    .connect(directory.join("control.sock"))
                    .map_err(|e| {
                        format!(
                            "desktop service unavailable: {e}; start `univim-desktop serve` first"
                        )
                    })?;
                socket.send(command.as_bytes())?;
                if command == "status" {
                    let mut buffer = [0u8; 4096];
                    let n = socket.recv(&mut buffer)?;
                    println!("{}", std::str::from_utf8(&buffer[..n])?);
                }
            }
        }
        Ok(())
    }

    #[cfg(test)]
    mod tests {
        use super::private_directory_metadata;

        #[test]
        fn runtime_directories_require_private_effective_uid_ownership() {
            assert!(private_directory_metadata(true, 1000, 0o40700, 1000));
            assert!(private_directory_metadata(true, 0, 0o40700, 0));
            assert!(!private_directory_metadata(true, 1001, 0o40700, 1000));
            for mode in [0o40701, 0o40710, 0o40720, 0o40740, 0o40704, 0o40702] {
                assert!(!private_directory_metadata(true, 1000, mode, 1000));
            }
        }

        #[test]
        fn runtime_directories_reject_symlinks_and_other_non_directories() {
            for mode in [0o120700, 0o100700, 0o140700] {
                assert!(!private_directory_metadata(false, 1000, mode, 1000));
            }
        }
    }
}

#[cfg(target_os = "linux")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    linux::main()
}
#[cfg(not(target_os = "linux"))]
fn main() {
    eprintln!("univim-desktop requires Linux/Wayland");
    std::process::exit(1);
}
