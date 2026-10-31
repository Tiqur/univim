//! Read-only Hyprland IPC. Never changes the user's configuration or focus.
use super::geometry::Output;
use serde_json::Value;
use std::{
    io::{Read, Write},
    os::unix::net::UnixStream,
    path::PathBuf,
    time::Duration,
};

fn query(command: &str) -> Result<Value, String> {
    let runtime = std::env::var_os("XDG_RUNTIME_DIR").ok_or("XDG_RUNTIME_DIR is unset")?;
    let signature = std::env::var("HYPRLAND_INSTANCE_SIGNATURE")
        .map_err(|_| "HYPRLAND_INSTANCE_SIGNATURE is unset; this backend requires Hyprland")?;
    if signature.contains('/') || signature.contains("..") {
        return Err("invalid Hyprland instance signature".into());
    }
    let path = PathBuf::from(runtime)
        .join("hypr")
        .join(signature)
        .join(".socket.sock");
    let mut socket = UnixStream::connect(path).map_err(|e| format!("Hyprland IPC: {e}"))?;
    socket
        .set_read_timeout(Some(Duration::from_secs(1)))
        .map_err(|e| e.to_string())?;
    socket
        .set_write_timeout(Some(Duration::from_secs(1)))
        .map_err(|e| e.to_string())?;
    socket
        .write_all(format!("j/{command}").as_bytes())
        .map_err(|e| e.to_string())?;
    let mut reply = Vec::new();
    socket
        .take(4 * 1024 * 1024)
        .read_to_end(&mut reply)
        .map_err(|e| e.to_string())?;
    serde_json::from_slice(&reply).map_err(|e| format!("Hyprland {command}: {e}"))
}

fn monitors() -> Result<Vec<Value>, String> {
    query("monitors")?
        .as_array()
        .cloned()
        .ok_or("invalid monitor reply".into())
}

fn output(value: &Value) -> Result<Output, String> {
    let integer = |name: &str| {
        value[name]
            .as_i64()
            .and_then(|v| i32::try_from(v).ok())
            .ok_or_else(|| format!("invalid monitor {name}"))
    };
    let result = Output {
        name: value["name"].as_str().ok_or("monitor lacks name")?.into(),
        x: integer("x")?,
        y: integer("y")?,
        width: integer("width")?,
        height: integer("height")?,
        scale: value["scale"].as_f64().ok_or("monitor lacks scale")?,
        transform: integer("transform")?,
    };
    result.logical_size()?;
    Ok(result)
}

pub fn selected_output(name: Option<&str>) -> Result<Output, String> {
    let monitors = monitors()?;
    let monitor = monitors
        .iter()
        .find(|m| {
            !m["disabled"].as_bool().unwrap_or(false)
                && m["dpmsStatus"].as_bool().unwrap_or(true)
                && name.map_or_else(
                    || m["focused"].as_bool().unwrap_or(false),
                    |n| m["name"].as_str() == Some(n),
                )
        })
        .ok_or("requested/focused output is not available")?;
    output(monitor)
}

/// Deliberately excludes keyboard focus: a layer-shell surface owns that while
/// selecting. Client/workspace/layout changes still invalidate captured targets.
#[derive(Clone, Debug, PartialEq)]
pub struct Context {
    outputs: Vec<(Output, bool)>,
    focused_output: String,
    workspaces: Vec<(String, i64, i64)>,
    client: Option<String>,
    visible_clients: Vec<Client>,
}

#[derive(Clone, Debug, PartialEq, serde::Deserialize)]
struct Workspace {
    id: i64,
}

#[derive(Clone, Debug, PartialEq, serde::Deserialize)]
struct Client {
    address: String,
    at: [i32; 2],
    size: [i32; 2],
    workspace: Workspace,
    monitor: i64,
    mapped: bool,
    hidden: bool,
    fullscreen: i64,
    floating: bool,
    pinned: bool,
}

fn visible_clients(
    reply: Value,
    workspaces: &[(String, i64, i64)],
    monitors: &[i64],
) -> Result<Vec<Client>, String> {
    let mut clients: Vec<Client> =
        serde_json::from_value(reply).map_err(|e| format!("invalid clients reply: {e}"))?;
    if clients.iter().any(|c| c.address.is_empty()) {
        return Err("client lacks address".into());
    }
    clients.retain(|c| {
        c.mapped
            && !c.hidden
            && (workspaces.iter().any(|(_, active, special)| {
                c.workspace.id == *active || (*special != 0 && c.workspace.id == *special)
            }) || (c.pinned && monitors.contains(&c.monitor)))
    });
    clients.sort_by(|a, b| a.address.cmp(&b.address));
    if clients
        .windows(2)
        .any(|pair| pair[0].address == pair[1].address)
    {
        return Err("duplicate client address".into());
    }
    Ok(clients)
}

pub fn context() -> Result<Context, String> {
    let monitors = monitors()?;
    let mut outputs = Vec::new();
    let mut workspaces = Vec::new();
    let mut focused_output = String::new();
    let mut visible_monitors = Vec::new();
    for m in &monitors {
        if m["disabled"].as_bool().ok_or("invalid monitor disabled")? {
            continue;
        }
        let o = output(m)?;
        if m["focused"].as_bool().ok_or("invalid monitor focused")? {
            focused_output = o.name.clone();
        }
        workspaces.push((
            o.name.clone(),
            m["activeWorkspace"]["id"]
                .as_i64()
                .ok_or("invalid active workspace")?,
            m["specialWorkspace"]["id"]
                .as_i64()
                .ok_or("invalid special workspace")?,
        ));
        visible_monitors.push(m["id"].as_i64().ok_or("invalid monitor id")?);
        outputs.push((
            o,
            m["dpmsStatus"]
                .as_bool()
                .ok_or("invalid monitor DPMS state")?,
        ));
    }
    outputs.sort_by(|a, b| a.0.name.cmp(&b.0.name));
    workspaces.sort();
    let active = query("activewindow")?;
    let active = active.as_object().ok_or("invalid active window reply")?;
    let client = if active.is_empty() {
        None
    } else {
        Some(
            active
                .get("address")
                .and_then(Value::as_str)
                .filter(|address| !address.is_empty())
                .ok_or("invalid active window address")?
                .to_owned(),
        )
    };
    let visible_clients = visible_clients(query("clients")?, &workspaces, &visible_monitors)?;
    Ok(Context {
        outputs,
        focused_output,
        workspaces,
        client,
        visible_clients,
    })
}

pub fn still_current(previous: &Context) -> Result<bool, String> {
    let current = context()?;
    // Hyprland may report no active client while a layer owns keyboard focus.
    // A different nonempty active client, however, must never receive stale clicks.
    Ok(previous.outputs == current.outputs
        && previous.focused_output == current.focused_output
        && previous.workspaces == current.workspaces
        && previous.visible_clients == current.visible_clients
        && (current.client.is_none() || previous.client == current.client))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn client_snapshot_tracks_geometry_not_order_or_titles() {
        let client = serde_json::json!({
            "address":"0x1", "at":[10,20], "size":[300,200],
            "workspace":{"id":1}, "monitor":0, "mapped":true,
            "hidden":false, "fullscreen":0, "floating":false, "pinned":false,
            "title":"first", "focusHistoryID":0
        });
        let mut other = client.clone();
        other["address"] = serde_json::json!("0x2");
        let snapshot = |items| visible_clients(items, &[("DP-1".into(), 1, 0)], &[0]).unwrap();
        let original = snapshot(serde_json::json!([client, other]));
        assert_eq!(original, snapshot(serde_json::json!([other, client])));
        let mut changed = client.clone();
        changed["title"] = serde_json::json!("new title");
        changed["focusHistoryID"] = serde_json::json!(12);
        assert_eq!(original, snapshot(serde_json::json!([changed, other])));
        for field in ["at", "size"] {
            let mut changed = client.clone();
            changed[field][0] = serde_json::json!(999);
            assert_ne!(original, snapshot(serde_json::json!([changed, other])));
        }
    }

    #[test]
    fn client_snapshot_filters_visibility_and_rejects_malformed_replies() {
        let mut client = serde_json::json!({
            "address":"0x1", "at":[10,20], "size":[300,200],
            "workspace":{"id":2}, "monitor":0, "mapped":true,
            "hidden":false, "fullscreen":0, "floating":false, "pinned":false
        });
        let snapshot = |items| visible_clients(items, &[("DP-1".into(), 1, 0)], &[0]);
        assert!(snapshot(serde_json::json!([client])).unwrap().is_empty());
        client["pinned"] = serde_json::json!(true);
        assert_eq!(snapshot(serde_json::json!([client])).unwrap().len(), 1);
        client["hidden"] = serde_json::json!(true);
        assert!(snapshot(serde_json::json!([client])).unwrap().is_empty());
        client["size"] = serde_json::json!([10]);
        assert!(snapshot(serde_json::json!([client])).is_err());
        assert!(snapshot(serde_json::json!({})).is_err());
    }
    #[test]
    fn monitor_metadata_is_validated() {
        let m = serde_json::json!({"name":"rotated","x":-1080,"y":0,"width":1920,"height":1080,"scale":1.5,"transform":3});
        let o = output(&m).unwrap();
        assert_eq!(o.logical_size().unwrap(), (720., 1280.));
        let mut invalid = m;
        invalid["scale"] = serde_json::json!(0);
        assert!(output(&invalid).is_err());
    }
}
