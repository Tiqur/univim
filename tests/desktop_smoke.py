#!/usr/bin/env python3
"""Opt-in live Wayland smoke test: in-memory capture and overlay, never clicks.

Temporarily takes keyboard focus. Run only with explicit user approval. Does not
change bindings, inject keys, or persist screenshots. This is not a click test.
"""
import argparse
import json
from pathlib import Path
import subprocess
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-live-overlay", action="store_true", required=True)
    parser.add_argument("--binary", type=Path, default=Path("target/release/univim-desktop"))
    args = parser.parse_args()
    binary = str(args.binary.resolve())
    outputs = json.loads(subprocess.check_output(["hyprctl", "-j", "monitors"], text=True))
    results = []

    def command(name):
        return subprocess.check_output([binary, name], text=True, stderr=subprocess.PIPE, timeout=3)

    def wait(predicate, seconds=4):
        deadline = time.monotonic() + seconds
        last = None
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError("desktop service exited unexpectedly")
            try:
                last = json.loads(command("status"))
                if predicate(last):
                    return last
            except subprocess.CalledProcessError:
                pass
            time.sleep(0.05)
        raise RuntimeError(f"status condition timed out: {last}")

    for index, output in enumerate(outputs):
        if output.get("disabled") or not output.get("dpmsStatus", True):
            continue
        entry = {"output": output["name"], "transform": output["transform"], "checks": []}
        process = subprocess.Popen(
            [binary, "serve", "--no-click", "--turbo", "--timeout", "5", "--output", output["name"]],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        try:
            idle = wait(lambda s: not s["active"])
            assert idle["no_click"] and not idle["overlay_mapped"]
            for mode in ("grid", "hints"):
                command(mode)
                active = wait(lambda s: s["active"] and not s["loading"] and s["overlay_mapped"])
                assert active["no_click"] and active["output"] == output["name"]
                command("cancel")
                stopped = wait(lambda s: not s["active"] and not s["sync_pending"])
                assert not stopped["overlay_mapped"]
                entry["checks"].append({"mode": mode, "mapped": True, "clean_cancel": True})
            # Cancel immediately while capture/detection may still be in flight.
            command("hints")
            command("cancel")
            wait(lambda s: not s["active"] and not s["sync_pending"])
            time.sleep(0.4)
            assert not json.loads(command("status"))["active"]
            entry["stale_result_rejected"] = True
            command("grid")
            wait(lambda s: s["overlay_mapped"])
            wait(lambda s: not s["active"] and not s["overlay_mapped"], seconds=7)
            entry["timeout_cleanup"] = True
            if index % 2:
                process.terminate()
                entry["shutdown"] = "SIGTERM"
            else:
                command("quit")
                entry["shutdown"] = "quit"
            stdout, stderr = process.communicate(timeout=5)
            if process.returncode != 0:
                raise RuntimeError(f"desktop exited {process.returncode}: {stderr}")
            entry.update(passed=True, diagnostics=stderr)
            print(f"PASS {output['name']}: grid, hints, cancellation, stale result, timeout, {entry['shutdown']}")
        except Exception as error:
            entry.update(passed=False, error=str(error))
            print(f"FAIL {output['name']}: {error}")
        finally:
            if process.poll() is None:
                try:
                    command("quit")
                    stdout, stderr = process.communicate(timeout=4)
                except (subprocess.SubprocessError, OSError):
                    process.terminate()
                    try:
                        stdout, stderr = process.communicate(timeout=4)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        stdout, stderr = process.communicate()
                entry.setdefault("diagnostics", stderr)
            else:
                stdout, stderr = process.communicate()
                entry.setdefault("diagnostics", stderr)
            results.append(entry)
    report = Path("target/desktop-smoke.json")
    report.parent.mkdir(exist_ok=True)
    report.write_text(json.dumps(results, indent=2) + "\n")
    print(f"Report: {report}")
    return 0 if results and all(r["passed"] for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
