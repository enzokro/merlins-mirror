"""Utilities for starting and stopping py-spy profiling."""
import os
import subprocess
import signal

def start_profiling(main_pid: int, merlin_pid: int | None, main_profile_file: str = "main_profile.svg", merlin_profile_file: str = "merlin_profile.svg") -> tuple[subprocess.Popen | None, subprocess.Popen | None]:
    """
    Starts py-spy recording subprocesses for the main and Merlin processes.

    Args:
        main_pid: The Process ID of the main application process.
        merlin_pid: The Process ID of the Merlin worker process, or None if not available.
        main_profile_file: The output file name for the main process profile.
        merlin_profile_file: The output file name for the Merlin process profile.

    Returns:
        A tuple containing the subprocess.Popen objects for the main and Merlin
        py-spy processes, or None if a process couldn't be started.
    """
    py_spy_main_proc = None
    py_spy_merlin_proc = None
    py_spy_path = "py-spy"  # Assumes py-spy is in PATH

    print(f"Attempting to start profiling. Main PID: {main_pid}, Merlin PID: {merlin_pid}")

    # Ensure previous profiles are removed if they exist
    for f in [main_profile_file, merlin_profile_file]:
        if os.path.exists(f):
            try:
                os.remove(f)
                print(f"Removed existing profile: {f}")
            except OSError as e:
                print(f"Warning: Could not remove existing profile {f}: {e}")

    try:
        # Start profiling main process
        py_spy_main_proc = subprocess.Popen(
            [py_spy_path, "record", "-o", main_profile_file, "--pid", str(main_pid)],
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE
        )
        print(f"py-spy recording started for main process (PID: {main_pid}). Profile: '{main_profile_file}'")

        # Start profiling merlin process if PID is available
        if merlin_pid:
            py_spy_merlin_proc = subprocess.Popen(
                [py_spy_path, "record", "-o", merlin_profile_file, "--pid", str(merlin_pid)],
                stdout=subprocess.DEVNULL, stderr=subprocess.PIPE
            )
            print(f"py-spy recording started for Merlin process (PID: {merlin_pid}). Profile: '{merlin_profile_file}'")
        else:
            print("Merlin PID not provided. Profiling for Merlin process skipped.")

    except FileNotFoundError:
        print(f"Error: '{py_spy_path}' command not found. Please install py-spy (`pip install py-spy`) and ensure it's in your PATH.")
        # Clean up main process if merlin profiling failed due to py-spy not found
        if py_spy_main_proc and py_spy_main_proc.poll() is None:
            py_spy_main_proc.terminate()
            py_spy_main_proc.wait()
        py_spy_main_proc = None
        py_spy_merlin_proc = None # Already None or failed
    except Exception as e:
        print(f"Error starting py-spy: {e}")
        # Clean up any potentially started processes
        if py_spy_main_proc and py_spy_main_proc.poll() is None:
            py_spy_main_proc.terminate()
            py_spy_main_proc.wait()
        if py_spy_merlin_proc and py_spy_merlin_proc.poll() is None:
            py_spy_merlin_proc.terminate()
            py_spy_merlin_proc.wait()
        py_spy_main_proc = None
        py_spy_merlin_proc = None

    return py_spy_main_proc, py_spy_merlin_proc

def stop_profiling(py_spy_main_proc: subprocess.Popen | None, py_spy_merlin_proc: subprocess.Popen | None):
    """
    Stops the py-spy recording subprocesses gracefully.

    Args:
        py_spy_main_proc: The subprocess.Popen object for the main py-spy process.
        py_spy_merlin_proc: The subprocess.Popen object for the Merlin py-spy process.
    """
    print("Stopping py-spy recording...")
    stopped_cleanly = True
    profiling_was_active = False

    procs_to_stop = []
    if py_spy_main_proc:
        procs_to_stop.append((py_spy_main_proc, "main"))
        profiling_was_active = True
    if py_spy_merlin_proc:
        procs_to_stop.append((py_spy_merlin_proc, "merlin"))
        profiling_was_active = True

    if not profiling_was_active:
        print("Profiling was not active.")
        return

    for proc, name in procs_to_stop:
        if proc.poll() is not None: # Check if process already terminated
            print(f"py-spy ({name}) process was already stopped.")
            continue

        try:
            # Attempt graceful shutdown first
            print(f"Sending SIGINT to py-spy ({name})...")
            proc.send_signal(signal.SIGINT) # Ask py-spy to finalize the file
            proc.wait(timeout=10) # Wait longer for potentially large file saves
            print(f"py-spy ({name}) process stopped gracefully.")

            # Check for errors during py-spy execution after it stops
            stderr_output = proc.stderr.read().decode()
            if stderr_output:
                print(f"py-spy ({name}) stderr output:
{stderr_output}")

        except subprocess.TimeoutExpired:
            print(f"Warning: py-spy ({name}) did not stop gracefully after 10s. Forcing termination.")
            proc.terminate() # More forceful
            try:
                proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                print(f"Warning: py-spy ({name}) did not terminate after SIGTERM. Killing.")
                proc.kill() # Most forceful
                try:
                    proc.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    print(f"Error: Could not kill py-spy ({name}) process.") # Should not happen
            stopped_cleanly = False
        except Exception as e:
            print(f"Error stopping py-spy ({name}): {e}")
            stopped_cleanly = False
            # Ensure we try to kill if any other exception occurs during stop
            if proc.poll() is None:
                proc.kill()
                proc.wait(timeout=1)

    if stopped_cleanly:
         print("Profiling data saved successfully.")
    else:
        print("Profiling data might be incomplete due to errors or forceful termination during shutdown.") 