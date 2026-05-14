# Isaac Launchable Usage

This project provides a simple way to interact with Isaac Lab and Isaac Sim, either on the cloud or locally.
See the [project repo](https://github.com/isaac-sim/isaac-launchable) for more detailed instructions, but below is a quickstart guide.

## Web Viewer for Isaac Sim UI

### To view in a separate tab
1. Run a command that starts Isaac Sim, or Isaac Lab (see below)
2. Copy your current URL (ie `https://isaac.brevlab-1234`)
3. Open a new tab in your browser and paste the URL, changing the end to `/viewer/` (ie `https://isaac.brevlab-1234/viewer`)
4. The Isaac Sim UI will appear when the app is ready. The page will say "Waiting for stream..." until then.

### To view inside VSCode
1. Run a command that starts Isaac Sim, or Isaac Lab (see below)
2. Press `Ctrl+Shift+P` (or `Cmd+Shift+P` on Mac)
3. Type "Simple Browser: Show"
4. Enter URL: `/viewer/`
5. The Isaac Sim UI will appear when the app is ready. The page will say "Waiting for stream..." until then.

## Running Isaac Lab 2.3

You can run any of the Isaac Lab scripts with the streaming Isaac Sim experience with the following command:

```console
cd /workspace/isaaclab
./isaaclab.sh -p scripts/tutorials/00_sim/create_empty.py --livestream 2
```

To run any other Isaac Lab commands, simply append the same argument as shown above: `--livestream 2`.

Then follow the Web Viewer for Isaac Sim UI instructions, if not using headless mode.

## Running Isaac Sim 5.1

You can run the streaming Isaac Sim application at any time with the following command.

```console
# Option 1: requires EULA acceptance
/isaac-sim/runheadless.sh

# Option 2: to automatically accept the EULA:
ACCEPT_EULA=y /isaac-sim/runheadless.sh
```

Then follow the Web Viewer for Isaac Sim UI instructions above, if not using headless mode.
