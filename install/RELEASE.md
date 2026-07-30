# Making a release

This is for you, the person who builds and shares the app. The people you share
it with never read this and never need Node.js. They only run the install
script.

The app ships with its interface already built (static files in `frontend/out`).
That prebuilt interface is the same for every user, so you build it once here,
and it works on macOS, Windows, and Linux. Building it is the only step that
needs Node.js, and only you do it.

## Step 1: build the interface

From the project root, on a machine with Node.js 18+:

```bash
bash install/prepare_release.sh
```

This runs the web build and writes the static interface to `frontend/out`. It
checks the result and stops with a clear message if the build did not produce
`frontend/out/index.html`.

If you are on Windows and prefer not to use a bash shell, run the same build by
hand:

```powershell
cd frontend
npm ci
$env:NEXT_PUBLIC_API_BASE = ""
npm run build
```

`NEXT_PUBLIC_API_BASE` must be empty so the built app talks to the local backend
on its own address, which is how the launcher serves it.

## Step 2: package the folder

Package the project folder into one archive to share. Include `frontend/out` (the
prebuilt interface) and `assets` (the opening book and models). Leave out the big
folders the install script rebuilds on the user's machine (the Python
environments, the downloaded engine, the model weights) and the private working
notes.

`frontend/out` is in `.gitignore`, so do not build the archive from `git`. Copy
the files instead. This command, run from the folder ABOVE the project, makes a
clean archive (tested on macOS and Linux tar):

```bash
tar -czf chess-review.tar.gz \
  --exclude='Time-Aware-Chess-Analysis/.git' \
  --exclude='Time-Aware-Chess-Analysis/.venv' \
  --exclude='Time-Aware-Chess-Analysis/.venv_maia' \
  --exclude='Time-Aware-Chess-Analysis/data' \
  --exclude='Time-Aware-Chess-Analysis/engines' \
  --exclude='Time-Aware-Chess-Analysis/frontend/node_modules' \
  --exclude='Time-Aware-Chess-Analysis/frontend/.next' \
  --exclude='*/__pycache__' \
  --exclude='*.pyc' \
  --exclude='*/.pytest_cache' \
  --exclude='*/.ruff_cache' \
  --exclude='.DS_Store' \
  --exclude='Time-Aware-Chess-Analysis/BUILD.md' \
  --exclude='Time-Aware-Chess-Analysis/BUILD2.md' \
  --exclude='Time-Aware-Chess-Analysis/PLAN.md' \
  --exclude='Time-Aware-Chess-Analysis/context.md' \
  Time-Aware-Chess-Analysis
```

Before you share it, confirm the prebuilt interface made it in:

```bash
tar -tzf chess-review.tar.gz | grep frontend/out/index.html
```

If that prints nothing, you skipped Step 1. Build the interface, then package
again.

## What the user does

They unpack the archive and run the install script for their system
(`install/install.sh` on macOS or Linux, `install/install.ps1` on Windows). The
install downloads the engine, the model, and the model weights, sets up the
Python environments, and uses the prebuilt interface as-is. No Node.js, no web
build, ever, on their machine. If the prebuilt interface is somehow missing from
the package, the install stops early and tells you (the release builder) to run
`install/prepare_release.sh`, and it never falls back to needing Node.js.
