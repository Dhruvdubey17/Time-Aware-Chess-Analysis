# Making a release

This is for you, the person who builds and shares the app. The people you share
it with never read this and never need Node.js. They only run the install
script.

The app ships with its interface already built (static files in `frontend/out`).
That prebuilt interface is the same for every user, so you build it once here,
and it works on macOS, Windows, and Linux. Building it is the only step that
needs Node.js, and only you do it.

## Build and package (one command)

From anywhere inside the project, on a machine with Node.js 18+:

```bash
bash install/prepare_release.sh
```

This does everything:

1. Builds the web interface into static files at `frontend/out`.
2. Checks the build produced `frontend/out/index.html`, and stops with a clear
   message if it did not.
3. Packages the whole app into one archive, `chess-review.tar.gz`, placed one
   level ABOVE the project folder (so it is never packed inside itself).
4. Confirms the prebuilt interface actually made it into the archive before it
   says "Release ready".

The archive includes the prebuilt interface and the bundled assets. It leaves out
the big folders the user's install rebuilds (the Python environments, the
downloaded engine, the model weights, `node_modules`, build caches) and the
private working notes. It ends up around half a megabyte.

Share that one `chess-review.tar.gz` file. The person you send it to unpacks it
and runs the install script for their system:

- macOS or Linux: `bash install/install.sh`
- Windows: `powershell -ExecutionPolicy Bypass -File install\install.ps1`

The install downloads the engine, the model, and the model weights, sets up the
Python environments, and uses the prebuilt interface as-is. No Node.js, no web
build, ever, on their machine. If the prebuilt interface is somehow missing from
the package, the install stops early and tells you (the release builder) to run
`install/prepare_release.sh`, and it never falls back to needing Node.js.

## Packaging by hand (only if you cannot run the script)

If you build on Windows, or want to package yourself, build the interface first:

```powershell
cd frontend
npm ci
$env:NEXT_PUBLIC_API_BASE = ""
npm run build
cd ..
```

`NEXT_PUBLIC_API_BASE` must be empty so the built app talks to the local backend
on its own address, which is how the launcher serves it.

Then make the archive. `frontend/out` is in `.gitignore`, so do not build the
archive from `git`; copy the files. This `tar` must run from the folder ABOVE the
project (note the `cd ..`), because the last argument is the project folder by
name:

```bash
cd ..
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

If your project folder has a different name, use that name in the command. Before
you share it, confirm the prebuilt interface made it in:

```bash
tar -tzf chess-review.tar.gz | grep frontend/out/index.html
```

If that prints nothing, you skipped the build step.
