# Static demo gallery

HARPOR's public site is now a static gallery. Accounts, approvals, Supabase storage,
remote job controls, and personal workers have been removed. Run jobs with
`python -m harpor.webapp` or `python -m harpor.webapp.run` on the GPU machine.
Use the SSH tunnel printed by the local server to access it from another computer.

## Build and preview

Keep curated demos outside the repository. Each demo contains `demo.json` and its
video, poster, GLB and ZIP files. To create and validate demos:

```bash
python -m harpor.webapp.demos add /path/to/job --out /path/to/demos
python -m harpor.webapp.demos check --help
python -m harpor.webapp.site build --demos /path/to/demos
python -m http.server --directory site 8000
```

`WEBAPP_DEMOS` supplies the default demo folder. `--out DIR` chooses a build folder.
`--no-demos` explicitly builds an empty gallery, including when replacing a build
that contained demos. Existing demo validation and private-data checks still apply.
The local app also serves the gallery with `--demos DIR`.

The generated `index.html` and `demos.html` show the same gallery. The build includes
help, styles, the viewer and security headers; it needs no cloud configuration.
Deploy the generated folder with your static host's usual process.

## Existing deployments

Removing this code does not disable an already deployed site or delete a Supabase
project. Replace the deployed site with a gallery build to remove its old account UI.
Retire the former backend separately after preserving any data you need. Local job
folders and curated demo source files are unaffected.
