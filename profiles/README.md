## Profile Layout

- `profiles/pipeline.profile.example.yaml`: minimal baseline profile
- `profiles/local_attached/`: profiles that connect to an already-running local backend
- `profiles/local_spawned/`: profiles where scriba launches and manages the backend process
- `profiles/remote/`: hosted provider profiles (for example OpenRouter)
- `profiles/hybrid/`: mixed-topology profiles

All scripts and docs now reference profile paths from this directory structure.
