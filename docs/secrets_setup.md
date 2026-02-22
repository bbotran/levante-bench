# Secrets setup

The LEVANTE assets bucket is **public**; no authentication is required. The bucket base URL is set in config (`levante_bench.config.defaults`) and can be overridden with the environment variable `LEVANTE_ASSETS_BUCKET_URL`. Do **not** commit real secrets (API keys, private bucket URLs) in the repository.

If you add features that require secrets (e.g. a private Redivis token or GCP credentials), store them in a **gitignored** file such as `.secrets` at the project root, and document the expected format here. Never commit `.secrets` or any file containing secrets.
