# HTTP extension examples

`mymemory.json` is a reference extension for the free MyMemory translation API at `https://api.mymemory.translated.net/get`.

The packaged Windows distribution includes this example under its `examples/http_extensions` directory.

Copy a reviewed extension into the user's resolved `http_extensions` directory when you want to use it. PuriPuly does not copy this example automatically, make a startup or migration request, or include an API key.

Full format reference and JSON Schema: `docs/http-extensions.md` and `docs/http-extension.schema.json` in the repository.

MyMemory does not require an API key for the free service; usage is subject to the service's usage limits. The request body sets `mt` to `1` so text without a translation memory match falls back to machine translation.

Automated tests use a local fake HTTP server and never contact the public service.
