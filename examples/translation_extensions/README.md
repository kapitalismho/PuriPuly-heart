# Translation extension examples

`libretranslate.json` is a reference extension for the managed LibreTranslate service at `https://libretranslate.com/translate`.

The packaged Windows distribution includes this example under its `examples/translation_extensions` directory.

Copy a reviewed extension into the user's resolved `translation_extensions` directory when you want to use it. PuriPuly does not copy this example automatically, make a startup or migration request, or include an API key.

The managed `libretranslate.com` service requires an API key. Self-hosted or other LibreTranslate instances may use different credential requirements; edit the extension declaration for that deployment instead of assuming the managed service policy.

Automated tests use a local fake HTTP server and never contact the public service.
