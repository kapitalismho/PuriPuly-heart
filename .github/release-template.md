## How to download
Download `{{INSTALLER_EXE}}` from the Assets section below, then run it.

## How to use
- [English](https://github.com/kapitalismho/PuriPuly-heart/blob/dev/README.md#using-your-own-api-keys)
- [한국어](https://github.com/kapitalismho/PuriPuly-heart/blob/dev/docs/readme/README.ko.md#자신의-api-키-사용하기)
- [日本語](https://github.com/kapitalismho/PuriPuly-heart/blob/dev/docs/readme/README.ja.md#自分のapiキーを使う)
- [简体中文](https://github.com/kapitalismho/PuriPuly-heart/blob/dev/docs/readme/README.zh-CN.md#使用您自己的-api-密钥)
- [Русский](https://github.com/kapitalismho/PuriPuly-heart/blob/dev/docs/readme/README.ru.md#свои-api-ключи)


## Change logs
- Luna on the ChatGPT connection preconnects one standby socket while translation is off and reuses it when translation is enabled.

- Soniox speech recognition and API-key verification now authenticate when opening the WebSocket connection. Update before **January 15, 2027**, when Soniox will reject older builds that authenticate only in the first configuration message. Existing API keys continue to work; no key reissuance or settings change is required.
- Soniox's concurrent-connection limit now uses the existing bounded recovery policy.

---
**If PuriPuly has made your world a bit wider,
please consider hitting the ⭐ Star button at the top of the GitHub page.
It would mean a lot**
