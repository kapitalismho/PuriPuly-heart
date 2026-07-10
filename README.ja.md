<p align="center">
  <img src="src/puripuly_heart/data/icons/icon.png" alt="PuriPuly <3" width="128" />
</p>

<h1 align="center">PuriPuly <3</h1>

<p align="center">
  <img src="https://img.shields.io/badge/version-2.2.2-blue" alt="Version" />
  <img src="https://img.shields.io/badge/license-AGPL--3.0--or--later-blue" alt="License: AGPL-3.0-or-later" />
  <img src="https://img.shields.io/badge/python-3.12-yellow" alt="Python" />
  <img src="https://img.shields.io/badge/platform-Windows-lightgrey" alt="Platform" />
</p>

<p align="center">LLM-based two-way translator for VRChat</p>

<h2 align="center">
  <a href="README.md">🇺🇸 English</a> ·
  <a href="README.ar.md">🇸🇦 العربية</a> ·
  <a href="README.bg.md">🇧🇬 Български</a> ·
  <a href="README.ca.md">CA Català</a> ·
  <a href="README.cs.md">🇨🇿 Čeština</a> ·
  <a href="README.da.md">🇩🇰 Dansk</a> ·
  <a href="README.de.md">🇩🇪 Deutsch</a> ·
  <a href="README.el.md">🇬🇷 Ελληνικά</a> ·
  <a href="README.es.md">🇪🇸 Español</a> ·
  <a href="README.et.md">🇪🇪 Eesti</a> ·
  <a href="README.fi.md">🇫🇮 Suomi</a> ·
  <a href="README.fr.md">🇫🇷 Français</a> ·
  <a href="README.hi.md">🇮🇳 हिन्दी</a> ·
  <a href="README.hu.md">🇭🇺 Magyar</a> ·
  <a href="README.id.md">🇮🇩 Bahasa Indonesia</a> ·
  <a href="README.it.md">🇮🇹 Italiano</a> ·
  🇯🇵 日本語 ·
  <a href="README.ko.md">🇰🇷 한국어</a> ·
  <a href="README.lt.md">🇱🇹 Lietuvių</a> ·
  <a href="README.lv.md">🇱🇻 Latviešu</a> ·
  <a href="README.ms.md">🇲🇾 Bahasa Melayu</a> ·
  <a href="README.nl.md">🇳🇱 Nederlands</a> ·
  <a href="README.no.md">🇳🇴 Norsk</a> ·
  <a href="README.pl.md">🇵🇱 Polski</a> ·
  <a href="README.pt.md">🇵🇹 Português</a> ·
  <a href="README.ro.md">🇷🇴 Română</a> ·
  <a href="README.ru.md">🇷🇺 Русский</a> ·
  <a href="README.sk.md">🇸🇰 Slovenčina</a> ·
  <a href="README.sv.md">🇸🇪 Svenska</a> ·
  <a href="README.th.md">🇹🇭 ไทย</a> ·
  <a href="README.tr.md">🇹🇷 Türkçe</a> ·
  <a href="README.uk.md">🇺🇦 Українська</a> ·
  <a href="README.vi.md">🇻🇳 Tiếng Việt</a> ·
  <a href="README.zh-CN.md">🇨🇳 简体中文</a> ·
  <a href="README.zh-TW.md">🇹🇼 繁體中文</a>
</h2>

> ⚠️ **これはポータブルフォークです** [kapitalismho/PuriPuly-heart](https://github.com/kapitalismho/PuriPuly-heart)。簡単な配布と修正のために変更されています。[ポータブルビルドをダウンロード →](../../releases)

---

## Demo

![PuriPuly（Deepgram + Gemini 3 Flash）と VRCT（Google Web Speech + Google Translate）の翻訳比較。PuriPuly 音声認識：「진짜한개도없어서완전허탈했어.」、翻訳：「（本当に一つもなくて、すごくがっかりしちゃった。）」 | VRCT 音声認識：「진짜 한 개도 없어서 완전 허탈했다」、翻訳：「本当の犬もいませんでした。」](docs/images/demo/ko-jp_screenshot.png)

---

<video src="https://github.com/user-attachments/assets/c667f44d-b91d-42a9-b24a-e6a993b392d3" controls width="100%"></video>

PuriPulyを通じて他の外国人の友達と実際に交流している様子をもっと見たい方は：
- [デモ 1](https://www.youtube.com/watch?v=3p0CamYui0o)
- [デモ 2](https://youtu.be/DoX36Y7J_lc?si=YjbeVTS8v3jGQB1w)
- [デモ 3](https://www.youtube.com/watch?v=D0npvp68xNY)

---

## Finally, talk like real friends.

慰めたかったのに、  
「大丈夫？」としか声がかけられなかったこと、ありますよね。

伝えたい気持ちが、  
ただの「翻訳機」じゃ届かないこと、わかってますよね。

だから、作ったんです。

- **LLMベースのローカライズ** — スラング、口語、タメ口/敬語まで自然に
- **文脈の記憶** — 前後の流れを踏まえた自然な会話を維持
- **双方向の音声翻訳** — 相手の音声も一緒に翻訳、VR字幕オーバーレイ対応
- **Discordで始められる** — 複雑な設定なしですぐに使える

## Q&A

- **翻訳の品質はどのくらいですか？**
→ お互いにこの翻訳機を使えば、深い話までできるくらいです。定量的にはGemma 4でDeepLより6倍良い結果でした。詳しくは下の「翻訳比較」をご覧ください。

- **話してから翻訳されるまでどのくらいかかりますか？**
→ Gemma 4とクラウドSTTを使った場合、遅延は通常1秒中盤〜後半くらいです。

- **使うのにお金はかかりますか？**
→ はい、でも後からです。新規ユーザーには無料の使用枠が用意されています。それ以降もとても安く、1ドルで数千回使えます。

- **APIキーを発行する必要がありますか？**
→ はい、でもこれも後からです。最初はインストールしてDiscordで認証するだけで使えます。

- **相手の音声を翻訳する機能の完成度はどのくらいですか？**
→ 騒音の少ない1対1の環境で最もよく動作します。3人までなら問題ない場合もありますが、保証はできません。VRChatで使う場合は、Earmuff機能を使って環境をコントロールしてください。

- **音声認識がうまくいきません / 遅いです**
→ ローカルのQwen ASRを使っている場合は、クラウドSTTに切り替えるのをおすすめします。Intelユーザーの方は、PuriPulyをPコアのみに固定割り当てされるよう設定してください。

- **音声や会話の内容はどう扱われますか？**
→ 音声や会話の内容はローカルに保存され、Puripulyのサーバーには送信されません。また、相手の音声・文字起こし・翻訳結果は記録しません。ただし、STTサービスと翻訳プロバイダーがデータを処理することがあります。

### [📥 ダウンロード](https://github.com/kapitalismho/PuriPuly-heart/releases/latest)

---

## 翻訳比較
![翻訳品質ベンチマークチャート。韓国語から英語、日本語、中国語簡体字への216個のマルチターン会話サンプルをGemba MQMフレームワーク（判定モデル：Gemini 3.1 Pro Preview）で評価した1文あたりの平均エラーペナルティ（低いほど良）の結果を示す。各スコア：Gemini 3.1 Flash-lite 0.573、Gemini 3 Flash 0.596、Gemma 4 26B A4B 0.813、Qwen 3.5 Plus 0.958、DeepSeek V4 Flash 1.025、Gemma 4 26B A4B (no-context) 1.265、DeepSeek V4 Flash (no-context) 1.647、Qwen 3.5 Flash 2.198、DeepL 4.963、DeepL (no-context) 5.717、Google Translation Basic 5.998。](docs/images/performance/1.png)

- マイクロソフトのGemba MQMフレームワークを使って実験しました。
- 実際の会話に近づけるため、マルチターン環境で構成しました。
- 全体の実験結果は[こちら](https://github.com/kapitalismho/korean-llm-context-translation-benchmark)を参照してください。

## コスト

### 1ドルあたりの使用可能回数

#### 推奨モデル

| LLM \ ASR | Qwen ASR (Local) | Qwen ASR (Cloud) | Soniox | Deepgram |
|---|---|---|---|---|
| **Gemma 4 26B A4B** | 14,380回 | 2,920回 | 3,710回 | 1,180回 |
| **DeepSeek V4 Flash** | 19,410回 | 3,080回 | 3,980回 | 1,210回 |

#### その他のモデル

| LLM \ ASR | Qwen ASR (Local) | Qwen ASR (Cloud) | Soniox | Deepgram |
|---|---|---|---|---|
| **Gemma 4 31B (Cerebras)** | 920回 | 730回 | 770回 | 540回 |
| **DeepSeek V4 Pro** | 6,400回 | 2,330回 | 2,810回 | 1,070回 |
| **Gemini 3 Flash** | 1,710回 | 1,170回 | 1,280回 | 740回 |
| **Gemini 3.1 Flash-Lite** | 3,430回 | 1,770回 | 2,030回 | 940回 |
| **Qwen 3.5 Plus** | 7,460回 | 2,460回 | — | — |
| **Local LLMs** | 無制限 | 3,660回 | 5,000回 | 1,290回 |

### 発話あたりのコスト

#### 推奨モデル

| LLM \ ASR | Qwen ASR (Local) | Qwen ASR (Cloud) | Soniox | Deepgram |
|---|---|---|---|---|
| **Gemma 4 26B A4B** | ~0.01円 | ~0.05円 | ~0.04円 | ~0.13円 |
| **DeepSeek V4 Flash** | ~0.008円 | ~0.05円 | ~0.04円 | ~0.12円 |

#### その他のモデル

| LLM \ ASR | Qwen ASR (Local) | Qwen ASR (Cloud) | Soniox | Deepgram |
|---|---|---|---|---|
| **Gemma 4 31B (Cerebras)** | ~0.16円 | ~0.20円 | ~0.19円 | ~0.28円 |
| **DeepSeek V4 Pro** | ~0.02円 | ~0.06円 | ~0.05円 | ~0.14円 |
| **Gemini 3 Flash** | ~0.09円 | ~0.13円 | ~0.12円 | ~0.20円 |
| **Gemini 3.1 Flash-Lite** | ~0.04円 | ~0.08円 | ~0.07円 | ~0.16円 |
| **Qwen 3.5 Plus** | ~0.02円 | ~0.06円 | — | — |
| **Local LLMs** | 0円 | ~0.04円 | ~0.03円 | ~0.12円 |

*   *（入力 900トークン + 出力 12トークン）× 発話1回あたりの平均LLM呼び出し回数 1.2回と仮定*
*   *1ドルあたりの使用可能回数は、発話あたりのコスト表の四捨五入前の値を基準に算出*
*   *すべてのコストと使用可能回数は概算*
*   *DeepSeekはキャッシュヒット率70%を仮定しています*
*   *Qwen APIコストは北京リージョン基準*
*   *料金表基準: 2026年5月25日 / 高速応答モード有効時*
*   *1ドル = 150円*

### 無料クレジット

| サービス | 無料クレジット | 期限 | 備考 |
|--------|------------|------|------|
| **Deepgram** | $200 | なし | - |
| **Google AI Studio** | $10 | 1年 | Gemini加入者に毎月付与 |
| **Alibaba Cloud** | モデルごと100万トークン | 90日 | シンガポールリージョン基準 |
| **Alibaba Cloud** | ¥300 | 1年 | 中国国内の学生向け |
| **Cerebras** | 毎日100万トークン | なし | 1分あたり5回の呼び出し制限 |

---

# 問題が起きたり、分かりにくいところがあれば、気軽に[Twitter/X](https://x.com/kapitalismho)でDMしてください。

## 使い方

1. [ダウンロードページ](https://github.com/kapitalismho/PuriPuly-heart/releases/latest)から最新バージョンをダウンロード
2. PuriPulyをインストール
3. **STT** ボタンをクリック
4. **TRANS** ボタンをクリックしてDiscord認証
5. **Subtitles** ボタンを押してVR字幕をオン
6. （任意）**Peer** ボタンを押して相手の音声翻訳をオン

   > 相手の音声翻訳機能がきちんと動作するには、騒がしくない環境が必要です。VRChatで使う場合は、Earmuff機能を使って環境をコントロールしてください。

7. VRChatでOSCを有効化: Action menu → Settings → OSC → Enable

### 音声がキャプチャされない場合
音声がキャプチャされない場合は、**設定 > 一般** で次の手順を行ってください。

1. **オーディオホストAPI** を **自動選択** または **MME** に変更
2. 正しいマイクを選択
3. アプリを再起動

---

### 中国のユーザーへ向けた案内

Soniox / Gemini / Deepgramへのアクセスがブロックされている地域の場合は、以下の組み合わせをお使いください。

- STT: **Qwen ASR**
- LLM: **DeepSeek V4 Flash**

   > Discordの代わりにQQで認証できます。

---

### 自分のAPIキーを使う

利用するサービスに合わせて、適切なガイドを見ながら進めてください。

翻訳用LLMは、OpenRouter経由でGemma 4モデルを使うことをおすすめします。

もしよければ、設定するついでに、ASR側も一緒に設定しませんか？
PuriPulyはクラウドSTTと組み合わせると最良の体験になります。
たとえば同じQwen ASRでも、ローカルとクラウドでは音声認識性能にかなり差があります。

まずはDeepgramから始めるのをおすすめします。
登録するだけで200ドル分の無料クレジットがもらえます。

<details>
<summary><h3>OpenRouter</h3></summary>

1. 赤い丸の中のオプションをスクリーンショットのとおりに設定してください。
   ![step0](docs/images/openrouter/0.png)

2. アプリ内で赤い丸の中のボタンを押します。
   ![step1](docs/images/openrouter/1.png)

3. OpenRouterでログインします。
   ![step2](docs/images/openrouter/2.png)

4. 赤い丸の中のボタンを押して決済画面を抜けます。
   ![step3](docs/images/openrouter/3.png)

5. **Authorize** ボタンを押します。
   ![step4](docs/images/openrouter/4.png)

6. 使う分だけ前払いでチャージします。
   ![step5](docs/images/openrouter/5.png)

<details>
<summary><h3>Authorizeボタンを押しても認証されない場合</h3></summary>

Authorizeボタンを押しても認証されない場合は、再試行するか、以下の手順で直接APIキーを発行して貼り付けてください。

6. 右上のアカウントをクリックし、左のAPI Keysタブを開いて、中央のCreateボタンを押します。
   ![step6](docs/images/openrouter/6.png)

7. Createボタンを押します。
   ![step7](docs/images/openrouter/7.png)

8. ボタンを押してAPIキーをコピーし、翻訳機のAPIタブに貼り付けます。
   ![step8](docs/images/openrouter/8.png)

</details>

</details>

<details>
<summary><h3>DeepSeek</h3></summary>

1. 赤い丸の中のオプションをスクリーンショットのとおりに設定してください。
   ![step0](docs/images/deepseek/0.png)

2. [DeepSeek公式サイト](https://www.deepseek.com/en/)にアクセスし、**Access API** ボタンをクリックします。
   ![step1](docs/images/deepseek/1.png)

3. サイトでログインします。
   ![step2](docs/images/deepseek/2.png)

4. API Keysタブに移動して **Create new API Keys** を押します。
   ![step3](docs/images/deepseek/3.png)

5. ボタンを押してAPIキーをコピーし、翻訳機のAPIタブに貼り付けます。
   ![step4](docs/images/deepseek/4.png)

6. Top Upタブに移動し、使う分だけ前払いでチャージします。
   ![step5](docs/images/deepseek/5.png)

</details>

<details>
<summary><h3>Deepgram</h3></summary>

1. [Deepgram Console](https://console.deepgram.com/)にアクセスしてログインします。
   ![step1](docs/images/deepgram/1.png)

2. 歓迎メッセージとアンケートが表示されたら、**Skip** を押してスキップします。
   ![step2](docs/images/deepgram/2.png)

3. サービス選択画面で **STT (Speech-to-Text)** を選択します。
   ![step3](docs/images/deepgram/3.png)

4. API Keysメニューで **Create a New API Key** をクリックします。
   ![step4](docs/images/deepgram/4.png)

5. キーの名前を入力し（例：`puripuly`）、作成します。
   ![step5](docs/images/deepgram/5.png)

6. 作成されたキーをコピーして、PuriPulyの設定に貼り付けます。
   ![step6](docs/images/deepgram/6.png)

</details>

<details>
<summary><h3>Gemini</h3></summary>

1. [Google AI Studio](https://aistudio.google.com/apikey)にアクセスし、**Get API key** ボタンをクリックします。
   ![step1](docs/images/gemini/1.png)

2. 新しいプロジェクトを作成します。
   ![step2](docs/images/gemini/2.png)

3. 任意の名前を付けます。
   ![step3](docs/images/gemini/3.png)

4. 作成したプロジェクトを選択し、**Create key** を押します。
   ![step4](docs/images/gemini/4.png)

5. 丸で囲まれた部分を押します。
   ![step5](docs/images/gemini/5.png)

6. 丸で囲まれた部分を押してキーをコピーします。
   ![step6](docs/images/gemini/6.png)

7. （推奨）黄色で強調表示されている **Set Up Billing** ボタンを押し、有料プランに切り替えます。
プラン切り替えには少し時間がかかることがあります。
   ![step7](docs/images/gemini/7.png)

<details>
<summary><h3>Geminiの有料サブスクリプションをお持ちの方</h3></summary>

8. [Google Developer Program](https://developers.google.com/program/my-benefits) にアクセスし、プログラムに参加してください。
   ![step8](docs/images/gemini/8.png)

9. ステップ7で設定した有料プランのプロジェクトを選択してください。
   ![step9](docs/images/gemini/9.png)

</details>

</details>

<details>
<summary><h3>Qwen</h3></summary>

1. 地域に合った経路でAlibaba Cloud Model Studioにアクセスします。
   - [中国本土](https://bailian.console.aliyun.com/cn-beijing)
   - [中国本土以外の地域](https://bailian.console.alibabacloud.com)

2. アクセスしたアドレスからログインします。APIキーを発行したいリージョン（Region）を正確に選択してください（例：Beijing）。
   ![step2](docs/images/qwen/1.png)

3. 右上の **歯車アイコン** をクリックします。
   ![step3](docs/images/qwen/2.png)

4. ワークスペースを作成し、**API-KEY** ページに移動します。
   ![step4](docs/images/qwen/3.png)

5. **Create API Key** をクリックします。
   ![step5](docs/images/qwen/4.png)

6. アカウントとワークスペースを割り当てて、OKボタンを押します。
   ![step6](docs/images/qwen/5.png)

7. 丸で囲まれた部分を押してキーをコピーします。
   ![step7](docs/images/qwen/6.png)

</details>

<details>
<summary><h3>Soniox</h3></summary>

1. [Soniox Console](https://console.soniox.com/)にログインします。
   ![step1](docs/images/soniox/1.png)

2. 組織の名前を任意で入力します。
   ![step2](docs/images/soniox/2.png)

3. **Add Funds** ボタンを押し、支払い方法を登録します。
   ![step3](docs/images/soniox/3.png)

4. Sonioxはプリペイド方式のチャージが必要です。チャージ完了後、**API Keys** メニューへ移動します。
   ![step4](docs/images/soniox/4.png)

5. 新しいAPI Keyを作成します。
   ![step5](docs/images/soniox/5.png)

6. 作成されたキーをコピーして、PuriPulyの設定に貼り付けます。
   ![step6](docs/images/soniox/6.png)

</details>

<details>
<summary><h3>Cerebras</h3></summary>

1. [Cerebras](https://www.cerebras.ai/)にアクセスし、**Get started** ボタンを押してください。
   ![step1](docs/images/cerebras/1.png)

2. ログインしてください。
   ![step2](docs/images/cerebras/2.png)

3. 希望するプランを選択してください。最初は無料プランをおすすめします。
   ![step3](docs/images/cerebras/3.png)

4. APIキーをコピーしてPuriPulyに貼り付けてください。
   ![step4](docs/images/cerebras/4.png)

<details>
<summary><h3>有料プランに切り替えるには</h3></summary>

5. **Billing** タブに移動してください。
   ![step5](docs/images/cerebras/5.png)

6. 自分の名前を入力してください。
   ![step6](docs/images/cerebras/6.png)

7. 必要な分だけクレジットをチャージしてください。
   ![step7](docs/images/cerebras/7.png)

</details>

</details>

---

## 開発

### 開発環境のまとめ

| 領域 | 推奨環境 |
|---|---|
| Python アプリ | Windows |
| VR オーバーレイ | Windows |
| Broker サービス | Linux / WSL |

### Python アプリ

```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
```

```bash
# pip
pip install -e '.[dev]'

# または uv
uv sync --dev
```

```bash
pre-commit install
```

### GUI 実行

```bash
# 仮想環境を有効化した後
python -m puripuly_heart.main run-gui

# または uv で実行
uv run python -m puripuly_heart.main run-gui
```

```bash
# 隠れたUIを確認可能
python -m puripuly_heart.main run-gui --debug-ui-preview
```

### テストとリント

```bash
black src tests          # フォーマット
ruff check src tests     # リント
python -m pytest         # テスト（仮想環境での実行を推奨）
```

### VR オーバーレイ

VR字幕オーバーレイは `native/overlay/` のRustプロジェクトでビルドします。

```powershell
cargo test --manifest-path native/overlay/Cargo.toml -q

cargo build `
  --manifest-path native/overlay/Cargo.toml `
  --locked `
  --release `
  --bin PuriPulyHeartOverlay `
  --target-dir target

New-Item -ItemType Directory -Force -Path build/overlay | Out-Null
Copy-Item target/release/PuriPulyHeartOverlay.exe build/overlay/PuriPulyHeartOverlay.exe -Force
Copy-Item third_party/openvr/win64/openvr_api.dll build/overlay/openvr_api.dll -Force

.\build\overlay\PuriPulyHeartOverlay.exe --check-startup-contract
```

### Broker サービス

詳しくは `broker/README.md` を参照してください。

```bash
pnpm install --frozen-lockfile
pnpm run typecheck
pnpm exec vitest run
pnpm --filter @puripuly-heart/broker run verify:config
pnpm --filter @puripuly-heart/broker run dev
```

---

## 開発者

[salee](https://github.com/kapitalismho)

---

## コントリビューター

[RICHARDwuxiaofei](https://github.com/RICHARDwuxiaofei)

---

## Special Thanks

SUI\_32C, Nagikokoro, motoka96, \_Ykol魚, kascr\_, Just Monika V, FLUVIA, Han โชเล่ย์, EA\_PE, Ephedrine, ~ eri ~

---

## ライセンス

[AGPL-3.0-or-later](LICENSE)

サードパーティライセンスおよび通知: `src/puripuly_heart/data/THIRD_PARTY_NOTICES.txt`
