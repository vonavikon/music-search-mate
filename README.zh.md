# Music Search Mate

<p align="center"><img src="assets/logo.png" width="150" alt="Music Search Mate logo"></p>

[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![tests](https://github.com/vonavikon/music-search-mate/actions/workflows/tests.yml/badge.svg)](https://github.com/vonavikon/music-search-mate/actions/workflows/tests.yml)
[![VirusTotal](https://img.shields.io/badge/VirusTotal-0%2F61_clean-brightgreen)](https://www.virustotal.com/gui/file/61c789f0735b0a6d59184576371bf35c2008aef816a0775b528ef53e3ed7459f)

[Русский](README.ru.md) · [English](README.md) · **中文** · [Español](README.es.md) · [Français](README.fr.md)

一个能根据自由文字描述找到音乐的 Telegram 机器人："AC/DC，但要更现代一点的，用德语唱的"、"出租车追奔驰那场追车戏里的那首歌"、"和 Moody Blues 的 Nights in White Satin 相似的"。它还能按链接下载音频和视频，通过自托管的本地 Bot API 服务器，单文件最大 2 GB。目前有一个在线实例在运行：[@music_search_mate_bot](https://t.me/music_search_mate_bot)。

<p align="center"><img src="assets/demo.gif" alt="演示：按描述搜索，返回带封面和按钮的曲目卡片" width="420"></p>
完整画质视频：[demo.mp4](assets/demo.mp4)

## 工作原理

流水线示意图如下（保留英文原文，以免中文的字符宽度破坏箭头对齐）：

```text
[user description]      free-form text, any language
        ↓
[LLM #1: extract]       intent (this_track / similar / from_source) + search queries;
                        destructive requests (drugs, weapons, violence, self-harm,
                        extremism) are cut here and by a regex prefilter before the LLM
        ↓
[youtube search]        yt-dlp runs the queries in parallel, ~20 raw candidates
        ↓
[filter + enrich]       dedup and cleanup; Last.fm tags, iTunes metadata and album artwork
        ↓
[LLM #2: rank]          picks the best candidate, sets confidence; not confident enough
                        → asks a clarifying question instead of downloading
        ↓
[track card]            top-1 with artwork and artist bio, "hit / miss" buttons
        ↓
[download]              audio as a tagged file with embedded artwork; video in background
```

- **相似搜索。** 参考曲目先通过 Last.fm 的 `track.search` 解析出规范艺术家，再把相邻曲目（`getSimilar`）注入搜索查询，而参考曲目本身会被禁止出现在结果里。
- **反馈循环。** 点一下"未命中"会记住被否决的候选并给出下一个；后续搜索会排除你已经拒绝过的内容。
- **经得起花招的审核。** 请求文本在任何检查之前先做规范化：NFKC，加上拉丁与西里尔同形字符的双向折叠，所以混淆拼写和提示词注入都钻不过空子。

## 功能

- **按描述搜索：** 一首具体的歌、"类似 X 的东西"、或者"电影/游戏/广告里的那首歌"；由 LLM 判断意图，并把查询改写成适合搜索的形式。
- **按链接下载：** 从白名单内的主机（YouTube、SoundCloud、Bandcamp、VK）下载音频或视频（之所以用白名单，是因为放开任意 URL 会把这个容器变成一个 SSRF 代理）。视频画质通过按钮选择，从 480p 到 1440p，经由一条从原生 mp4 到合并回退的编解码器级联。
- **2 GB 文件：** 上传走自托管的本地 Bot API 服务器；没有它，标准的 Telegram 云端 API 会把文件限制在 50 MB。
- **带标签的音频：** 以文件形式发送，写入规范的艺术家/标题/专辑标签和内嵌专辑封面（有 iTunes 封面就用，否则以视频缩略图兜底）。下载路径优先选择 m4a 里的原生 AAC，尽可能不重编码直接 remux，实在不行回退到 mp3 320k。
- **曲目卡片：** 封面、规范艺术家、简短简介、"命中 / 未命中"按钮。
- **曲目小知识：** 💡 按钮通过 Perplexity sonar 获取一条有网络依据的曲目小知识，缓存 30 天；重复点击不再计入限额。
- **缓存：** Telegram file_id（之前发过的曲目直接重发、无需重新下载）、小知识缓存、相似曲目池、带 TTL 的 Last.fm 查询。
- **群组：** 机器人只响应 @提及（关闭隐私模式，外加代码里的一道显式门控）；每用户限额处处生效。
- **本地化：** 错误、状态、按钮以及 LLM 提出的澄清式问题都有 RU/EN 两种语言；用 /lang 切换。
- **限额按次数计、不按流量：** 每用户每小时 20 次请求，在线实例每天 100 次搜索（代码默认 200，由环境变量设定）和每天 30 个视频，每天 2 条小知识（缓存命中不计入），另有面向所有用户的全局每日预算。
- **诚实的进度：** 百分比进度条（大约每 3% / 1.2 秒更新一次），外加一个单独的"正在准备文件"阶段，覆盖文件转码、写入标签和上传的过程。
- **可观测性：** 用 SQLite 记录请求结果的漏斗（ok / clarify / empty / blocked / limit / error），以及可选的、覆盖每个流水线阶段的 Langfuse 追踪；没有密钥时追踪保持关闭。

## 从源码安装

你需要 Python 3.12+ 和一个从 @BotFather 领取的机器人 token。本项目没有发布为软件包，只能从源码运行。所有 LLM 调用都走你自己的 OpenAI 兼容网关、用你自己的密钥；可选层（Last.fm、Langfuse）用的也是你自己的密钥。

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.example .env
# edit .env, then start (the same command the Docker image runs):
.venv/bin/python -m bot.main
```

`.env` 里的密钥：

- `BOT_TOKEN`：来自 @BotFather 的 token。必填。
- `BOTHUB_API_KEY` + `BOTHUB_BASE`：任意 OpenAI 兼容网关。一个模型负责提取与排序（`LLM_MODEL`），另一个负责有网络依据的小知识（`LLM_MODEL_SONAR`，留空则禁用小知识层）。
- `LASTFM_API_KEY`：可选；没有它，Last.fm 层会被静默跳过。
- `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` / `LANGFUSE_HOST`：可选追踪；没有密钥时追踪关闭。
- `TG_API_ID` / `TG_API_HASH` / `LOCAL_BOT_API_URL`：可选的本地 Bot API 服务器，用于最大 2 GB 的文件（你在 my.telegram.org 申请的自有应用凭据）。没有它就适用 50 MB 的云端限制。首次把机器人切到本地服务器时，需要对 api.telegram.org 调用一次 `logOut`，否则更新会一直发往云端。

测试：301 个测试全部离线跑在 mock 上，不需要网络。

```bash
.venv/bin/python -m pytest -q
```

另有一个走真实网络的冒烟关卡（`make smoke`，约 2 分钟），会请求真实的网关和 yt-dlp；日常开发跑离线套件就够了。

### Docker

```bash
docker compose -f docker-compose.public.yml up -d
```

镜像从本仓库构建：Python 3.12-slim、ffmpeg、deno（yt-dlp 应对 YouTube 的 n-challenge 所需的 JS 运行时），以及在锁定的稳定版之上改用的 nightly 版 yt-dlp。

## 架构

```text
bot/
  main.py                     entrypoint (python -m bot.main): handlers, rate limiter,
                              local Bot API wiring, error handler
  handler.py                  message flow: search pipeline, URL fast path,
                              feedback / similar / clarify / download callbacks
services/
  llm_search.py               LLM #1 (extract) and LLM #2 (rank), moderation block,
                              similar-reference resolve via Last.fm
  search.py                   yt-dlp metadata search without downloading
                              (YouTube first, SoundCloud as fallback)
  media.py                    audio/video download via yt-dlp subprocess, live progress,
                              temp TTL cleanup, circuit breaker
  metadata.py                 metadata cascade: yt-dlp info dict → Last.fm → iTunes artwork
  tag_enrichment.py           writes canonical tags and artwork into the file (ffmpeg)
  match_score.py              deterministic rule-based match score that guards auto-download
  store.py                    SQLite (aiosqlite): queries, feedback, usage limits, caches
  observability.py            circuit breaker for upstreams, PII redaction in logs
  tracing.py                  optional Langfuse tracing, one trace per search
config.py                     env config: models, limits, allowlists, retention
tests/                        301 tests, offline mocks, pytest
```

## 状态

一个单人维护的业余项目。版本 1.17（2026 年 11 月）；公开实例运行的就是这份代码。

项目是从真实使用中长出来的，不是按路线图规划出来的：变速（放慢/加快）上传的版本曾不断弄坏标签和封面，直到候选过滤器学会把它们丢弃；群组模式是在真实群聊的抱怨声中重写的；本地 Bot API 服务器则是在 50 MB 云端上限成为真正的瓶颈时才出现的。

坦率地讲，它有这些局限：搜索质量取决于 LLM 有多懂你，它没把握时会反过来问你、而不是瞎猜；yt-dlp 依赖 YouTube 的内部实现，YouTube 一有改动，下载就可能出问题，要等 nightly 构建跟上才恢复；你需要自备 LLM 网关的 API 密钥（想要元数据层的话，Last.fm 的也要）；而修复的节奏，就是一个人能有的节奏。

## 许可证

MIT，详见 [LICENSE](LICENSE)。
