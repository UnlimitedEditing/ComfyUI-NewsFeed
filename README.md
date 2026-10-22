# ComfyUI-NewsFeed

Nodes for a news-to-video pipeline on Graydient: `NewsPickStory` reads live RSS, picks a story by seed,
downloads the full article (text + captioned images) and builds an LLM brief; `NewsFinishScript` validates the
LLM's JSON and returns one result as a data image. Tone/visual presets live in `presets.json`, feeds in `feeds.json`.
Article images belong to their publishers.
