// Development-only stand-in for the WebLLM engine. main.js loads it only under
// `npm run dev` with `?mock-engine` in the URL (add `&cached` to simulate a model
// already saved in this browser). It replays WebLLM-shaped progress text, streams
// canned multi-line chat replies, and answers the feed roles with simple JSON or
// text so the UI can be checked without a GPU. Production builds never import it.

const sleep = (ms) => new Promise((resolve) => { setTimeout(resolve, ms); });

const CHAT_REPLIES = [
  "おー、はじめまして\nこれ全部ブラウザの中で動いてるやつ\n気になることあったら聞いて",
  "たしかに？\nそれはちょっと考えたことなかった",
  "そこは記録にないから分からない\n適当に答えるのはやめとく",
];

export async function createMockEngine(initProgressCallback, { cached = false } = {}) {
  const shards = 30;
  const totalMb = 2250;
  if (!cached) {
    for (let index = 1; index <= shards; index += 1) {
      await sleep(160);
      const fraction = index / shards;
      initProgressCallback({ progress: fraction, timeElapsed: index * 0.6, text: `Fetching param cache[${index}/${shards}]: ${Math.round(fraction * totalMb)}MB fetched. ${Math.floor(fraction * 100)}% completed, ${Math.round(index * 0.6)} secs elapsed. It can take a while when we first visit this page to populate the cache. Later refreshes will become faster.` });
    }
  }
  for (let index = 1; index <= shards; index += 1) {
    await sleep(30);
    const fraction = index / shards;
    initProgressCallback({ progress: fraction, text: `Loading model from cache[${index}/${shards}]: ${Math.round(fraction * totalMb)}MB loaded. ${Math.floor(fraction * 100)}% completed, 11 secs elapsed.` });
  }
  for (let index = 1; index <= 10; index += 1) {
    await sleep(60);
    initProgressCallback({ progress: index / 10, text: `Loading GPU shader modules[${index}/10]: ${index * 10}% completed, 12 secs elapsed.` });
  }
  initProgressCallback({ progress: 1, text: "Finish loading on Mock GPU (dev)" });

  let turn = 0;
  let interrupted = false;

  function roleReply(request) {
    const system = request.messages?.[0]?.content || "";
    const user = request.messages?.[1]?.content || "";
    const schema = String(request.response_format?.schema || "");
    if (schema.includes("verdict") || /verdict/.test(system)) return JSON.stringify({ verdict: "supported" });
    if (schema.includes("angle") || /angle/.test(system) && /抜粋/.test(user)) {
      const title = /記事: (.+)/.exec(user)?.[1] || "この記事";
      return JSON.stringify({ angle: "意外な使われ方", facts: [`${title}についての記述`] });
    }
    if (system.includes("元の呟き")) return "それな\nもうちょい調べてみる";
    const title = /題材: (.+)/.exec(user)?.[1] || "これ";
    return `${title}、思ってたより奥が深いっぽい`;
  }

  return {
    async resetChat() {},
    async interruptGenerate() { interrupted = true; },
    chat: {
      completions: {
        async create(request) {
          if (!request.stream) {
            await sleep(500);
            const content = roleReply(request);
            return { choices: [{ message: { content }, finish_reason: "stop" }] };
          }
          const reply = CHAT_REPLIES[turn % CHAT_REPLIES.length];
          turn += 1;
          interrupted = false;
          async function* stream() {
            await sleep(700);
            yield { choices: [{ delta: { content: "<think>\n\n</think>\n\n" } }] };
            for (const char of reply) {
              if (interrupted) return;
              await sleep(char === "\n" ? 380 : 55);
              yield { choices: [{ delta: { content: char } }] };
            }
          }
          return stream();
        },
      },
    },
  };
}
