// Development-only stand-in for the WebLLM engine, loaded only by `npm run dev`
// with `?mock-engine` in the URL. It replays WebLLM-shaped progress messages and
// streams a canned multi-line reply so the loading, reading, typing, and memory
// states can be checked without using the GPU. It never runs in production builds.

const sleep = (ms) => new Promise((resolve) => { setTimeout(resolve, ms); });

const CANNED = [
  "それな\nWebGPUで動いてるから\n全部ブラウザの中で完結してる",
  "たしかに？\nちょっと考えてみる",
  "記録にはないから\nそこは分からない",
];

export async function createMockEngine(initProgressCallback) {
  const shards = 24;
  for (let index = 1; index <= shards; index += 1) {
    await sleep(90);
    initProgressCallback({ progress: index / shards, text: `Fetching param cache[${index}/${shards}]: ${index * 92}MB fetched. ${Math.floor((index / shards) * 100)}% completed, ${Math.round(index * 0.4)} secs elapsed. It can take a while when we first visit this page to populate the cache. Later refreshes will become faster.` });
  }
  for (let index = 1; index <= shards; index += 1) {
    await sleep(25);
    initProgressCallback({ progress: index / shards, text: `Loading model from cache[${index}/${shards}]: ${index * 92}MB loaded. ${Math.floor((index / shards) * 100)}% completed, 11 secs elapsed.` });
  }
  for (let index = 1; index <= 10; index += 1) {
    await sleep(40);
    initProgressCallback({ progress: index / 10, text: `Loading GPU shader modules[${index}/10]: ${index * 10}% completed, 12 secs elapsed.` });
  }
  initProgressCallback({ progress: 1, text: "Finish loading on Mock GPU (dev)" });

  let turn = 0;
  return {
    async resetChat() {},
    async interruptGenerate() {},
    chat: {
      completions: {
        async create() {
          const reply = CANNED[turn % CANNED.length];
          turn += 1;
          async function* stream() {
            await sleep(900);
            yield { choices: [{ delta: { content: "<think>\n\n</think>\n\n" } }] };
            for (const char of reply) {
              await sleep(char === "\n" ? 260 : 70);
              yield { choices: [{ delta: { content: char } }] };
            }
          }
          return stream();
        },
      },
    },
  };
}
