import { GUESS, QUESTION } from "./reply_decision.js";
import { HARSH } from "./safety.js";
import { chars, pick, rand, shuffle, sleep } from "./util.js";

// Mock inference: canned lines with the same complete(task, input) as
// the real model. It ignores PROMPTS.

export function createMockLLM() {
  // Generic voice lines only: thoughts, reactions, and moods, with no
  // claims about kobajin's real life, plans, or preferences.
  const POSTS = {
    "疑問": ["なんで月曜って存在するん", "結局どっちが正解なん", "これ誰が決めたん", "冷静に考えて謎", "なんでこうなった？"],
    "ツッコミ": ["いやそうはならんやろ", "情報量が多い", "それはそう", "たしかに？", "急に正論やめて"],
    "感心": ["天才の発想すぎる", "仕組み考えた人えぐい", "よくできてるなこれ", "発想が強い", "これ考えた人すごい"],
    "テンション": ["今日いける気がする", "これは勝ち", "完全に理解した", "優勝", "流れきてる"],
    "作業": ["あと一歩が遠い", "動いたけど理由不明", "バグが仕様になった", "一旦保存", "進捗えぐい"],
    "考えごと": ["言語化むずい", "わかるようでわからん", "考えすぎたかもしれん", "シンプルが一番", "結局タイミング"],
    "天気・季節": ["寒すぎてえぐい", "急に夏", "空が広い", "季節の変わり目すぎる", "風つよ"],
    "やりたいこと": ["なんか作りたい", "どっか行きたい", "新しいこと始めたい", "全部やりたい", "一回ちゃんとやる"],
    "眠い": ["もう寝たい", "ねむすぎて終わった"],
  };
  // What each post line meant, for 「具体的に」「詳しく」. They spell out
  // the thought itself; plans and tastes stay undecided.
  const ELABORATIONS = {
    "なんで月曜って存在するん": ["土日のあとに急に平日来るのがおかしい", "休みとの落差がでかすぎるって話"],
    "結局どっちが正解なん": ["どっちにも言い分あって決め手がない", "両方よさそうに見えて選べんやつ"],
    "これ誰が決めたん": ["気づいたら当たり前になってるルールのこと", "最初に決めた人の理由が知りたい"],
    "冷静に考えて謎": ["当たり前のことほど理由聞かれると答えられん", "普通に受け入れてたけど意味わからんくなった"],
    "なんでこうなった？": ["やったことと結果がつながってない", "途中まで合ってたはずなのにって感じ"],
    "いやそうはならんやろ": ["前提から結論まで飛びすぎてる", "途中の説明が一個もないって話"],
    "情報量が多い": ["一画面に言いたいことが詰まりすぎ", "一個ずつならわかるけど同時はむり"],
    "それはそう": ["反論のしようがない正論って意味", "言われてみれば当たり前すぎるやつ"],
    "たしかに？": ["納得しかけたけどまだ半信半疑", "一瞬うなずいたけど理由は説明できん"],
    "急に正論やめて": ["ふざけてたのに真面目に返されたとき", "正しいけど今それ言うんかって感じ"],
    "天才の発想すぎる": ["思いつきそうで誰も思いつかんやつ", "言われたら簡単なのに自分では出てこない"],
    "仕組み考えた人えぐい": ["使う側が何も考えんでいいのがすごい", "裏で全部うまく回ってるのがえぐい"],
    "よくできてるなこれ": ["細かいとこまで迷わんようになってる", "無駄な手順がひとつもない"],
    "発想が強い": ["普通と逆から考えてるのがいい", "前提から疑ってるのが強い"],
    "これ考えた人すごい": ["当たり前になってるけど最初はなかった", "みんな使ってるのに考えた人は見えへん"],
    "今日いける気がする": ["根拠はないけど手が止まらん感じ", "やること全部うまく転がりそうな日"],
    "これは勝ち": ["思った通りに一発で決まったときのやつ", "迷わず進めた時点で勝ち"],
    "完全に理解した": ["わかった気がしてるだけの段階", "説明はできんけど感覚はつかんだ"],
    "優勝": ["今日いちばんうれしいってこと", "全部うまくいったときの一言"],
    "流れきてる": ["小さいことが続けてうまくいってる", "今なら何やっても進みそう"],
    "あと一歩が遠い": ["九割できてから最後が終わらん", "残りちょっとが一番時間かかる"],
    "動いたけど理由不明": ["直したつもりないとこで直ってる", "触ってないのに動くのが一番こわい"],
    "バグが仕様になった": ["変な動きやけどそれはそれで使える", "直すより残したほうがおもろいやつ"],
    "一旦保存": ["キリはよくないけど消えるよりまし", "途中でも残しとくのが大事"],
    "進捗えぐい": ["思ったより一気に進んだ", "手を動かしたぶんだけ形になった"],
    "言語化むずい": ["頭ではわかってるのに言葉にならん", "感覚はあるのに説明すると別物になる"],
    "わかるようでわからん": ["説明は追えるのに自分で再現できん", "うなずけるけど人に説明できん"],
    "考えすぎたかもしれん": ["簡単な話を自分でややこしくしてた", "最初の案で良かったかもしれん"],
    "シンプルが一番": ["足すより削るほうが大事って話", "説明いらんくらいがちょうどいい", "機能は少ないほうが迷わん"],
    "結局タイミング": ["同じことでも言う時期で通り方が変わる", "早すぎても遅すぎてもあかん"],
    "寒すぎてえぐい": ["手が冷えて何もできん", "外出た瞬間に後悔する寒さ"],
    "急に夏": ["昨日まで上着いったのに今日は暑い", "季節の切り替わりが雑すぎる"],
    "空が広い": ["雲がなくて遠くまで見える", "見上げたら思ったより空がでかかった"],
    "季節の変わり目すぎる": ["朝寒くて昼暑い", "何着たらいいかわからん"],
    "風つよ": ["前に進まれへんくらい吹いてる", "髪がずっと顔にくる"],
    "なんか作りたい": ["何を作るかはまだ決めてない、気持ちだけある", "作りたい気持ちだけ先にある"],
    "どっか行きたい": ["行き先はまだ決めてない", "場所より出かけたい気分の話"],
    "新しいこと始めたい": ["何を始めるかはまだ決めてない", "いつもと違うことをしたい気分"],
    "全部やりたい": ["やりたいことが多すぎて絞れん", "一個に決められへんって話"],
    "一回ちゃんとやる": ["中途半端で止めずに最後までやる", "雑に済ませてたのを丁寧にやり直す"],
    "もう寝たい": ["まぶたが限界", "画面の文字が頭に入ってこん"],
    "ねむすぎて終わった": ["何してても意識が飛びそう", "考えがまとまらん"],
  };
  // A line the mock did not write itself (a reply such as 「ラーメンいいな」)
  // has no stored meaning, so it says what it picked up on.
  function elaborations(parent) {
    // A voiced line (なんで月曜って存在するん（たぶん）) still has its meaning.
    const known = ELABORATIONS[parent] || ELABORATIONS[Object.keys(ELABORATIONS).find((line) => parent.includes(line))];
    if (known) return known;
    const word = keyword(parent);
    return word
      ? [`${word}ってとこに反応しただけ`, `${word}の話が気になったって意味`]
      : ["そこまで深い意味はない", "ノリで言ったからこれ以上はない"];
  }
  // Replies follow the post: its own words first, then its tone.
  // "harsh" comes first: insults and threats get pushback, never an
  // echo or agreement. The lookaheads keep 少しねむい and カスタム out.
  const TONES = [
    ["harsh", HARSH],
    ["question", /[?？]\s*$|なんで|どう|なに|何|どっち/],
    ["funny", /草|笑|w\s*$|ｗ|ウケ|おもろ|じわ/],
    ["rough", /疲|眠|ねむ|寝|無理|つら|しんど|過密|忙|だる|痛|寒|暑|やば|終わ|詰|きつ|多すぎ|足りな|ない$/],
    ["happy", /できた|完成|勝|最高|好き|楽し|うま|嬉|うれし|神|優勝|天才|いいこと|やった|かわい|すごい|きれい/],
  ];
  const TONE_REPLIES = {
    harsh: ["落ち着け", "物騒すぎる", "どうした急に", "言い方よ", "こわ", "治安わるい", "それは言いすぎ", "なにがあった", "一回寝よ", "深呼吸しよ", "荒れてるな", "強火すぎる", "穏やかにいこ", "ストレスたまってる？"],
    question: ["たしかに？", "わからん", "むずい", "どうなんやろ", "気になる", "それはそう"],
    funny: ["草", "おもろ", "それは草", "じわる", "笑った"],
    rough: ["それはえぐい", "おつかれ", "それはきつい", "わかる", "しんどいな", "えぐいって"],
    happy: ["天才か", "いいね", "強い", "最高やん", "えらい", "優勝"],
    neutral: ["たしかに？", "それはそう", "わかる", "なるほど", "それな", "ほんまそれ"],
    // A shout, catchphrase, or anything that is not an opinion: agreeing
    // makes no sense, so be curious instead.
    odd: ["なにそれ", "どうした急に", "急に？", "唐突すぎる", "テンション高", "元気やな", "詳しく", "ほう"],
  };
  // Agreement only fits a statement, which has hiragana in it. A post
  // with none (ブンブン, 全力ダッシュ!!) gets the curious lines.
  const replyTone = (text) => {
    const tone = toneOf(text);
    if (tone !== "neutral") return tone;
    return /[ぁ-ん]/.test(text.replace(/@[A-Za-z0-9_]+/g, "")) ? "neutral" : "odd";
  };
  const toneOf = (text) => (TONES.find(([, pattern]) => pattern.test(text)) || ["neutral"])[0];
  const TONE_EMOJI = {
    harsh: ["😨", "😨", "👀", "🫠"],
    funny: ["😂", "😂", "🤔"],
    rough: ["🫠", "😭", "🫠", "🤔"],
    happy: ["🔥", "💯", "👍"],
    question: ["🤔", "🤔", "👀"],
    neutral: ["🤔", "👀", "🤔"],
  };

  const GREETINGS = [
    [/おはよ/, ["おはよう", "おは", "おはよー", "はやいな"]],
    [/おやすみ/, ["おやすみ", "おやすみー", "いい夢を", "ねよねよ"]],
    [/こんにちは|こんちゃ/, ["こんにちは", "こんちゃ", "やっほー"]],
    [/こんばんは/, ["こんばんは", "ばんは", "やっほー"]],
    [/ただいま/, ["おかえり", "おかえりー", "おつかれ"]],
    [/ハロー|hello|やっほ|ヤッホー|おっす|よっす|^やあ/i, ["ハロー", "やっほー", "よっ", "元気やな", "おっす"]],
  ];

  // When someone answers this account's own line, it talks back: a
  // question gets an answer (by its question word), cheers get thanks,
  // and agreement gets agreement, never another 「たしかに？」. Answers
  // stay vague, since the mock knows nothing about kobajin's plans.
  const ANSWERS = [
    [/どうした|どしたん|大丈夫/, ["なんでもない", "いや別に", "ちょっと思っただけ", "大丈夫"]],
    [/なんで(?!す)|なぜ|何で|どうして/, ["なんとなく", "そういう気分", "流れで", "特に理由はない", "気分の問題"]],
    [/いつ(?!も)/, ["そのうち", "近いうち", "気が向いたら", "今から"]],
    [/どこ/, ["どっか", "まだ決めてない", "ここではない"]],
    [/だれ|誰/, ["自分", "みんな", "誰やろな"]],
    [/どっち|どちら/, ["両方", "どっちも", "選べん"]],
    [/どれ/, ["全部", "選べん"]],
    [/どう(いう|ゆう)|どゆ|意味|つまり/, ["そのまんま", "言葉どおり", "ノリで言った", "説明むずい"]],
    [/どう|どんな/, ["いい感じ", "まあまあ", "ぼちぼち", "そこそこ"]],
    [/なに|何|なん[?？]|なんな|なんです|なんや(?!ろ)/, ["いろいろ", "全部", "まだ決めてない", "それを今考えてる", "頭の中のやつ", "うまく言えんやつ"]],
    // A guess (sb3ファイルとか？) gets a yes, not a question back. It
    // comes after the question words, so 何を〜ですか is a what-question.
    [GUESS, ["そうそう", "それそれ", "それもある", "まあそんな感じ", "そういうこと"]],
    [/ほんま|まじ|マジ|本当|ほんと/, ["ほんま", "まじ", "たぶん"]],
    [/./, ["うん", "まあね", "そんな感じ", "たぶんな"]],
  ];
  const BACK_TALK = [
    [/がんば|頑張|ファイト|応援/, ["ありがと", "がんばる", "がんばるわ"]],
    [/えらい|すごい|いいね|最高|天才|強い|さすが/, ["ありがと", "うれしい", "せやろ", "照れる"]],
    [/わかる|それな|たしかに|それはそう|ほんまそれ|同じ|せやな/, ["やろ", "せやろ", "わかってくれるか", "それな"]],
  ];
  function answer(text) {
    if (toneOf(text) === "harsh") return null;
    if (QUESTION.test(text)) {
      const [pattern, lines] = ANSWERS.find(([p]) => p.test(text));
      // A guess names the thing, so the answer can name it back.
      const word = keyword(text);
      const named = pattern === GUESS && word ? [`そう、${word}`, `${word}もある`].filter((line) => chars(line).length <= 10) : [];
      return [...lines, ...named];
    }
    const back = BACK_TALK.find(([pattern]) => pattern.test(text));
    return back ? back[1] : null;
  }

  const NOUN_ECHOES = {
    question: (w) => [`${w}？`, `${w}かぁ`, `${w}むずい`],
    funny: (w) => [`${w}で草`, `${w}おもろ`],
    rough: (w) => [`${w}えぐい`, `${w}おつかれ`, `${w}はきつい`, `${w}しんどいな`],
    happy: (w) => [`${w}いいな`, `${w}最高`, `${w}強い`],
    neutral: (w) => [`${w}いいな`, `${w}気になる`, `${w}かぁ`, `${w}わかる`],
    odd: (w) => [`${w}って何`, `急に${w}`, `${w}？`],
  };

  function echoes(word, tone) {
    if (!word || tone === "harsh") return [];
    const adjective = /[ぁ-ん]い$/.test(word) || /^[一-龠々]+い$/.test(word);
    const lines = adjective && tone !== "question"
      ? [`${word}よな`, `まじ${word}`, `${word}のわかる`, `わかる、${word}`]
      : NOUN_ECHOES[tone](word);
    return lines.filter((line) => chars(line).length <= 10);
  }
  // Each alt account's style (accounts.js `voice`), bent onto a canned
  // line now and then. Questions stay as they are (なんで月曜って存在
  // するん、えらい reads wrong), and so does a bent line that would run long.
  const VOICES = {
    calm: [(t) => t.replace(/[！!]+$/, ""), (t) => (/[なねよわ]$/.test(t) ? null : `${t.replace(/[。！!]+$/, "")}な`)],
    off: [(t) => `${t}（たぶん）`, (t) => t.replace(/えぐい/, "えぐみ深い").replace(/わかる/, "わかりみ"), (t) => `${t}、知らんけど`],
    hype: [(t) => `${t.replace(/[。]+$/, "")}！`, (t) => `${t.replace(/[。]+$/, "")}！！`, (t) => `え、${t}！`],
    sleepy: [(t) => `${t}〜`, (t) => `${t}…`, (t) => `${t}ねむ`],
    low: [(t) => `正直${t}`, (t) => `${t}…`, (t) => `まあ${t}`],
    kind: [(t) => `${t}、えらい`, (t) => `${t}！いいね`, (t) => (/[なねよわ]$/.test(t) ? null : `${t}な`)],
  };
  const QUESTIONISH = /[?？]|なんで|誰|どっち|どう|謎/;
  function voiced(line, voice, max) {
    const bends = VOICES[voice];
    if (!bends || QUESTIONISH.test(line) || Math.random() < 0.4) return line;
    const bent = pick(bends)(line);
    return bent && chars(bent).length <= max ? bent : line;
  }

  const TRENDS = [
    "#それはそう", "#たしかに？", "#言語化むずい", "#完全に理解した", "#天才の発想", "#情報量が多い",
    "#なんで月曜", "#あと一歩", "#優勝", "#もう寝たい", "#動いたけど謎", "#シンプルが一番",
    "#いやそうはならん", "#なんか作りたい", "#季節の変わり目", "#流れきてる",
  ];
  const used = [];

  function keyword(text) {
    // Trailing ー after katakana belongs to the word (ハロー), so it stays.
    const bare = text.replace(/@[A-Za-z0-9_]+/g, "")
      .replace(/[\s?？!！。、.,…〜~wｗ|｜¥\\]+$/g, "").replace(/(?<![ァ-ヴ])ー+$/, "")
      .replace(/^[\s#＃]+/, "");
    // A short post is its own keyword when it is one word: no kana
    // (過密, ラーメン) or a lone adjective (ねむい, 眠い).
    const oneWord = !/[ぁ-ん]/.test(bare) || (/^[一-龠々ぁ-ん]+い$/.test(bare) && !/[がをにはでとも]/.test(bare.slice(1, -1)));
    if (bare && chars(bare).length <= 4 && oneWord) return bare;
    // Whole runs only: a cut word (ブンブンハロ) reads wrong. Long runs
    // drop out later at the length check.
    // Katakana and English words before kanji, and time words (今日,
    // 最近) are never the topic: 今日ラーメン食べた is about ラーメン.
    const kanji = (bare.match(/[一-龠々]{2,4}/g) || []).filter((word) => !/^(今日|昨日|明日|今週|来週|先週|今年|最近|今朝|今夜|毎日|全員|自分|本当|普通)$/.test(word));
    return (bare.match(/[ァ-ヴ][ァ-ヴー]+/) || bare.match(/[A-Za-z]{2,}/) || kanji)[0] || "";
  }

  // Replies follow the post: greetings, answers when addressed, its own
  // words, then its tone.
  function mockReply(input) {
    const text = (input.text || "").trim();
    const others = input.others || [];
    const fresh = (list) => list.filter((line) => !others.includes(line));
    const greeting = GREETINGS.find(([pattern]) => pattern.test(text));
    if (greeting && fresh(greeting[1]).length) return pick(fresh(greeting[1]));
    const answers = input.addressed && answer(text);
    if (answers) return pick(fresh(answers).length ? fresh(answers) : answers);
    const tone = replyTone(text);
    const echoable = tone !== "question" || chars(text) <= 7;
    const echo = echoable ? fresh(echoes(keyword(text), tone)) : [];
    // The first reply picks up the post's own words; later ones
    // mostly react to its tone.
    const useEcho = echo.length && (others.length === 0 ? Math.random() < 0.85 : Math.random() < 0.3);
    const pool = useEcho ? echo : fresh(TONE_REPLIES[tone]);
    return pick(pool.length ? pool : TONE_REPLIES[tone]);
  }

  return {
    name: "mock",
    async complete(task, input) {
      await sleep(task === "judge" ? rand(400, 900) : rand(500, 1400));
      if (task === "post") {
        const lines = POSTS[input.kind] || Object.values(POSTS).flat();
        const fresh = lines.filter((line) => !used.includes(line));
        const line = pick(fresh.length ? fresh : lines);
        used.push(line);
        if (used.length > 20) used.shift();
        return voiced(line, input.voice, 22);
      }
      if (task === "reply") return voiced(mockReply(input), input.voice, 10);
      if (task === "elaborate") {
        const lines = elaborations(input.parent || "");
        const fresh = lines.filter((line) => !(input.others || []).includes(line));
        return pick(fresh.length ? fresh : lines);
      }
      if (task === "react") return pick(TONE_EMOJI[toneOf(input.text || "")]);
      if (task === "judge") {
        const chance = { harsh: 0, funny: 0.9, happy: 0.85, rough: 0.75, question: 0.5, neutral: 0.4 }[toneOf(input.text || "")];
        return Math.random() < chance ? "はい" : "いいえ";
      }
      if (task === "trend") return shuffle(TRENDS).slice(0, input.count).join("\n");
      throw new Error(`unknown task: ${task}`);
    },
  };
}
