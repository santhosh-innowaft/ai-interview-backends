import "dotenv/config";
import express from "express";
import http from "http";
import { WebSocketServer } from "ws";
import { v4 as uuidv4 } from "uuid";
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";
import OpenAI from "openai";

// Keep LangGraph minimal: use it for persona (stateful, no loops)
import { StateGraph, Annotation } from "@langchain/langgraph";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
const server = http.createServer(app);
const wss = new WebSocketServer({ server });

const PORT = Number(process.env.PORT || 3001);
const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

/* --------------------- Schema & State --------------------- */
/** Conversational (voice-only) interview:
 * - No question text sent to client
 * - We keep full transcript with speaker roles
 * - Persona (styleTemplate) created once via LangGraph
 */
const InterviewState = Annotation.Root({
  candidateName: { type: "string", optional: true },
  role: { type: "string" },
  roleId: { type: "string", optional: true },
  selectedLanguage: { type: "string", optional: true },
  selectedSubject: { type: "string", optional: true },
  selectedRound: { type: "string", optional: true },
  level: { type: "string" },
  language: { type: "string" },

  styleTemplate: { type: "string", optional: true },

  // conversational transcript (not just Q/A)
  transcript: {
    type: "array",
    items: {
      type: "object",
      properties: {
        from: { type: "string" }, // "interviewer" | "candidate"
        text: { type: "string" }
      }
    }
  },

  turns: { type: "number" },       // interviewer turns completed (excludes greeting)
  maxTurns: { type: "number" },    // total interviewer turns before evaluation
  done: { type: "boolean" },
  overallScore: { type: "number" },
});

function initialState(overrides = {}) {
  return {
    candidateName: undefined,
    role: "Software Engineer",
    roleId: undefined,
    selectedLanguage: undefined,
    selectedSubject: undefined,
    selectedRound: undefined,
    level: "junior",
    language: "en",

    styleTemplate: undefined,
    transcript: [],

    turns: 0,
    maxTurns: 6,   // number of interviewer replies (after greeting) before evaluation
    done: false,
    overallScore: 0,

    ...overrides,
  };
}

function normalizeState(s = {}) {
  return {
    candidateName: s.candidateName ?? undefined,
    role: s.role || "Software Engineer",
    roleId: s.roleId ?? undefined,
    selectedLanguage: s.selectedLanguage ?? undefined,
    selectedSubject: s.selectedSubject ?? undefined,
    selectedRound: s.selectedRound ?? undefined,
    level: s.level || "junior",
    language: s.language || "en",

    styleTemplate: s.styleTemplate ?? undefined,
    transcript: Array.isArray(s.transcript) ? s.transcript : [],

    turns: typeof s.turns === "number" ? s.turns : 0,
    maxTurns: typeof s.maxTurns === "number" ? s.maxTurns : 6,
    done: !!s.done,
    overallScore: typeof s.overallScore === "number" ? s.overallScore : 0,
  };
}

/* --------------------- LLM helpers --------------------- */

async function llm(messages, { model = "gpt-4o-mini", temperature = 0.5 } = {}) {
  const res = await openai.chat.completions.create({ model, temperature, messages });
  return res.choices?.[0]?.message?.content ?? "";
}

/* --------------------- Minimal LangGraph: persona only --------------------- */

async function generateStyle(state) {
  state = normalizeState(state);
  let context = `${state.level} ${state.role} interview`;
  if (state.selectedLanguage) {
    context += ` focusing on ${state.selectedLanguage}`;
  }
  if (state.selectedSubject) {
    context += ` covering ${state.selectedSubject}`;
  }
  if (state.selectedRound) {
    context += ` for ${state.selectedRound} round`;
  }
  
  const content = await llm(
    [
      {
        role: "user",
        content: `Generate a short interviewer persona in language code: ${state.language}.
Context: ${context}.
Keep it to 2–3 sentences. Include tone/style + 1–2 rules (be concise, ask relevant follow-ups).
Respond ONLY in ${state.language}.
IMPORTANT: The persona should NOT instruct to use "Interviewer:" prefix or any speaker labels.`
      }
    ],
    { temperature: 0.6 }
  );
  return { ...state, styleTemplate: content };
}

function buildStyleGraph() {
  const g = new StateGraph(InterviewState);
  g.addNode("generateStyle", generateStyle);
  g.setEntryPoint("generateStyle");
  return g.compile();
}

const styleGraph = buildStyleGraph();

/* --------------------- Conversation helpers --------------------- */

// TTS to the websocket as binary chunks
async function ttsToWS(ws, text, voice = "alloy", format = "mp3") {
  const audio = await openai.audio.speech.create({
    model: "gpt-4o-mini-tts",
    voice,
    input: text,
    format, // "mp3" | "wav"
  });
  const buf = Buffer.from(await audio.arrayBuffer());
  const CHUNK = 32 * 1024;
  for (let i = 0; i < buf.length; i += CHUNK) {
    ws.send(buf.subarray(i, i + CHUNK));
  }
  ws.send(JSON.stringify({ type: "tts_done", format }));
}

// Greeting to start the interview
async function speakGreeting(ws, state, voice = "alloy") {
  const s = normalizeState(state);
  const content = await llm(
    [
      {
        role: "system",
        content: `You are a professional interviewer. Greet the candidate briefly in ${s.language}. 2 sentences max. Mention role (${s.role}) and level (${s.level}). Ask them to introduce themselves in ~30 seconds before we begin.

CRITICAL RULE: You must NEVER start your response with the word 'Interviewer' or 'Interviewer:' or any speaker label. Always start directly with your greeting. Do not use any prefixes, labels, or speaker identifiers.`
      }
    ],
    { temperature: 0.6 }
  );
  
  // Strip any "Interviewer" prefix if the AI includes it (comprehensive removal)
  let cleaned = (content || "").trim();
  
  // Multiple passes to catch all variations
  // Pattern 1: "Interviewer:" with colon
  cleaned = cleaned.replace(/^Interviewer\s*:\s*/i, "");
  // Pattern 2: "Interviewer " with space
  cleaned = cleaned.replace(/^Interviewer\s+/i, "");
  // Pattern 3: "Interviewer-" with dash
  cleaned = cleaned.replace(/^Interviewer\s*-\s*/i, "");
  // Pattern 4: "Interviewer." with period
  cleaned = cleaned.replace(/^Interviewer\s*\.\s*/i, "");
  // Pattern 5: Just "Interviewer" at start (catch any remaining)
  cleaned = cleaned.replace(/^Interviewer\b\s*/i, "");
  // Pattern 6: With any punctuation after
  cleaned = cleaned.replace(/^Interviewer\s*[:\-\.\s]+\s*/i, "");
  // Pattern 7: With quotes or brackets
  cleaned = cleaned.replace(/^["'(\[]?\s*Interviewer\s*[:.\-\s)]+\s*/i, "");
  
  // Final trim
  cleaned = cleaned.trim();
  
  // If it still starts with "Interviewer", remove it one more time
  if (/^Interviewer/i.test(cleaned)) {
    cleaned = cleaned.replace(/^Interviewer\s*[:.\-\s]*\s*/i, "").trim();
  }
  
  await ttsToWS(ws, cleaned, voice);
  return cleaned;
}

/** Generate the interviewer's next conversational turn:
 * - Not just questions; can include brief feedback and then a follow-up question
 * - Keep it natural and short (20–60 words), in target language
 */
async function generateInterviewerTurn(state) {
  const s = normalizeState(state);
  const history = s.transcript
    .map((t) => `${t.from === "interviewer" ? "Interviewer" : "Candidate"}: ${t.text}`)
    .join("\n");

  const systemPrompt = (s.styleTemplate || "You are an interviewer.") + 
    " CRITICAL RULE: You must NEVER start your response with the word 'Interviewer' or 'Interviewer:' or any speaker label. Always start directly with your question or statement. Do not use any prefixes, labels, or speaker identifiers.";
  
  const content = await llm(
    [
      { role: "system", content: systemPrompt },
      {
        role: "user",
        content: `Conversation so far (reply in ${s.language}):
${history || "(no previous messages)"}

Now continue as the interviewer for a ${s.level} ${s.role} interview.
Be natural: you may briefly acknowledge their answer (1 short sentence) and ask a focused follow-up.
Keep it concise (20–60 words). One paragraph. Respond ONLY in ${s.language}.

CRITICAL RULE: You must NEVER start your response with the word 'Interviewer' or 'Interviewer:' or any speaker label. Always start directly with your question or statement. Do not use any prefixes, labels, or speaker identifiers.`
      }
    ],
    { temperature: 0.7 }
  );

  // Strip any "Interviewer" prefix if the AI includes it (comprehensive removal)
  let cleaned = (content || "").trim();
  
  // Multiple passes to catch all variations
  // Pattern 1: "Interviewer:" with colon
  cleaned = cleaned.replace(/^Interviewer\s*:\s*/i, "");
  // Pattern 2: "Interviewer " with space
  cleaned = cleaned.replace(/^Interviewer\s+/i, "");
  // Pattern 3: "Interviewer-" with dash
  cleaned = cleaned.replace(/^Interviewer\s*-\s*/i, "");
  // Pattern 4: "Interviewer." with period
  cleaned = cleaned.replace(/^Interviewer\s*\.\s*/i, "");
  // Pattern 5: Just "Interviewer" at start (catch any remaining)
  cleaned = cleaned.replace(/^Interviewer\b\s*/i, "");
  // Pattern 6: With any punctuation after
  cleaned = cleaned.replace(/^Interviewer\s*[:\-\.\s]+\s*/i, "");
  // Pattern 7: With quotes or brackets
  cleaned = cleaned.replace(/^["'(\[]?\s*Interviewer\s*[:.\-\s)]+\s*/i, "");
  
  // Final trim
  cleaned = cleaned.trim();
  
  // If it still starts with "Interviewer", remove it one more time
  if (/^Interviewer/i.test(cleaned)) {
    cleaned = cleaned.replace(/^Interviewer\s*[:.\-\s]*\s*/i, "").trim();
  }
  
  return cleaned;
}

// Final evaluation based on the whole transcript
async function evaluateConversation(state) {
  const s = normalizeState(state);
  
  // Count questions and answers
  const interviewerMessages = s.transcript.filter(t => t.from === "interviewer");
  const candidateMessages = s.transcript.filter(t => t.from === "candidate");
  const questionsAsked = interviewerMessages.length - 1; // Exclude greeting
  const answersGiven = candidateMessages.length;
  
  // Build conversation with Q&A pairs for context
  const qaPairs = [];
  let currentQuestion = null;
  for (const entry of s.transcript) {
    if (entry.from === "interviewer") {
      currentQuestion = entry.text;
    } else if (entry.from === "candidate" && currentQuestion) {
      qaPairs.push({
        question: currentQuestion,
        answer: entry.text,
      });
      currentQuestion = null;
    }
  }
  
  const convo = s.transcript
    .map((t) => `${t.from === "interviewer" ? "Interviewer" : "Candidate"}: ${t.text}`)
    .join("\n");

  // Build context string with all details
  let contextInfo = `Interview Context:
- Role: ${s.role} (${s.level} level)
- Questions Asked: ${questionsAsked}
- Answers Given: ${answersGiven}
- Completion Rate: ${questionsAsked > 0 ? Math.round((answersGiven / questionsAsked) * 100) : 0}%`;
  
  if (s.selectedLanguage) {
    contextInfo += `\n- Programming Language Focus: ${s.selectedLanguage}`;
  }
  if (s.selectedSubject) {
    contextInfo += `\n- Subject: ${s.selectedSubject}`;
  }
  if (s.selectedRound) {
    contextInfo += `\n- Interview Round: ${s.selectedRound}`;
  }

  // Strict evaluation with detailed analysis
  const rubricJSON = await llm(
    [
      {
        role: "system",
        content: `You are a STRICT and experienced tech interviewer evaluating a candidate. 

${contextInfo}

You must evaluate STRICTLY based on:
1. How many questions were answered (${answersGiven} out of ${questionsAsked})
2. Quality and correctness of each answer
3. Technical depth and accuracy
4. Communication clarity
5. Problem-solving approach
6. Relevance to the role
7. Confidence and articulation

Be STRICT in your evaluation:
- Deduct points for incomplete answers
- Deduct points for incorrect technical information
- Deduct points for vague or unclear responses
- Reward detailed, accurate, and well-structured answers
- Consider the ${s.level} level expectations

Return ONLY a JSON object:
{
  "communication": n (0-10, strict),
  "technical": n (0-10, strict),
  "problem_solving": n (0-10, strict),
  "relevance": n (0-10, strict),
  "confidence": n (0-10, strict),
  "questions_answered": ${answersGiven},
  "total_questions": ${questionsAsked},
  "answer_quality": "excellent/good/fair/poor",
  "notes": "Detailed evaluation in ${s.language} including: how many questions answered correctly, what was right/wrong, specific strengths and weaknesses"
}`
      },
      {
        role: "user",
        content: `Full Conversation Transcript:\n\n${convo}\n\nQ&A Pairs Analysis:\n${qaPairs.map((qa, idx) => `Q${idx + 1}: ${qa.question}\nA${idx + 1}: ${qa.answer}`).join('\n\n')}`
      }
    ],
    { temperature: 0.1, model: "gpt-4o-mini" }
  );

  let parsed = {
    communication: 5,
    technical: 5,
    problem_solving: 5,
    relevance: 5,
    confidence: 5,
    questions_answered: answersGiven,
    total_questions: questionsAsked,
    answer_quality: "fair",
    notes: "Evaluation in progress.",
  };
  try {
    const j = JSON.parse(rubricJSON);
    parsed = { ...parsed, ...j };
  } catch (e) {
    console.error("Failed to parse rubric JSON:", e);
  }

  const total =
    (Number(parsed.communication) || 0) +
    (Number(parsed.technical) || 0) +
    (Number(parsed.problem_solving) || 0) +
    (Number(parsed.relevance) || 0) +
    (Number(parsed.confidence) || 0);

  // Detailed summary with strict evaluation
  const summaryText = await llm(
    [
      {
        role: "system",
        content: `You are a STRICT HR recruiter evaluating a candidate. 

${contextInfo}

Provide a STRICT evaluation in ${s.language}:
1. Start with how many questions were answered (${answersGiven}/${questionsAsked})
2. Analyze answer quality and correctness
3. Identify what was answered correctly and what was wrong
4. Highlight specific strengths and weaknesses
5. Provide detailed feedback on technical accuracy
6. End with a clear recommendation: "Recommendation: Hire / Maybe / No Hire"

Be honest and strict. 5-8 sentences.`
      },
      {
        role: "user",
        content: `Full Conversation:\n\n${convo}\n\nDetailed Q&A:\n${qaPairs.map((qa, idx) => `Question ${idx + 1}: ${qa.question}\nAnswer ${idx + 1}: ${qa.answer}\n---`).join('\n\n')}`
      }
    ],
    { temperature: 0.2, model: "gpt-4o-mini" }
  );

  return {
    overallScore: total,
    summaryText: summaryText || "",
    rubric: parsed,
  };
}

/* --------------------- Sessions --------------------- */

const SESSIONS = new Map(); // sessionId -> { state }
const TMP_DIR = path.join(__dirname, "tmp");
if (!fs.existsSync(TMP_DIR)) fs.mkdirSync(TMP_DIR);

/* --------------------- WS Protocol (voice-only, conversational) ---------------------

Client -> Server:
  {type:"start", language, role, level, maxTurns, candidateName?, voice?}
  {type:"answer_audio_start"}
  (binary audio chunks...)
  {type:"answer_audio_end"}
  {type:"stop"}  // optional manual stop/eval early

Server -> Client:
  {type:"session", sessionId}
  {type:"persona", text}          // optional (for logs)
  (binary) TTS chunks (greeting / interviewer turns / final summary)
  {type:"tts_done", format}
  {type:"done", summaryText, overallScore, rubric} // optional to display
-------------------------------------------------------------------- */

wss.on("connection", (ws) => {
  let sessionId = null;
  let audioChunks = [];
  let voiceChoice = "alloy";

  ws.on("message", async (data, isBinary) => {
    try {
      if (isBinary) {
        audioChunks.push(Buffer.from(data));
        return;
      }

      let msg;
      try {
        msg = JSON.parse(data.toString());
      } catch {
        ws.send(JSON.stringify({ type: "error", error: "invalid_json" }));
        return;
      }

      // START
      if (msg.type === "start") {
        sessionId = uuidv4();
        voiceChoice = msg.voice || "alloy";

        let state = initialState({
          candidateName: msg.candidateName ?? undefined,
          role: msg.role || msg.roleName || "Software Engineer",
          roleId: msg.roleId,
          selectedLanguage: msg.selectedLanguage,
          selectedSubject: msg.selectedSubject,
          selectedRound: msg.selectedRound,
          level: msg.level || "junior",
          language: msg.language || "en",
          maxTurns: Number(msg.maxTurns || 6),
        });

        // persona via LangGraph
        state = await styleGraph.invoke(state);
        state = normalizeState(state);
        SESSIONS.set(sessionId, { state });

        ws.send(JSON.stringify({ type: "session", sessionId }));
        if (state.styleTemplate) {
          ws.send(JSON.stringify({ type: "persona", text: state.styleTemplate }));
        }

        // greeting (voice)
        const greeting = await speakGreeting(ws, state, voiceChoice);
        state.transcript.push({ from: "interviewer", text: greeting });
        SESSIONS.set(sessionId, { state });

        // Send initial transcript with greeting
        ws.send(JSON.stringify({
          type: "transcript_update",
          transcript: state.transcript,
        }));

        // Expect candidate intro as first audio message
        return;
      }

      // BEGIN audio
      if (msg.type === "answer_audio_start") {
        audioChunks = [];
        return;
      }

      // END audio → transcribe → push candidate msg → continue or evaluate
      if (msg.type === "answer_audio_end") {
        const session = SESSIONS.get(sessionId);
        if (!session) {
          ws.send(JSON.stringify({ type: "error", error: "session_not_found" }));
          return;
        }
        let state = normalizeState(session.state);

        // Save & transcribe (webm/opus)
        const tmpPath = path.join(__dirname, `ans_${Date.now()}.webm`);
        fs.writeFileSync(tmpPath, Buffer.concat(audioChunks));

        let transcriptText = "";
        try {
          const resp = await openai.audio.transcriptions.create({
            model: "whisper-1",
            file: fs.createReadStream(tmpPath),
            response_format: "json",
            temperature: 0,
          });
          transcriptText = (resp.text || "").trim();
        } catch (e) {
          console.error("STT error:", e);
        } finally {
          fs.unlink(tmpPath, () => {});
        }

        if (transcriptText) {
          state.transcript.push({ from: "candidate", text: transcriptText });
        } else {
          state.transcript.push({ from: "candidate", text: "(no speech recognized)" });
        }
        // Send transcript update to frontend so UI can display user's answer
        ws.send(JSON.stringify({
          type: "transcript_update",
          transcript: state.transcript,
        }));

        // If we still have interviewer turns left, respond conversationally
        if (state.turns < state.maxTurns - 1) {
          const reply = await generateInterviewerTurn(state);
          state.transcript.push({ from: "interviewer", text: reply });
          state.turns += 1;
          SESSIONS.set(sessionId, { state });

          // Send transcript update with interviewer's question
          ws.send(JSON.stringify({
            type: "transcript_update",
            transcript: state.transcript,
          }));

          await ttsToWS(ws, reply, voiceChoice);
          return;
        }

        // Last turn reached → evaluate
        const evaluation = await evaluateConversation(state);
        state.done = true;
        state.overallScore = evaluation.overallScore;
        state.transcript.push({ from: "interviewer", text: evaluation.summaryText });
        SESSIONS.set(sessionId, { state });

        // Send final transcript update
        ws.send(JSON.stringify({
          type: "transcript_update",
          transcript: state.transcript,
        }));

        await ttsToWS(ws, evaluation.summaryText, voiceChoice);
        ws.send(JSON.stringify({
          type: "done",
          summaryText: evaluation.summaryText,
          overallScore: evaluation.overallScore,
          rubric: evaluation.rubric,
        }));
        return;
      }

      // Manual stop (optional) → evaluate early
      if (msg.type === "stop") {
        const session = SESSIONS.get(sessionId);
        if (!session) {
          ws.send(JSON.stringify({ type: "error", error: "session_not_found" }));
          return;
        }
        let state = normalizeState(session.state);
        if (state.done) return;

        const evaluation = await evaluateConversation(state);
        state.done = true;
        state.overallScore = evaluation.overallScore;
        state.transcript.push({ from: "interviewer", text: evaluation.summaryText });
        SESSIONS.set(sessionId, { state });

        // Send final transcript update
        ws.send(JSON.stringify({
          type: "transcript_update",
          transcript: state.transcript,
        }));

        await ttsToWS(ws, evaluation.summaryText, voiceChoice);
        ws.send(JSON.stringify({
          type: "done",
          summaryText: evaluation.summaryText,
          overallScore: evaluation.overallScore,
          rubric: evaluation.rubric,
        }));
        return;
      }
    } catch (err) {
      console.error(err);
      ws.send(JSON.stringify({ type: "error", error: "server_exception" }));
    }
  });
});

app.get("/health", (_, res) => res.json({ ok: true }));

server.listen(PORT, () => {
  console.log(`WS+HTTP server running on http://localhost:${PORT}`);
});





