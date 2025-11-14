import dotenv from "dotenv";
import express from "express";
import http from "http";
import {  WebSocketServer } from "ws";
import { v4 as uuidv4 } from "uuid";
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";
import OpenAI from "openai";

dotenv.config();

// Keep LangGraph minimal: use it for persona (stateful, no loops)
import { StateGraph, Annotation } from "@langchain/langgraph";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();

// CORS middleware for Express
app.use((req, res, next) => {
  const origin = req.headers.origin;
  
  // Function to check if origin is allowed
  const isAllowedOrigin = (origin) => {
    if (!origin) return false;
    
    // Allow localhost with any port
    if (origin.match(/^https?:\/\/(localhost|127\.0\.0\.1|0\.0\.0\.0)(:\d+)?$/)) {
      return true;
    }
    
    // Allow Vercel domains
    if (origin.includes('vercel.app')) {
      return true;
    }
    
    // Allow Google Cloud Run domains
    if (origin.includes('.run.app')) {
      return true;
    }
    
    // Allow custom domains (HTTPS only in production)
    if (process.env.NODE_ENV === 'production' && origin.startsWith('https://')) {
      return true; // Allow any HTTPS origin in production (for custom domains)
    }
    
    // In development, allow all origins for easier testing
    if (process.env.NODE_ENV !== 'production') {
      return true;
    }
    
    return false;
  };
  
  if (isAllowedOrigin(origin)) {
    res.setHeader('Access-Control-Allow-Origin', origin);
  }
  
  res.setHeader('Access-Control-Allow-Methods', 'GET, POST, PUT, DELETE, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type, Authorization, X-Requested-With');
  res.setHeader('Access-Control-Allow-Credentials', 'true');
  res.setHeader('Access-Control-Max-Age', '86400'); // 24 hours
  
  if (req.method === 'OPTIONS') {
    return res.sendStatus(200);
  }
  next();
});

const server = http.createServer(app);

// WebSocket CORS verification function
const verifyClient = (info) => {
  const origin = info.origin;
  
  // Function to check if origin is allowed for WebSocket
  const isAllowedOrigin = (origin) => {
    if (!origin) return true; // Allow connections without origin (e.g., Postman, curl)
    
    // Allow localhost with any port
    if (origin.match(/^https?:\/\/(localhost|127\.0\.0\.1|0\.0\.0\.0)(:\d+)?$/)) {
      return true;
    }
    
    // Allow Vercel domains
    if (origin.includes('vercel.app')) {
      return true;
    }
    
    // Allow Google Cloud Run domains
    if (origin.includes('.run.app')) {
      return true;
    }
    
    // Allow custom domains (HTTPS only in production)
    if (process.env.NODE_ENV === 'production' && origin.startsWith('https://')) {
      return true; // Allow any HTTPS origin in production (for custom domains)
    }
    
    // In development, allow all origins for easier testing
    if (process.env.NODE_ENV !== 'production') {
      return true;
    }
    
    return false;
  };
  
  const allowed = isAllowedOrigin(origin);
  if (!allowed && origin) {
    console.warn(`WebSocket connection rejected from origin: ${origin}`);
  }
  return allowed;
};

const wss = new WebSocketServer({ 
  server,
  verifyClient, // CORS verification for WebSocket connections
  perMessageDeflate: false
});

// Add ping interval to keep connections alive (only in production to prevent load balancer timeouts)
// In development, this can cause issues, so we skip it locally
// Check NODE_ENV explicitly - default to development if not set
const isProduction = process.env.NODE_ENV === 'production';
const pingInterval = isProduction ? setInterval(() => {
  wss.clients.forEach((ws) => {
    try {
      // Skip if connection is already closed or closing
      // WebSocket.OPEN is 1, but we can also check the readyState property
      if (ws.readyState !== 1) { // 1 = OPEN
        return;
      }
      
      // Only check isAlive if it was set (i.e., connection was established with keepalive)
      // If isAlive is undefined, it means this connection was established before keepalive was enabled
      // or in development mode, so skip the check
      if (ws.isAlive === undefined) {
        return; // Connection doesn't have keepalive enabled, skip
      }
      
      // Only terminate if connection was marked as dead from previous ping cycle
      if (ws.isAlive === false) {
        console.log('Terminating dead WebSocket connection (no pong response after 30s)');
        ws.terminate();
        return;
      }
      
      // Mark as potentially dead, send ping, wait for pong to set it back to true
      ws.isAlive = false;
      ws.ping();
    } catch (error) {
      console.error('Error in ping interval:', error);
    }
  });
}, 30000) : null; // Send ping every 30 seconds in production only

if (pingInterval) {
  console.log('WebSocket keepalive (ping/pong) enabled for production');
} else {
  console.log('WebSocket keepalive disabled (development mode)');
}

// Clean up interval when server closes
if (pingInterval) {
  server.on('close', () => {
    clearInterval(pingInterval);
  });
}

// OpenAI API key with fallback (you can set your actual key here as fallback)
// IMPORTANT: Replace 'your-openai-api-key-here' with your actual API key
// Or set it via environment variable: OPENAI_API_KEY=sk-your-key-here
const OPENAI_API_KEY = process.env.OPENAI_API_KEY || 'YOUR_OPENAI_API_KEY_HERE';
console.log('OpenAI API Key configured:', OPENAI_API_KEY ? 'Yes' : 'No');

// Google Cloud Run uses PORT env var, default to 8080 for production
const PORT = Number(process.env.PORT || (process.env.NODE_ENV === 'production' ? 8080 : 3001));
const openai = new OpenAI({ apiKey: OPENAI_API_KEY });

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
  customJobRole: { type: "string", optional: true },
  jobDescription: { type: "string", optional: true },
  isSelfPrep: { type: "boolean", optional: true },
  company: { type: "string", optional: true },
  selectedCompany: { type: "string", optional: true },
  customCompany: { type: "string", optional: true },
  selectedLanguage: { type: "string", optional: true },
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
    role:  "Software Engineer",
    roleId: undefined,
    customJobRole: undefined,
    jobDescription: undefined,
    isSelfPrep: false,
    company: undefined,
    selectedCompany: undefined,
    customCompany: undefined,
    selectedLanguage: undefined,
    selectedRound: undefined,
    level: "junior",
    language:  "en",

    styleTemplate: undefined,
    transcript: [],

    turns: 0,
    maxTurns: 6 ,  // number of interviewer replies (after greeting) before evaluation
    done: false,
    overallScore: 0,

    ...overrides,
  };
}

function normalizeState(s = {}, frontendValues = {}) {
  // Helper to check if value is valid (not undefined, null, or empty string)
  const isValid = (val) => val !== undefined && val !== null && val !== "";
  
  // Helper to preserve frontend values - use frontend values if provided, otherwise use state, otherwise default
  const getValue = (key, defaultValue) => {
    // Priority: frontend value > state value > default
    if (isValid(frontendValues[key])) {
      return String(frontendValues[key]).trim();
    }
    if (isValid(s[key])) {
      const value = s[key];
      if (typeof value === "string") {
        return value.trim();
      }
      return value;
    }
    return defaultValue;
  };

  return {
    // Use frontend values if available, otherwise preserve state, otherwise use defaults
    candidateName: isValid(s.candidateName) ? String(s.candidateName).trim() : undefined,
    role: getValue("role", "Software Engineer"),
    roleId: isValid(s.roleId) ? s.roleId : undefined,
    customJobRole: isValid(s.customJobRole) ? String(s.customJobRole).trim() : undefined,
    jobDescription: isValid(s.jobDescription) ? String(s.jobDescription).trim() : undefined,
    isSelfPrep: s.isSelfPrep === true || s.isSelfPrep === "true",
    company: isValid(s.company) ? String(s.company).trim() : undefined,
    selectedCompany: isValid(s.selectedCompany) ? String(s.selectedCompany).trim() : undefined,
    customCompany: isValid(s.customCompany) ? String(s.customCompany).trim() : undefined,
    selectedLanguage: getValue("selectedLanguage", "java"),
    selectedRound: getValue("selectedRound", "technical"),
    level: getValue("level", "junior"),
    language: getValue("language", "en"),

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
  // Don't normalize here - use state as-is to preserve frontend values
  // normalizeState will be called after this
  
  // Map round IDs to round names and descriptions
  const roundInfo = {
    "technical": { name: "Technical Round", focus: "technical skills, problem-solving, coding abilities, algorithms, and data structures" },
    "hr": { name: "HR Round", focus: "behavioral questions, communication skills, cultural fit, soft skills, and team collaboration" },
    "managerial": { name: "Managerial Round", focus: "leadership, management experience, strategic thinking, decision-making, and team management" },
    "system-design": { name: "System Design Round", focus: "architecture design, scalability, system planning, distributed systems, and technical architecture" },
    "coding": { name: "Coding Round", focus: "live coding, algorithms, data structures, problem-solving, and code quality" },
  };
  
  const round = state.selectedRound ? roundInfo[state.selectedRound] || { name: state.selectedRound, focus: "relevant skills" } : null;
  
  // Use custom job role if provided
  const jobRole = state.customJobRole || state.role || "Software Engineer";
  
  let context = `${state.level || "junior"} ${jobRole} interview`;
  
  // Add company information if company specific is selected
  if (state.isSelfPrep && state.company && state.company.trim() !== "") {
    context += ` for ${state.company.trim()}`;
  }
  
  // Add job description to context if provided
  if (state.jobDescription && state.jobDescription.trim() !== "") {
    context += `. Job Description: ${state.jobDescription.trim()}`;
  }
  
  if (state.selectedLanguage && state.selectedLanguage.trim() !== "") {
    context += ` focusing on ${state.selectedLanguage}`;
  }
  if (round) {
    context += ` for ${round.name}`;
  }
  
  // Map language codes to language names for better LLM understanding
  const languageNames = {
    "en": "English",
    "hi": "Hindi",
    "te": "Telugu",
    "ta": "Tamil",
    "fr": "French",
  };
  
  // Ensure we have a valid language - use the actual state value
  const actualLanguage = state.language || "en";
  const languageName = languageNames[actualLanguage] || actualLanguage;
  
  console.log("=== GENERATE STYLE - LANGUAGE & ROUND CHECK ===");
  console.log({
    stateLanguage: state.language,
    actualLanguage: actualLanguage,
    languageName: languageName,
    selectedRound: state.selectedRound,
    round: round ? round.name : "none",
    jobRole: jobRole,
    hasJobDescription: !!(state.jobDescription && state.jobDescription.trim() !== ""),
  });
  
  const roundContext = round ? ` This is a ${round.name}, so focus on ${round.focus}.` : "";
  
  // Build job context with description if available
  let jobContext = `Role: ${jobRole} (${state.level || "junior"} level)`;
  if (state.jobDescription && state.jobDescription.trim() !== "") {
    jobContext += `\nJob Description: ${state.jobDescription.trim()}`;
  }
  
  const content = await llm(
    [
      {
        role: "user",
        content: `You are an interviewer. Generate a short interviewer persona.

CRITICAL LANGUAGE REQUIREMENT: You MUST respond in ${languageName} (language code: ${actualLanguage}). Every word you generate must be in ${languageName}. Do not use English or any other language.

Context: ${context}.${roundContext}
${round ? `ROUND TYPE: This is a ${round.name}. You MUST ask questions focused on: ${round.focus}.` : ""}

${jobContext}

Keep it to 2–3 sentences. Include tone/style + 1–2 rules (be concise, ask relevant follow-ups).
${round ? `As this is a ${round.name}, emphasize asking questions related to ${round.focus}.` : ""}
${state.jobDescription ? `Use the job description provided above to tailor questions to the specific role requirements.` : ""}

IMPORTANT: 
- Respond ONLY in ${languageName}. Do not use English or any other language.
- The persona should NOT instruct to use "Interviewer:" prefix or any speaker labels.`
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
async function ttsToWS(ws, text, voice = "alloy", format = "mp3", language = "en") {
  // Log the text being sent to TTS for debugging
  console.log("=== TTS REQUEST ===");
  console.log({
    language: language,
    textPreview: text.substring(0, 100) + (text.length > 100 ? "..." : ""),
    textLength: text.length,
    voice: voice,
  });
  
  const audio = await openai.audio.speech.create({
    model: "tts-1", // Use tts-1 which has better multilingual support
    voice,
    input: text,
    format, // "mp3" | "wav"
    // Note: OpenAI TTS auto-detects language from text, no language parameter available
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
  
  // Map language codes to language names for better LLM understanding
  const languageNames = {
    "en": "English",
    "hi": "Hindi",
    "te": "Telugu",
    "ta": "Tamil",
    "fr": "French",
  };
  
  // Build greeting prompt with candidate name if provided
  const candidateGreeting = s.candidateName && s.candidateName.trim() !== "" 
    ? `Greet the candidate by name (${s.candidateName}) and`
    : "Greet the candidate and";
  
  // Map round IDs to round names for greeting context
  const roundInfo = {
    "technical": { name: "Technical Round" },
    "hr": { name: "HR Round" },
    "managerial": { name: "Managerial Round" },
    "system-design": { name: "System Design Round" },
    "coding": { name: "Coding Round" },
  };
  const round = s.selectedRound ? roundInfo[s.selectedRound] || { name: s.selectedRound } : null;
  
  // Ensure we have a valid language
  const actualLanguage = s.language || "en";
  const languageName = languageNames[actualLanguage] || actualLanguage;
  
  console.log("=== SPEAK GREETING - LANGUAGE & ROUND CHECK ===");
  console.log({
    language: s.language,
    actualLanguage: actualLanguage,
    languageName: languageName,
    selectedRound: s.selectedRound,
    round: round ? round.name : "none",
  });
  
  // Use custom job role if provided
  const jobRole = s.customJobRole || s.role || "Software Engineer";
  
  // Build greeting context with job description if available
  let greetingContext = `Mention role (${jobRole}) and level (${s.level}).`;
  // Company specific interview - add company context
  if (s.isSelfPrep && s.company && s.company.trim() !== "") {
    greetingContext += ` Mention that this is a practice interview for ${s.company.trim()}.`;
  }
  if (s.jobDescription && s.jobDescription.trim() !== "") {
    greetingContext += ` Briefly reference the job requirements if relevant.`;
  }
  if (round) {
    greetingContext += ` Mention that this is a ${round.name}.`;
  }
  
  const content = await llm(
    [
      {
        role: "system",
        content: `You are a professional interviewer. ${candidateGreeting} briefly in ${languageName} (language code: ${actualLanguage}). 

CRITICAL LANGUAGE REQUIREMENT: You MUST respond in ${languageName}. Every word must be in ${languageName}. Do not use English or any other language.

2 sentences max. ${greetingContext} Ask them to introduce themselves in ~30 seconds before we begin.

${s.jobDescription ? `Job Description Context: ${s.jobDescription.trim()}\nUse this to make the greeting more specific to the role if appropriate.` : ""}

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
  
  // Log the generated greeting text to verify it's in the correct language
  console.log("=== GENERATED GREETING TEXT ===");
  console.log({
    language: s.language,
    actualLanguage: actualLanguage,
    languageName: languageName,
    textPreview: cleaned.substring(0, 200) + (cleaned.length > 200 ? "..." : ""),
    textLength: cleaned.length,
  });
  
  // Pass language to TTS for logging
  await ttsToWS(ws, cleaned, voice, "mp3", actualLanguage);
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

  // Map language codes to language names for better LLM understanding
  const languageNames = {
    "en": "English",
    "hi": "Hindi",
    "te": "Telugu",
    "ta": "Tamil",
    "fr": "French",
  };
  
  // Ensure we have a valid language - use the actual state value
  const actualLanguage = s.language || "en";
  const languageName = languageNames[actualLanguage] || actualLanguage;

  // Map round IDs to round names and focus areas
  const roundInfo = {
    "technical": { name: "Technical Round", focus: "technical skills, problem-solving, coding abilities, algorithms, data structures, and technical depth" },
    "hr": { name: "HR Round", focus: "behavioral questions, communication skills, cultural fit, soft skills, team collaboration, and work experience" },
    "managerial": { name: "Managerial Round", focus: "leadership, management experience, strategic thinking, decision-making, team management, and conflict resolution" },
    "system-design": { name: "System Design Round", focus: "architecture design, scalability, system planning, distributed systems, technical architecture, and trade-offs" },
    "coding": { name: "Coding Round", focus: "live coding, algorithms, data structures, problem-solving, code quality, and optimization" },
  };
  
  const round = s.selectedRound ? roundInfo[s.selectedRound] || { name: s.selectedRound, focus: "relevant skills" } : null;
  
  // Use custom job role if provided
  const jobRole = s.customJobRole || s.role || "Software Engineer";
  
  console.log("=== GENERATE INTERVIEWER TURN - LANGUAGE & ROUND CHECK ===");
  console.log({
    language: s.language,
    actualLanguage: actualLanguage,
    languageName: languageName,
    selectedRound: s.selectedRound,
    round: round ? round.name : "none",
    jobRole: jobRole,
    hasJobDescription: !!(s.jobDescription && s.jobDescription.trim() !== ""),
  });

  const systemPrompt = (s.styleTemplate || "You are an interviewer.") + 
    " CRITICAL RULE: You must NEVER start your response with the word 'Interviewer' or 'Interviewer:' or any speaker label. Always start directly with your question or statement. Do not use any prefixes, labels, or speaker identifiers.";
  
  const roundInstructions = round 
    ? `This is a ${round.name} interview. Focus specifically on ${round.focus}. Ask questions that are appropriate for this round type.`
    : "";
  
  // Build job context with description
  let jobContext = `Role: ${jobRole} (${s.level} level)`;
  // Company specific interview - add company context
  if (s.isSelfPrep && s.company && s.company.trim() !== "") {
    jobContext += `\n\nCompany: ${s.company.trim()}\n\nThis is a practice interview for ${s.company.trim()}. Tailor your questions to reflect the interview style, culture, and technical expectations typical of ${s.company.trim()}. Ask questions that would be relevant for someone interviewing at ${s.company.trim()}.`;
  }
  if (s.jobDescription && s.jobDescription.trim() !== "") {
    jobContext += `\n\nJob Description:\n${s.jobDescription.trim()}\n\nUse this job description to tailor your questions to the specific role requirements, responsibilities, and skills needed. Ask questions that are directly relevant to what the job description mentions.`;
  }
  
  const content = await llm(
    [
      { role: "system", content: systemPrompt },
      {
        role: "user",
        content: `Conversation so far:
${history || "(no previous messages)"}

CRITICAL LANGUAGE REQUIREMENT: You MUST respond in ${languageName} (language code: ${s.language}). Every word you generate must be in ${languageName}. Do not use English or any other language.

${jobContext}

Now continue as the interviewer for this interview${round ? ` (${round.name})` : ""}.
${roundInstructions}
${round ? `ROUND TYPE: This is a ${round.name}. You MUST ask questions focused on: ${round.focus}.` : ""}
${s.jobDescription ? `IMPORTANT: Reference the job description above when asking questions. Make your questions relevant to the specific role requirements mentioned in the job description.` : ""}

Be natural: you may briefly acknowledge their answer (1 short sentence) and ask a focused follow-up that is relevant to ${round ? round.focus : "the role and level"}${s.jobDescription ? " and the job description" : ""}.
Keep it concise (20–60 words). One paragraph. 

CRITICAL REQUIREMENTS:
1. Respond ONLY in ${languageName}. Do not use English or any other language.
2. You must NEVER start your response with the word 'Interviewer' or 'Interviewer:' or any speaker label. Always start directly with your question or statement. Do not use any prefixes, labels, or speaker identifiers.
3. ${s.jobDescription ? "Make sure your questions are tailored to the job description provided above." : ""}`
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
  
  // Log the generated text to verify it's in the correct language
  console.log("=== GENERATED INTERVIEWER TEXT ===");
  console.log({
    language: s.language,
    actualLanguage: actualLanguage,
    languageName: languageName,
    selectedRound: s.selectedRound,
    round: round ? round.name : "none",
    textPreview: cleaned.substring(0, 200) + (cleaned.length > 200 ? "..." : ""),
    textLength: cleaned.length,
  });
  
  // Validate response relevance before returning
  const isValid = await validateResponseRelevance(cleaned, s, round, jobRole);
  if (!isValid) {
    console.warn("Response failed relevance check, regenerating...");
    // Retry once with stricter instructions
    const retryContent = await llm(
      [
        { role: "system", content: systemPrompt + " CRITICAL: Your response MUST be directly related to the interview context, role, and candidate's previous answer. Do not go off-topic." },
        {
          role: "user",
          content: `Conversation so far:
${history || "(no previous messages)"}

CRITICAL LANGUAGE REQUIREMENT: You MUST respond in ${languageName} (language code: ${s.language}). Every word you generate must be in ${languageName}. Do not use English or any other language.

${jobContext}

IMPORTANT: The candidate just answered a question. You MUST:
1. Acknowledge their answer briefly (1 sentence)
2. Ask a FOLLOW-UP question that is DIRECTLY related to:
   - Their previous answer
   - The ${round ? round.name : "interview"} context
   - The ${jobRole} role at ${s.level} level
   ${round ? `- ${round.focus}` : ""}
   ${s.jobDescription ? "- The job description requirements" : ""}

DO NOT ask unrelated questions. DO NOT change topics abruptly. Stay focused on the interview.

Keep it concise (20–60 words). One paragraph.`
        }
      ],
      { temperature: 0.5 } // Lower temperature for more focused responses
    );
    
    let retryCleaned = (retryContent || "").trim();
    // Apply same cleaning
    retryCleaned = retryCleaned.replace(/^Interviewer\s*:\s*/i, "").trim();
    retryCleaned = retryCleaned.replace(/^Interviewer\s+/i, "").trim();
    
    const retryValid = await validateResponseRelevance(retryCleaned, s, round, jobRole);
    if (retryValid || retryCleaned.length > 10) {
      return retryCleaned;
    }
  }
  
  return cleaned;
}

// Validate if AI response is relevant to interview context
async function validateResponseRelevance(response, state, round, jobRole) {
  if (!response || response.length < 10) {
    return false; // Too short to be meaningful
  }
  
  // Check for common off-topic indicators
  const offTopicPatterns = [
    /^(thank you|thanks|goodbye|bye|that's all|end of interview)/i,
    /^(how can i help|what can i do|anything else)/i,
    /^(let me know|tell me if|please let)/i,
  ];
  
  for (const pattern of offTopicPatterns) {
    if (pattern.test(response)) {
      return false;
    }
  }
  
  // Use LLM to validate relevance
  try {
    const validationPrompt = `You are validating an interviewer's response in an interview context.

Interview Context:
- Role: ${jobRole} (${state.level} level)
${round ? `- Round: ${round.name} (${round.focus})` : ""}
${state.jobDescription ? `- Job Description: ${state.jobDescription.substring(0, 200)}...` : ""}

Candidate's Last Answer: ${state.transcript.filter(t => t.from === "candidate").slice(-1)[0]?.text || "N/A"}

Interviewer's Response: "${response}"

Is this response:
1. Relevant to the interview context and role?
2. A natural follow-up to the candidate's answer?
3. Appropriate for a ${round ? round.name : "general"} interview?

Respond with ONLY "YES" or "NO".`;
    
    const validation = await llm(
      [{ role: "user", content: validationPrompt }],
      { temperature: 0.1, model: "gpt-4o-mini" }
    );
    
    const isValid = validation.trim().toUpperCase().includes("YES");
    if (!isValid) {
      console.warn("Response validation failed:", {
        response: response.substring(0, 100),
        validation: validation.trim()
      });
    }
    return isValid;
  } catch (error) {
    console.error("Error validating response:", error);
    // If validation fails, allow the response (fail open)
    return true;
  }
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
  
  // Truncate transcript if too long to speed up processing
  const convo = s.transcript
    .map((t) => `${t.from === "interviewer" ? "Interviewer" : "Candidate"}: ${t.text}`)
    .join("\n");
  
  // Limit transcript length to last 4000 characters for faster processing
  const truncatedConvo = convo.length > 4000 ? convo.slice(-4000) + "\n[... earlier conversation truncated ...]" : convo;
  
  // Build context string
  let contextInfo = `Interview Context:
- Role: ${s.role} (${s.level} level)
- Questions Asked: ${questionsAsked}
- Answers Given: ${answersGiven}
- Completion Rate: ${questionsAsked > 0 ? Math.round((answersGiven / questionsAsked) * 100) : 0}%`;
  
  if (s.selectedLanguage) {
    contextInfo += `\n- Programming Language Focus: ${s.selectedLanguage}`;
  }
  if (s.selectedRound) {
    contextInfo += `\n- Interview Round: ${s.selectedRound}`;
  }

  // OPTIMIZATION: Run both LLM calls in PARALLEL instead of sequential
  // This reduces total evaluation time from ~20-30s to ~10-15s
  const [rubricJSON, summaryText] = await Promise.all([
    // First call: Generate rubric
    llm(
      [
        {
          role: "system",
          content: `You are a FRIENDLY and experienced tech interviewer evaluating a candidate. Your goal is to provide constructive, encouraging feedback that helps the candidate improve.

${contextInfo}

You must evaluate based on:
1. Communication Skills: Clarity, articulation, structure, listening, engagement
2. Technical Skills: Accuracy, depth, knowledge, correctness
3. Problem Solving: Approach, logic, creativity, analytical thinking
4. Behavior: Professionalism, attitude, confidence, enthusiasm
5. Relevance: How well answers match the role and level expectations

Return ONLY a JSON object:
{
  "communication": n (0-10),
  "technical": n (0-10),
  "problem_solving": n (0-10),
  "behavior": n (0-10),
  "relevance": n (0-10),
  "questions_answered": ${answersGiven},
  "total_questions": ${questionsAsked},
  "answer_quality": "excellent/good/fair/poor",
  "strengths": ["List 2-3 specific strengths in ${s.language}"],
  "improvements": {
    "communication": "Specific suggestion in ${s.language}",
    "technical": "Specific suggestion in ${s.language}",
    "problem_solving": "Specific suggestion in ${s.language}",
    "behavior": "Specific suggestion in ${s.language}"
  },
  "hiring_tips": ["Tip 1 in ${s.language}", "Tip 2", "Tip 3"],
  "dos": ["Do 1 in ${s.language}", "Do 2", "Do 3"],
  "donts": ["Don't 1 in ${s.language}", "Don't 2", "Don't 3"],
  "notes": "Overall evaluation summary in ${s.language}"
}`
        },
        {
          role: "user",
          content: `Conversation:\n\n${truncatedConvo}\n\nQ&A Pairs:\n${qaPairs.map((qa, idx) => `Q${idx + 1}: ${qa.question}\nA${idx + 1}: ${qa.answer}`).join('\n\n')}`
        }
      ],
      { temperature: 0.3, model: "gpt-4o-mini" }
    ),
    
    // Second call: Generate summary (can run in parallel)
    llm(
      [
        {
          role: "system",
          content: `You are a FRIENDLY and supportive interviewer providing constructive feedback. Be encouraging, specific, and helpful.

${contextInfo}

Provide a FRIENDLY evaluation in ${s.language}:
1. Start with a positive, encouraging tone
2. Acknowledge what went well (${answersGiven}/${questionsAsked} questions answered)
3. Highlight 2-3 key strengths
4. Identify 2-3 specific areas for improvement
5. Provide practical tips
6. End with encouragement

Be friendly, constructive, and specific. 8-12 sentences. Use a warm, supportive tone.`
        },
        {
          role: "user",
          content: `Conversation:\n\n${truncatedConvo}\n\nQ&A Pairs:\n${qaPairs.map((qa, idx) => `Q${idx + 1}: ${qa.question}\nA${idx + 1}: ${qa.answer}`).join('\n\n')}`
        }
      ],
      { temperature: 0.4, model: "gpt-4o-mini" }
    )
  ]);

  // Parse rubric JSON
  let parsed = {
    communication: 5,
    technical: 5,
    problem_solving: 5,
    behavior: 5,
    relevance: 5,
    questions_answered: answersGiven,
    total_questions: questionsAsked,
    answer_quality: "fair",
    strengths: [],
    improvements: {
      communication: "",
      technical: "",
      problem_solving: "",
      behavior: ""
    },
    hiring_tips: [],
    dos: [],
    donts: [],
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
    (Number(parsed.behavior) || 0) +
    (Number(parsed.relevance) || 0);

  return {
    overallScore: total,
    summaryText: summaryText || "",
    rubric: parsed,
  };
}

/* --------------------- Sessions --------------------- */

const SESSIONS = new Map(); // sessionId -> { state }
// Use /tmp for Cloud Run (read-only filesystem except /tmp) or local tmp directory
const TMP_DIR = process.env.NODE_ENV === 'production' ? '/tmp' : path.join(__dirname, "tmp");

// Create tmp directory if it doesn't exist (only needed for local)
if (process.env.NODE_ENV !== 'production') {
  try {
    if (!fs.existsSync(TMP_DIR)) {
      fs.mkdirSync(TMP_DIR, { recursive: true });
    }
  } catch (err) {
    console.warn('Could not create tmp directory:', err.message);
  }
} else {
  console.log('Using /tmp directory for Cloud Run');
}

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
  let audioFormat = "webm"; // Default format, will be updated from client
  
  // Log connection for debugging
  console.log(`WebSocket connection opened. NODE_ENV: ${process.env.NODE_ENV || 'development'}`);
  
  // Initialize connection as alive for keepalive mechanism (only needed in production)
  // IMPORTANT: Only set isAlive if ping interval is actually running
  if (isProduction && pingInterval) {
    ws.isAlive = true;
    
    // Handle pong response to keep connection alive
    ws.on('pong', () => {
      ws.isAlive = true;
    });
  }
  
  // Handle connection errors
  ws.on('error', (error) => {
    console.error('WebSocket error:', error);
  });
  
  // Handle connection close
  ws.on('close', (code, reason) => {
    console.log(`WebSocket connection closed. Code: ${code}, Reason: ${reason?.toString() || 'none'}`);
  });

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
        voiceChoice = msg.voice ?? "alloy";

        // Log received values for debugging
        console.log("=== RECEIVED FROM FRONTEND ===");
        console.log("Raw message values:", {
          candidateName: msg.candidateName,
          role: msg.role,
          roleName: msg.roleName,
          roleId: msg.roleId,
          customJobRole: msg.customJobRole,
          jobDescription: msg.jobDescription,
          selectedLanguage: msg.selectedLanguage,
          selectedRound: msg.selectedRound,
          level: msg.level,
          language: msg.language,
          maxTurns: msg.maxTurns,
        });
        console.log("Value types:", {
          role: typeof msg.role,
          roleName: typeof msg.roleName,
          selectedLanguage: typeof msg.selectedLanguage,
          level: typeof msg.level,
          language: typeof msg.language,
        });

        // Preserve frontend values - directly use them, only use defaults if truly missing (undefined/null/empty string)
        // Frontend sends: role (roleName), roleId, selectedLanguage, selectedRound, level, language, maxTurns
        // Helper to check if value is valid (not undefined, null, or empty string)
        const isValid = (val) => val !== undefined && val !== null && val !== "";
        
        // Function to detect programming language from role or job description
        // Job description is optional - if not provided, detection works based on role alone
        const detectProgrammingLanguage = (roleText, jobDescriptionText) => {
          // If no role or job description, cannot detect
          if (!roleText && !jobDescriptionText) {
            return null;
          }
          
          // Combine role and job description (job description is optional)
          // Prioritize role text, then job description
          const roleLower = (roleText || "").toLowerCase();
          const jobDescLower = (jobDescriptionText || "").toLowerCase();
          const combinedText = `${roleLower} ${jobDescLower}`.trim();
          
          // Language detection patterns - map keywords to language IDs
          const languagePatterns = {
            "python": ["python", "django", "flask", "pandas", "numpy", "tensorflow", "pytorch", "scikit-learn", "fastapi"],
            "java": ["java", "spring", "hibernate", "maven", "gradle", "jvm", "jdk", "j2ee", "jee"],
            "javascript": ["javascript", "js", "node", "nodejs", "react", "vue", "angular", "typescript", "express", "next.js", "nextjs"],
            "cpp": ["c++", "cpp", "c plus plus", "qt", "boost", "stl"],
            "c": ["c programming", "embedded c", "ansi c", " c ", " c,"],
            "csharp": ["c#", "csharp", ".net", "asp.net", "entity framework", "dotnet"],
            "go": ["golang", "go language", "go programming", " go "],
            "rust": ["rust", "rustlang", " rust "],
            "php": ["php", "laravel", "symfony", "wordpress", "drupal"],
            "ruby": ["ruby", "rails", "ruby on rails", "ror"],
            "swift": ["swift", "ios development", "swiftui", "swift programming"],
            "kotlin": ["kotlin", "android kotlin", " kotlin "],
            "scala": ["scala", "akka", "spark scala", " scala "],
            "r": ["r programming", "r language", "r studio", " r ", " r,"],
            "matlab": ["matlab", "simulink", " matlab "],
            "sql": ["sql", "mysql", "postgresql", "oracle sql", "sql server", "postgres", "mongodb"],
          };
          
          // Check each language pattern (order matters - check more specific first)
          for (const [languageId, keywords] of Object.entries(languagePatterns)) {
            for (const keyword of keywords) {
              if (combinedText.includes(keyword)) {
                return languageId;
              }
            }
          }
          
          return null;
        };
        
        // Detect language from role or job description if not provided
        // Job description is optional - detection works with role alone if JD is missing
        const roleText = isValid(msg.role) ? msg.role : (isValid(msg.roleName) ? msg.roleName : "");
        const jobDescriptionText = isValid(msg.jobDescription) ? msg.jobDescription : "";
        const detectedLanguage = detectProgrammingLanguage(roleText, jobDescriptionText);
        
        let state = {
          candidateName: isValid(msg.candidateName) ? msg.candidateName : undefined,
          role: isValid(msg.role) ? msg.role : 
                (isValid(msg.roleName) ? msg.roleName : "Software Engineer"),
          roleId: isValid(msg.roleId) ? msg.roleId : undefined,
          customJobRole: isValid(msg.customJobRole) ? msg.customJobRole : undefined,
          jobDescription: isValid(msg.jobDescription) ? msg.jobDescription : undefined,
          selectedLanguage: isValid(msg.selectedLanguage) ? msg.selectedLanguage : (detectedLanguage || "java"),
          selectedRound: isValid(msg.selectedRound) ? msg.selectedRound : "technical",
          level: isValid(msg.level) ? msg.level : "junior",
          language: isValid(msg.language) ? msg.language : "en",
          maxTurns: (msg.maxTurns !== undefined && msg.maxTurns !== null) ? Number(msg.maxTurns) : 6,
          styleTemplate: undefined,
          transcript: [],
          turns: 0,
          done: false,
          overallScore: 0,
        };

        console.log("=== STATE AFTER INITIALIZATION ===");
        console.log({
          candidateName: state.candidateName,
          role: state.role,
          roleId: state.roleId,
          selectedLanguage: state.selectedLanguage,
          selectedRound: state.selectedRound,
          level: state.level,
          language: state.language,
          maxTurns: state.maxTurns,
        });

        // persona via LangGraph
        // Store frontend values before LangGraph might modify them
        const frontendValues = {
          role: state.role,
          roleId: state.roleId,
          selectedLanguage: state.selectedLanguage,
          selectedRound: state.selectedRound,
          level: state.level,
          language: state.language,
        };
        
        state = await styleGraph.invoke(state);
        
        console.log("=== STATE AFTER STYLEGRAPH ===");
        console.log({
          role: state.role,
          roleId: state.roleId,
          selectedLanguage: state.selectedLanguage,
          selectedRound: state.selectedRound,
          level: state.level,
          language: state.language,
        });
        
        console.log("=== FRONTEND VALUES TO RESTORE ===");
        console.log(frontendValues);
        
        // Restore frontend values that might have been modified by LangGraph
        // IMPORTANT: Directly assign to ensure values are preserved
        state.role = frontendValues.role;
        state.roleId = frontendValues.roleId;
        state.selectedLanguage = frontendValues.selectedLanguage;
        state.selectedRound = frontendValues.selectedRound;
        state.level = frontendValues.level;
        state.language = frontendValues.language;
        
        console.log("=== STATE AFTER DIRECT ASSIGNMENT ===");
        console.log({
          role: state.role,
          selectedRound: state.selectedRound,
          language: state.language,
        });
        
        // Only normalize fields that weren't set by frontend
        state = normalizeState(state, frontendValues);
        
        console.log("=== FINAL STATE AFTER NORMALIZE ===");
        console.log({
          role: state.role,
          roleId: state.roleId,
          selectedLanguage: state.selectedLanguage,
          selectedRound: state.selectedRound,
          level: state.level,
          language: state.language,
        });
        
        // Verify round mapping
        const roundInfo = {
          "technical": { name: "Technical Round" },
          "hr": { name: "HR Round" },
          "managerial": { name: "Managerial Round" },
          "system-design": { name: "System Design Round" },
          "coding": { name: "Coding Round" },
        };
        const mappedRound = state.selectedRound ? roundInfo[state.selectedRound] : null;
        
        // Map language codes to language names for logging
        const languageNames = {
          "en": "English",
          "hi": "Hindi",
          "te": "Telugu",
          "ta": "Tamil",
          "fr": "French",
        };
        
        console.log("=== ROUND MAPPING ===");
        console.log({
          selectedRound: state.selectedRound,
          mappedRound: mappedRound,
          language: state.language,
          languageName: languageNames[state.language] || state.language,
        });
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
        audioFormat = msg.format || "webm"; // Store format from client
        console.log(`Audio recording started with format: ${audioFormat}`);
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
        
        // If interview is already done/stopped, don't process audio
        if (state.done) {
          return;
        }

        // Check if we received any audio chunks
        if (audioChunks.length === 0) {
          console.warn("No audio chunks received for transcription");
          state.transcript.push({ from: "candidate", text: "(no audio received)" });
          ws.send(JSON.stringify({
            type: "transcript_update",
            transcript: state.transcript,
          }));
          return;
        }

        // Determine file extension based on format (default to webm for backward compatibility)
        const fileExtension = audioFormat === 'mp4' ? 'mp4' : 
                             audioFormat === 'aac' ? 'm4a' : 
                             'webm';
        
        // Save & transcribe (webm/opus or mp4/aac)
        const tmpPath = path.join(TMP_DIR, `ans_${Date.now()}.${fileExtension}`);
        const audioBuffer = Buffer.concat(audioChunks);
        console.log(`Received ${audioChunks.length} audio chunks, total size: ${audioBuffer.length} bytes, format: ${audioFormat}`);
        
        try {
          fs.writeFileSync(tmpPath, audioBuffer);
        } catch (writeError) {
          console.error("Error writing audio file:", writeError);
          state.transcript.push({ from: "candidate", text: "(error saving audio)" });
          ws.send(JSON.stringify({
            type: "transcript_update",
            transcript: state.transcript,
          }));
          return;
        }

        let transcriptText = "";
        try {
          console.log("Starting transcription with Whisper...");
          const resp = await openai.audio.transcriptions.create({
            model: "whisper-1",
            file: fs.createReadStream(tmpPath),
            response_format: "json",
            temperature: 0,
            language: state.language || "en",
          });
          transcriptText = (resp.text || "").trim();
          console.log("Transcription result:", transcriptText || "(empty)");
        } catch (e) {
          console.error("STT error:", e.message || e);
          console.error("Error details:", {
            code: e.code,
            status: e.status,
            type: e.type,
            message: e.message
          });
        } finally {
          try {
            fs.unlinkSync(tmpPath);
          } catch (unlinkError) {
            console.warn("Error deleting temp file:", unlinkError);
          }
        }

        if (transcriptText) {
          state.transcript.push({ from: "candidate", text: transcriptText });
        } else {
          console.warn("No transcription text received");
          state.transcript.push({ from: "candidate", text: "(no speech recognized)" });
        }
        // Send transcript update to frontend so UI can display user's answer
        ws.send(JSON.stringify({
          type: "transcript_update",
          transcript: state.transcript,
        }));

        // CRITICAL: Re-check state.done after processing audio - it might have been set by a "stop" message
        // that arrived while we were processing the audio
        state = normalizeState(SESSIONS.get(sessionId)?.state || state);
        
        // If interview is done/stopped, don't generate question - evaluation should start
        if (state.done) {
          // Evaluation was triggered, don't generate question
          return;
        }

        // If we still have interviewer turns left, respond conversationally
        // BUT: Check if interview is done/stopped first - don't generate question if evaluation is starting
        if (state.turns < state.maxTurns - 1 && !state.done) {
          const reply = await generateInterviewerTurn(state);
          state.transcript.push({ from: "interviewer", text: reply });
          state.turns += 1;
          SESSIONS.set(sessionId, { state });

          // Send transcript update with interviewer's question
          ws.send(JSON.stringify({
            type: "transcript_update",
            transcript: state.transcript,
          }));

          // Get language from state for TTS logging
          const currentLanguage = state.language || "en";
          console.log("=== SENDING INTERVIEWER REPLY TO TTS ===");
          console.log({
            language: currentLanguage,
            textPreview: reply.substring(0, 200) + (reply.length > 200 ? "..." : ""),
          });
          await ttsToWS(ws, reply, voiceChoice, "mp3", currentLanguage);
          return;
        }

        // Last turn reached → evaluate
        const evaluation = await evaluateConversation(state);
        state.done = true;
        state.overallScore = evaluation.overallScore;
        state.transcript.push({ from: "interviewer", text: evaluation.summaryText });
        SESSIONS.set(sessionId, { state });

        // OPTIMIZATION: Send evaluation results IMMEDIATELY (don't wait for TTS)
        // This allows frontend to show results right away while TTS generates in background
        ws.send(JSON.stringify({
          type: "done",
          summaryText: evaluation.summaryText,
          overallScore: evaluation.overallScore,
          rubric: evaluation.rubric,
        }));

        // Send final transcript update
        ws.send(JSON.stringify({
          type: "transcript_update",
          transcript: state.transcript,
        }));

        // Generate TTS in background (non-blocking) - don't await, let it run async
        const summaryLanguage = state.language || "en";
        ttsToWS(ws, evaluation.summaryText, voiceChoice, "mp3", summaryLanguage)
          .catch(error => {
            console.error("TTS generation error (non-critical):", error);
            // Don't block if TTS fails - evaluation is already sent to frontend
          });
        
        return;
      }

      // Manual stop (optional) → evaluate early
      if (msg.type === "stop") {
        // Clear any pending audio chunks to stop processing
        audioChunks = [];
        
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

        // OPTIMIZATION: Send evaluation results IMMEDIATELY (don't wait for TTS)
        ws.send(JSON.stringify({
          type: "done",
          summaryText: evaluation.summaryText,
          overallScore: evaluation.overallScore,
          rubric: evaluation.rubric,
        }));

        // Send final transcript update
        ws.send(JSON.stringify({
          type: "transcript_update",
          transcript: state.transcript,
        }));

        // Generate TTS in background (non-blocking)
        const summaryLanguage = state.language || "en";
        ttsToWS(ws, evaluation.summaryText, voiceChoice, "mp3", summaryLanguage)
          .catch(error => {
            console.error("TTS generation error (non-critical):", error);
            // Don't block if TTS fails - evaluation is already sent to frontend
          });
        
        return;
      }
    } catch (err) {
      console.error(err);
      ws.send(JSON.stringify({ type: "error", error: "server_exception" }));
    }
  });
});

app.get("/health", (_, res) => res.json({ ok: true }));

// Start the server
server.listen(PORT, () => {
  console.log(`Server running on http://localhost:${PORT}`);
  console.log(`WebSocket server ready for connections`);
});





