/**
 * ContextEngine implementation for the Graphiti knowledge graph.
 *
 * When the host OpenClaw version supports registerContextEngine (v2026.3.7+),
 * this replaces the sidecar hooks (before_agent_start, before_compaction,
 * before_reset) with first-class assemble/ingest/compact methods.
 */

import { createRequire } from "node:module";
import type { GraphitiClient, GraphitiMessage } from "./client.js";
import type { DebugLog } from "./debug-log.js";
import { buildEpisodeName, buildProvenance, extractEpisodeContinuity, extractTextContent, extractTextsFromMessages, formatContinuityBlock, formatFactsAsContext, hasDeicticReferences, isContinuityGap, readSessionFileTail, sanitizeForCapture } from "./shared.js";
import type { LifecycleEvent } from "./shared.js";

const _require = createRequire(import.meta.url);
const PLUGIN_VERSION: string = (_require("./package.json") as any).version;

// ---------------------------------------------------------------------------
// Types from openclaw/plugin-sdk (canonical source: context-engine/types.ts).
// Declared locally so the plugin compiles without a hard dependency on a
// specific OpenClaw version. Keep structurally compatible with the SDK types.
// ---------------------------------------------------------------------------

export interface ContextEngineInfo {
  id: string;
  name: string;
  version: string;
  ownsCompaction: boolean;
  /**
   * Durable-commit contract declarations (openclaw/plugin-sdk
   * context-engine/types.ts). Hosts with a turn recorder require both the
   * current-turn transcript fence and atomic-idempotent turn advancement
   * before selecting a non-baseline engine; missing declarations degrade the
   * engine to legacy for the turn.
   */
  transcriptSemantics?: {
    currentTurnFence?: "before-current-turn-entry-v1";
    turnAdvancementIdempotency?: "atomic-idempotent-v1";
  };
}

export interface IngestResult {
  ingested: boolean;
}

export interface IngestBatchResult {
  ingestedCount: number;
}

export interface CommitTurnResult {
  status: "committed" | "duplicate";
}

export interface AssembleResult {
  messages: unknown[];
  systemPromptAddition?: string;
  estimatedTokens?: number;
}

export interface CompactResult {
  ok: boolean;
  compacted: boolean;
  reason?: string;
}

export interface BootstrapResult {
  bootstrapped: boolean;
  reason?: string;
  episodeCount?: number;
}

export interface SubagentSpawnPreparation {
  systemPromptAddition?: string;
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

interface PluginConfig {
  recallMaxFacts?: number;
  autoRecall?: boolean;
  autoCapture?: boolean;
  debug?: boolean;
  /** Allow extension without breaking. */
  [key: string]: unknown;
}

// ---------------------------------------------------------------------------
// GraphitiContextEngine
// ---------------------------------------------------------------------------

export class GraphitiContextEngine {
  readonly info: ContextEngineInfo = {
    id: "graphiti",
    name: "Graphiti Knowledge Graph",
    version: PLUGIN_VERSION,
    ownsCompaction: false,
    transcriptSemantics: {
      currentTurnFence: "before-current-turn-entry-v1",
      turnAdvancementIdempotency: "atomic-idempotent-v1",
    },
  };

  /** Cached healthy() result with TTL to avoid redundant HTTP round-trips. */
  private _healthCache: { result: boolean; ts: number } | null = null;
  private static readonly HEALTH_CACHE_TTL_MS = 15_000;

  /** Smart autoRecall state — tracks lifecycle events for continuity gap detection. */
  private _lastEvent: LifecycleEvent | null = null;
  private _sessionFile: string | null = null;
  private _sessionId: string | null = null;
  private _threadId: string | null = null;
  private _recalledSessions = new Set<string>();
  /** Advancement keys committed this process (durable idempotency layer 1). */
  private _committedTurns = new Set<string>();
  /** True once the host has invoked commitTurn — capture moves to that path. */
  private _durableMode = false;

  constructor(
    private client: GraphitiClient,
    private cfg: PluginConfig,
    private groupId: string,
    private debugLog: DebugLog,
    private logger?: { info?: (...args: any[]) => void; warn: (...args: any[]) => void },
  ) {}

  /** Signal that the next assemble() should fire recovery for this session. */
  private signalRecovery(sessionId: string, event: LifecycleEvent): void {
    this._lastEvent = event;
    this._recalledSessions.delete(sessionId);
  }

  /** Health check with short TTL cache (15s). */
  private async cachedHealthy(): Promise<boolean> {
    const now = Date.now();
    if (this._healthCache && now - this._healthCache.ts < GraphitiContextEngine.HEALTH_CACHE_TTL_MS) {
      return this._healthCache.result;
    }
    const result = await this.client.healthy();
    this._healthCache = { result, ts: now };
    return result;
  }

  // ========================================================================
  // Required methods
  // ========================================================================

  /**
   * Ingest a single message into the knowledge graph.
   */
  async ingest(params: {
    sessionId: string;
    message: { role: string; content: unknown };
    isHeartbeat?: boolean;
  }): Promise<IngestResult> {
    if (this.cfg.autoCapture === false) {
      this.debugLog.log("ce-ingest", { skipped: true, reason: "autoCapture_disabled" });
      return { ingested: false };
    }

    if (params.isHeartbeat) {
      this.debugLog.log("ce-ingest", { skipped: true, reason: "heartbeat" });
      return { ingested: false };
    }

    const { role, content } = params.message;
    if (role !== "user" && role !== "assistant") {
      this.debugLog.log("ce-ingest", { skipped: true, reason: "role", role });
      return { ingested: false };
    }

    const rawText = extractTextContent(content);
    const text = rawText ? sanitizeForCapture(rawText) : null;
    if (!text || text.length < 20) {
      this.debugLog.log("ce-ingest", { skipped: true, reason: text ? "sanitized_short" : "short_content" });
      return { ingested: false };
    }

    try {
      const start = Date.now();
      await this.client.ingest([{
        content: `${role}: ${text.slice(0, 2000)}`,
        role_type: role as "user" | "assistant",
        role,
        name: `ingest-${params.sessionId}-${Date.now()}`,
        timestamp: new Date().toISOString(),
        source_description: buildProvenance(this.groupId, { event: "ingest", session_key: params.sessionId, thread_id: this._threadId ?? undefined }),
      }]);

      this.debugLog.log("ce-ingest", { ingested: true, role, ms: Date.now() - start });
      return { ingested: true };
    } catch (err) {
      this.logger?.warn(`graphiti: ingest failed: ${String(err)}`);
      this.debugLog.log("ce-ingest", { error: String(err) });
      return { ingested: false };
    }
  }

  /**
   * Assemble context for the agent via Smart autoRecall.
   *
   * Only fires when a continuity gap is detected (bootstrap, compaction,
   * few messages) or the user prompt contains deictic references to prior
   * context. Normal turns with sufficient history get no injection.
   *
   * Two-stage pipeline:
   *   A. Continuity recovery — session transcript tail, then episode-based
   *      fallback (filtered by session_key/thread_id provenance)
   *   B. Targeted semantic recall — /search against recovered continuity,
   *      or /get-memory on the current window when no continuity is available
   */
  async assemble(params: {
    sessionId: string;
    messages: unknown[];
    tokenBudget?: number;
    threadId?: string;
    /** The incoming user prompt for this turn (sanctioned fence channel). */
    prompt?: string;
  }): Promise<AssembleResult> {
    const passThrough: AssembleResult = { messages: params.messages };
    const start = Date.now();

    try {
      const healthy = await this.cachedHealthy();
      if (!healthy) {
        this.debugLog.log("ce-assemble", { skipped: true, reason: "unhealthy" });
        return passThrough;
      }

      // Respect opt-in config — engine still registers for capture/compaction
      // but recall injection requires explicit autoRecall: true
      if (!this.cfg.autoRecall) {
        this.debugLog.log("ce-assemble", { skipped: true, reason: "autoRecall_disabled" });
        return passThrough;
      }

      if (params.threadId) this._threadId = params.threadId;

      // Fence: the current turn's user entry arrives via the sanctioned
      // `prompt` channel; only fall back to the window tail when the host
      // does not provide it (non-durable path).
      const lastUserText = params.prompt || this.extractLastUserText(params.messages);
      const gapDetected = isContinuityGap(params.messages.length, { recentEvent: this._lastEvent });
      const deicticDetected = !!lastUserText && hasDeicticReferences(lastUserText);

      // Session-scoped short-circuit: skip recall if this session already received
      // Graphiti context.  Lifecycle events (bootstrap/compact via _lastEvent) and
      // deictic references override the gate so recovery still fires when needed.
      // TODO(#171): if the plugin-sdk exposes effective system-prompt contents or
      // placement control for ContextEngine additions, replace this session-scoped
      // short-circuit with direct Graphiti-tag detection or append-based placement.
      if (this._recalledSessions.has(params.sessionId) && !this._lastEvent && !deicticDetected) {
        this.debugLog.log("ce-assemble", { skipped: true, reason: "already_injected" });
        return passThrough;
      }

      if (!gapDetected && !deicticDetected) {
        this.debugLog.log("ce-assemble", { skipped: true, reason: "no_recovery_needed" });
        return passThrough;
      }

      // Consume the one-shot event flag.  Safe: JS is single-threaded so no
      // concurrent assemble() can read between capture and clear.
      const triggerEvent = this._lastEvent;
      this._lastEvent = null;
      const maxFacts = this.cfg.recallMaxFacts ?? 10;

      // Stage A: recover continuity from session transcript
      let continuityBlock: string | null = null;
      let continuityTail: string | null = null;
      if (this._sessionFile) {
        continuityTail = await readSessionFileTail(this._sessionFile);
        if (continuityTail && params.prompt) {
          continuityTail = this.stripTrailingPrompt(continuityTail, params.prompt);
        }
        if (continuityTail) {
          continuityBlock = formatContinuityBlock(continuityTail);
        }
      }

      // Stage A fallback: episode-based recovery when session file is empty/missing
      if (!continuityTail && this._sessionId) {
        const episodes = await this.client.episodes(20);
        const episodeText = extractEpisodeContinuity(episodes, this._sessionId, {
          threadId: this._threadId ?? undefined,
        });
        if (episodeText) {
          continuityTail = episodeText;
          continuityBlock = formatContinuityBlock(episodeText);
        }
      }

      // Stage B: semantic recall — use recovered continuity as the query
      // so facts are relevant to what was actually discussed, not to the
      // possibly-empty current message window
      let semanticBlock: string | null = null;
      if (continuityTail) {
        const query = continuityTail.slice(-2000);
        const facts = await this.client.search(query, maxFacts);
        if (facts.length > 0) {
          semanticBlock = formatFactsAsContext(facts);
        }
      } else {
        // Fence: drop a trailing current-turn user entry so recall reads
        // only pre-fence transcript.
        let recallWindow = params.messages;
        if (params.prompt) {
          const last = recallWindow[recallWindow.length - 1];
          if (last && typeof last === "object" && (last as { role?: unknown }).role === "user") {
            recallWindow = recallWindow.slice(0, -1);
          }
        }
        const graphitiMessages = this.buildGraphitiMessages(recallWindow);
        if (graphitiMessages.length > 0) {
          const facts = await this.client.getMemory(graphitiMessages, maxFacts);
          if (facts.length > 0) {
            semanticBlock = formatFactsAsContext(facts);
          }
        }
      }

      // --- Combine ---
      const parts = [continuityBlock, semanticBlock].filter(Boolean) as string[];
      if (parts.length === 0) {
        this.debugLog.log("ce-assemble", { recovery: true, trigger: triggerEvent ?? "deictic", empty: true, ms: Date.now() - start });
        return passThrough;
      }

      const systemPromptAddition = parts.join("\n\n");
      const estimatedTokens = Math.ceil(systemPromptAddition.length / 4);
      const trigger = triggerEvent ?? (deicticDetected ? "deictic" : "gap");

      this.logger?.info?.(`graphiti: smart autoRecall fired (trigger=${trigger}, continuity=${!!continuityBlock}, facts=${semanticBlock ? "yes" : "none"})`);
      this.debugLog.log("ce-assemble", {
        recovery: true,
        trigger,
        hasContinuity: !!continuityBlock,
        factCount: semanticBlock ? (semanticBlock.match(/^- \*\*/gm)?.length ?? 0) : 0,
        tokens: estimatedTokens,
        ms: Date.now() - start,
      });

      if (this._recalledSessions.size >= 1000) {
        this.debugLog.log("ce-assemble", { action: "recalled_sessions_reset", previousSize: this._recalledSessions.size });
        this._recalledSessions.clear();
      }
      this._recalledSessions.add(params.sessionId);
      return { messages: params.messages, systemPromptAddition, estimatedTokens };
    } catch (err) {
      this.logger?.warn(`graphiti: assemble failed: ${String(err)}`);
      this.debugLog.log("ce-assemble", { error: String(err), ms: Date.now() - start });
      return passThrough;
    }
  }

  /** Convert raw messages to GraphitiMessage format for /get-memory. */
  private buildGraphitiMessages(messages: unknown[]): GraphitiMessage[] {
    const recentMessages = messages.slice(-10);
    const graphitiMessages: GraphitiMessage[] = [];

    for (const msg of recentMessages) {
      if (!msg || typeof msg !== "object") continue;
      const m = msg as Record<string, unknown>;
      const role = m.role as string;
      if (role !== "user" && role !== "assistant") continue;

      const text = extractTextContent(m.content, 1);
      if (!text) continue;

      graphitiMessages.push({
        content: text.slice(0, 2000),
        role_type: role as "user" | "assistant",
        role,
      });
    }

    return graphitiMessages;
  }

  /** Extract the text of the last user message for deictic reference detection. */
  private extractLastUserText(messages: unknown[]): string | null {
    for (let i = messages.length - 1; i >= 0; i--) {
      const msg = messages[i];
      if (!msg || typeof msg !== "object") continue;
      const m = msg as Record<string, unknown>;
      if (m.role !== "user") continue;
      return extractTextContent(m.content, 1);
    }
    return null;
  }

  /**
   * Fence helper: remove a trailing occurrence of the current turn's prompt
   * text from a session-file tail so continuity recovery never reads past
   * the current-turn entry. No-op when the tail does not contain the prompt.
   */
  private stripTrailingPrompt(tail: string, prompt: string): string {
    if (!prompt) return tail;
    const idx = tail.lastIndexOf(prompt);
    if (idx === -1) return tail;
    return tail.slice(0, idx).replace(/\s+$/, "");
  }

  /**
   * Graph-aware compaction: ingest messages being compacted, then signal
   * the runtime that it's safe to truncate. The graph holds long-term memory.
   */
  async compact(params: {
    sessionId: string;
    sessionFile?: string;
    tokenBudget?: number;
    force?: boolean;
    messages?: unknown[];
    customInstructions?: string; // reserved for future runtimeContext.compactBuiltIn callback
    runtimeContext?: Record<string, unknown>; // reserved for future runtimeContext.compactBuiltIn callback
    legacyParams?: { bridge?: { compact: () => Promise<any> } };
  }): Promise<CompactResult> {
    try {
      if (params.messages) {
        const healthy = await this.cachedHealthy();
        if (!healthy) {
          if (params.legacyParams?.bridge) {
            this.debugLog.log("ce-compact", { action: "legacy_bridge_fallback" });
            await params.legacyParams.bridge.compact();
            return { ok: true, compacted: true };
          }
          this.debugLog.log("ce-compact", { action: "unhealthy", compacted: false });
          return { ok: true, compacted: false, reason: "server-unhealthy" };
        }

        const texts = extractTextsFromMessages(params.messages);
        if (texts.length === 0) {
          // No user/assistant text to preserve — safe to signal compaction complete.
          // The graph already holds any knowledge from prior ingestion cycles.
          this.debugLog.log("ce-compact", { action: "no_text", compacted: true });
          return { ok: true, compacted: true };
        }

        const episode = sanitizeForCapture(texts.join("\n\n")).slice(0, 12000);
        await this.client.ingest([{
          content: episode,
          role_type: "user",
          role: "conversation",
          name: buildEpisodeName("compact", { sessionKey: params.sessionId }),
          timestamp: new Date().toISOString(),
          source_description: buildProvenance(this.groupId, { event: "compact", session_key: params.sessionId, thread_id: this._threadId ?? undefined }),
        }]);

        this.signalRecovery(params.sessionId, "compact");
        this.debugLog.log("ce-compact", { action: "ingested", texts: texts.length, compacted: true });
        return { ok: true, compacted: true };
      }

      if (params.legacyParams?.bridge) {
        this.signalRecovery(params.sessionId, "compact");
        this.debugLog.log("ce-compact", { action: "legacy_bridge" });
        await params.legacyParams.bridge.compact();
        return { ok: true, compacted: true };
      }

      this.debugLog.log("ce-compact", { action: "no_messages", compacted: false });
      return { ok: true, compacted: false, reason: "no-messages" };
    } catch (err) {
      this.logger?.warn(`graphiti: compact failed: ${String(err)}`);
      this.debugLog.log("ce-compact", { error: String(err) });
      return { ok: false, compacted: false, reason: "error" };
    }
  }

  // ========================================================================
  // Optional methods
  // ========================================================================

  /**
   * Batch-ingest multiple messages at once.
   */
  async ingestBatch(params: {
    sessionId: string;
    messages: Array<{ role: string; content: unknown }>;
    isHeartbeat?: boolean;
  }): Promise<IngestBatchResult> {
    if (this.cfg.autoCapture === false) {
      this.debugLog.log("ce-ingestBatch", { skipped: true, reason: "autoCapture_disabled" });
      return { ingestedCount: 0 };
    }

    if (params.isHeartbeat) {
      this.debugLog.log("ce-ingestBatch", { skipped: true, reason: "heartbeat" });
      return { ingestedCount: 0 };
    }

    const rawTexts = extractTextsFromMessages(params.messages);
    const texts = rawTexts.map(sanitizeForCapture).filter(t => t.length > 0);
    if (texts.length === 0) {
      this.debugLog.log("ce-ingestBatch", { skipped: true, reason: "no_content" });
      return { ingestedCount: 0 };
    }

    try {
      const start = Date.now();
      const episode = texts.join("\n\n").slice(0, 12000);

      await this.client.ingest([{
        content: episode,
        role_type: "user",
        role: "conversation",
        name: buildEpisodeName("batch", { sessionKey: params.sessionId }),
        timestamp: new Date().toISOString(),
        source_description: buildProvenance(this.groupId, {
          event: "ingest_batch",
          session_key: params.sessionId,
          thread_id: this._threadId ?? undefined,
        }),
      }]);

      this.debugLog.log("ce-ingestBatch", { count: texts.length, ms: Date.now() - start });
      return { ingestedCount: texts.length };
    } catch (err) {
      this.logger?.warn(`graphiti: ingestBatch failed: ${String(err)}`);
      this.debugLog.log("ce-ingestBatch", { error: String(err) });
      return { ingestedCount: 0 };
    }
  }

  /**
   * After-turn hook: ingest new messages from this turn.
   * When implemented, runtime does NOT call ingestBatch/ingest fallback.
   */
  async afterTurn(params: {
    sessionId: string;
    sessionFile?: string;
    messages: Array<{ role: string; content: unknown }>;
    prePromptMessageCount: number;
    isHeartbeat?: boolean;
    tokenBudget?: number;
  }): Promise<void> {
    // Keep session state fresh for Smart autoRecall
    if (params.sessionFile) this._sessionFile = params.sessionFile;
    if (params.sessionId) this._sessionId = params.sessionId;

    // Durable-commit hosts capture turns in commitTurn (exactly-once per
    // advancementKey); afterTurn remains the capture point for non-durable
    // hosts. Boundary note: the first durable turn may already have been
    // captured here before the first commitTurn flips _durableMode.
    if (this._durableMode) {
      this.debugLog.log("ce-afterTurn", { skipped: true, reason: "durable_commit_turn" });
      return;
    }

    if (this.cfg.autoCapture === false) {
      this.debugLog.log("ce-afterTurn", { skipped: true, reason: "autoCapture_disabled" });
      return;
    }

    if (params.isHeartbeat) {
      this.debugLog.log("ce-afterTurn", { skipped: true, reason: "heartbeat" });
      return;
    }

    // When Pi auto-compaction runs during a prompt, prePromptMessageCount
    // reflects the pre-compaction count while messages is post-compaction.
    // Detect this mismatch and sweep all messages to avoid losing the
    // current turn's content from the graph.
    // Note: when compaction removes exactly `turnMessageCount` messages,
    // prePromptMessageCount === messages.length and this check is false.
    // Using >= would cause false-positive sweeps on empty turns. The rare
    // missed content is captured on the next afterTurn cycle.
    const compactionOccurred = params.prePromptMessageCount > params.messages.length;
    // Sweep may re-ingest messages already captured in a prior afterTurn call.
    // Intentional: preventing data loss is more important than deduplication here.
    // Use the "after_turn_sweep" provenance event to identify and filter these.
    const newMessages = compactionOccurred
      ? params.messages
      : params.messages.slice(params.prePromptMessageCount);

    if (newMessages.length === 0) {
      this.debugLog.log("ce-afterTurn", { skipped: true, reason: "no_new_messages" });
      return;
    }

    const texts = extractTextsFromMessages(newMessages);
    if (texts.length === 0) {
      this.debugLog.log("ce-afterTurn", { skipped: true, reason: "no_content" });
      return;
    }

    const event = compactionOccurred ? "after_turn_sweep" : "after_turn";

    try {
      const start = Date.now();
      const joined = sanitizeForCapture(texts.join("\n\n"));
      if (!joined) {
        this.debugLog.log("ce-afterTurn", { skipped: true, reason: "sanitized_empty" });
        return;
      }
      const episode = compactionOccurred
        ? joined.slice(-12000)   // sweep: keep tail (newest messages)
        : joined.slice(0, 12000);

      await this.client.ingest([{
        content: episode,
        role_type: "user",
        role: "conversation",
        name: buildEpisodeName("turn", { sessionKey: params.sessionId }),
        timestamp: new Date().toISOString(),
        source_description: buildProvenance(this.groupId, {
          event,
          session_key: params.sessionId,
          thread_id: this._threadId ?? undefined,
        }),
      }]);

      this.logger?.info?.(`graphiti: after-turn ingested ${texts.length} messages`);
      this.debugLog.log("ce-afterTurn", { count: texts.length, sweep: compactionOccurred, ms: Date.now() - start });

      // Auto-compaction truncated the message window — clear the session
      // recall flag so the next assemble() fires recovery, mirroring the
      // explicit compact() method.
      if (compactionOccurred) {
        this.signalRecovery(params.sessionId, "compact");
      }
    } catch (err) {
      this.logger?.warn(`graphiti: afterTurn failed: ${String(err)}`);
      this.debugLog.log("ce-afterTurn", { error: String(err) });
    }
  }

  /**
   * Durable turn advancement (durable-commit hosts only).
   *
   * Contract: atomically and idempotently commit one accepted durable
   * transcript turn. `messages` span the admitted user entry through the
   * accepted terminal entry; hosts may retry the same `advancementKey`
   * after process or plugin failure.
   *
   * Idempotency is layered:
   *   1. In-process: bounded Set of advancement keys
   *   2. Cross-restart: deterministic episode name (turn-commit-<key>)
   *      checked against the most recent 100 episodes before capture
   *
   * Capture errors propagate (and the key is NOT recorded as committed)
   * so the host outbox retries the advancement; no-content / no-capture
   * turns ack "committed" without touching the graph.
   */
  async commitTurn(params: {
    advancementKey: string;
    admission?: unknown;
    terminal?: unknown;
    messages: Array<{ role: string; content: unknown }>;
    sessionId: string;
    sessionKey?: string;
    isHeartbeat?: boolean;
  }): Promise<CommitTurnResult> {
    const key = params.advancementKey;
    this._durableMode = true;

    if (this._committedTurns.has(key)) {
      this.debugLog.log("ce-commitTurn", { key, status: "duplicate", reason: "in_process" });
      return { status: "duplicate" };
    }

    const episodeName = `turn-commit-` + key;
    try {
      const recent = await this.client.episodes(100);
      if (recent.some((ep) => ep.name === episodeName)) {
        this._committedTurns.add(key);
        this.debugLog.log("ce-commitTurn", { key, status: "duplicate", reason: "episode_exists" });
        return { status: "duplicate" };
      }
    } catch (err) {
      // Episode scan is best-effort dedup; fall through to capture.
      this.debugLog.log("ce-commitTurn", { key, episodeScan: "failed", error: String(err) });
    }

    if (this.cfg.autoCapture === false || params.isHeartbeat) {
      this._committedTurns.add(key);
      this.debugLog.log("ce-commitTurn", {
        key,
        status: "committed",
        captured: false,
        reason: params.isHeartbeat ? "heartbeat" : "autoCapture_disabled",
      });
      return { status: "committed" };
    }

    const texts = extractTextsFromMessages(params.messages);
    const joined = texts.length > 0 ? sanitizeForCapture(texts.join("\n\n")) : null;
    if (!joined) {
      this._committedTurns.add(key);
      this.debugLog.log("ce-commitTurn", { key, status: "committed", captured: false, reason: "no_content" });
      return { status: "committed" };
    }

    // Errors must propagate: the host re-queues failed advancements.
    await this.client.ingest([{
      content: joined.slice(0, 12000),
      role_type: "user",
      role: "conversation",
      name: episodeName,
      timestamp: new Date().toISOString(),
      source_description: buildProvenance(this.groupId, {
        event: "commit_turn",
        session_key: params.sessionKey ?? params.sessionId,
        thread_id: this._threadId ?? undefined,
      }),
    }]);

    if (this._committedTurns.size >= 5000) this._committedTurns.clear();
    this._committedTurns.add(key);
    this.debugLog.log("ce-commitTurn", { key, status: "committed", captured: true, chars: joined.length });
    return { status: "committed" };
  }

  /**
   * Bootstrap: health-check the server, report graph population,
   * and prime Smart autoRecall state for continuity recovery.
   */
  async bootstrap(params: {
    sessionId: string;
    sessionFile?: string;
    threadId?: string;
  }): Promise<BootstrapResult> {
    try {
      this._sessionFile = params.sessionFile ?? null;
      this._sessionId = params.sessionId;
      this._threadId = params.threadId ?? this._threadId;

      const healthy = await this.cachedHealthy();
      if (!healthy) {
        this.debugLog.log("ce-bootstrap", { healthy: false });
        return { bootstrapped: false, reason: "server-unhealthy" };
      }

      this.signalRecovery(params.sessionId, "bootstrap");

      const stats = await this.client.episodeCount();
      this.debugLog.log("ce-bootstrap", { healthy: true, episodes: stats.count });
      return { bootstrapped: true, episodeCount: stats.count };
    } catch (err) {
      this.logger?.warn(`graphiti: bootstrap failed: ${String(err)}`);
      this.debugLog.log("ce-bootstrap", { error: String(err) });
      return { bootstrapped: false, reason: "error" };
    }
  }

  /**
   * Called when a subagent finishes. Ingests its findings into the parent's graph scope.
   */
  async onSubagentEnded(params: {
    childSessionKey: string;
    reason: string;
    summary?: string;
    messages?: unknown[];
  }): Promise<void> {
    const hasSummary = typeof params.summary === "string" && params.summary.length >= 20;

    if (!hasSummary && (!params.messages || params.messages.length === 0)) {
      this.debugLog.log("ce-subagentEnded", {
        child: params.childSessionKey,
        reason: params.reason,
        action: "no-op",
      });
      return;
    }

    try {
      const healthy = await this.cachedHealthy();
      if (!healthy) {
        this.debugLog.log("ce-subagentEnded", { child: params.childSessionKey, skipped: true, reason: "unhealthy" });
        return;
      }

      let episode: string;
      if (hasSummary) {
        episode = sanitizeForCapture(params.summary!).slice(0, 12000);
      } else {
        const texts = extractTextsFromMessages(params.messages!);
        if (texts.length === 0) {
          this.debugLog.log("ce-subagentEnded", { child: params.childSessionKey, action: "no_content" });
          return;
        }
        episode = sanitizeForCapture(texts.join("\n\n")).slice(0, 12000);
      }

      await this.client.ingest([{
        content: episode,
        // "system" is a valid GraphitiMessage.role_type — subagent results are
        // neither user nor assistant from the parent conversation's perspective.
        role_type: "system",
        role: "subagent-result",
        name: buildEpisodeName("subagent", { sessionKey: params.childSessionKey }),
        timestamp: new Date().toISOString(),
        source_description: buildProvenance(this.groupId, {
          event: "subagent_ended",
          session_key: params.childSessionKey,
          thread_id: this._threadId ?? undefined,
        }),
      }]);

      this.debugLog.log("ce-subagentEnded", {
        child: params.childSessionKey,
        action: hasSummary ? "ingested_summary" : "ingested_messages",
      });
    } catch (err) {
      this.logger?.warn(`graphiti: onSubagentEnded failed: ${String(err)}`);
      this.debugLog.log("ce-subagentEnded", { child: params.childSessionKey, error: String(err) });
    }
  }

  /**
   * Prepare context for a spawning subagent by injecting relevant facts.
   */
  async prepareSubagentSpawn(params: {
    parentSessionKey: string;
    childSessionKey: string;
    ttlMs?: number;
    taskDescription?: string;
  }): Promise<SubagentSpawnPreparation | undefined> {
    if (!params.taskDescription) return undefined;

    try {
      const healthy = await this.cachedHealthy();
      if (!healthy) return undefined;

      const maxFacts = this.cfg.recallMaxFacts ?? 10;
      const facts = await this.client.search(params.taskDescription, maxFacts);
      if (facts.length === 0) return undefined;

      return { systemPromptAddition: formatFactsAsContext(facts) };
    } catch (err) {
      this.logger?.warn(`graphiti: prepareSubagentSpawn failed: ${String(err)}`);
      this.debugLog.log("ce-prepareSubagentSpawn", { error: String(err) });
      return undefined;
    }
  }

  /**
   * Dispose: no-op (no persistent connections).
   */
  dispose(): void {
    // No persistent connections to clean up
  }
}
