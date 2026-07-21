#!/usr/bin/env node
// ============================================================
// TurboContext MCP Server — Session Memory for Claude Code
// ============================================================
// Exposes three tools via stdio transport:
//   turbocontext_recall  — find similar past sessions
//   turbocontext_record  — save completed session
//   turbocontext_status  — corpus statistics
// ============================================================

import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import {
  CallToolRequestSchema,
  ListToolsRequestSchema,
} from "@modelcontextprotocol/sdk/types.js";
import { SessionMemory } from "./core/session-memory.js";

const TASK_TYPES = [
  "code_review", "code_generation", "code_refactor",
  "debugging", "testing", "analysis", "design",
  "documentation", "general",
] as const;

// ── Server setup ──

const memory = SessionMemory.load();

const server = new Server(
  { name: "turbocontext", version: "2.0.0" },
  { capabilities: { tools: {} } },
);

// ── Tool: list ──

server.setRequestHandler(ListToolsRequestSchema, async () => ({
  tools: [
    {
      name: "turbocontext_recall",
      description:
        "Find similar past sessions and get file/strategy recommendations. " +
        "Call this BEFORE starting work on a task to get context from past sessions.",
      inputSchema: {
        type: "object",
        properties: {
          taskDescription: {
            type: "string",
            description: "The task the user wants to accomplish",
          },
          workingDirectory: {
            type: "string",
            description: "Current working directory (project root path)",
          },
          taskType: {
            type: "string",
            enum: [...TASK_TYPES],
            description: "Optional task type hint for better matching",
          },
          maxResults: {
            type: "number",
            default: 5,
            description: "Maximum number of similar sessions to return",
          },
        },
        required: ["taskDescription", "workingDirectory"],
      },
    },
    {
      name: "turbocontext_record",
      description:
        "Record a completed session for future recall. " +
        "Call this AFTER finishing a task to save what worked.",
      inputSchema: {
        type: "object",
        properties: {
          taskDescription: {
            type: "string",
            description: "The task that was completed",
          },
          taskType: {
            type: "string",
            enum: [...TASK_TYPES],
            description: "Task type classification",
          },
          workingDirectory: {
            type: "string",
            description: "Project root directory",
          },
          filesRead: {
            type: "array",
            items: { type: "string" },
            description: "Absolute paths of files that were read",
          },
          filesModified: {
            type: "array",
            items: { type: "string" },
            description: "Absolute paths of files that were modified",
          },
          strategy: {
            type: "string",
            description: "1-3 sentence description of the approach taken",
          },
          outcome: {
            type: "string",
            enum: ["success", "partial", "failure"],
            description: "How the task turned out",
          },
          selfAssessment: {
            type: "number",
            description:
              "0-1 self-assessed quality. 0.9-1.0=excellent, 0.7-0.9=good, " +
              "0.5-0.7=acceptable, 0-0.5=incomplete",
          },
          notes: {
            type: "string",
            description: "What went well, what to do differently next time",
          },
          roundCount: {
            type: "number",
            description: "Number of understand→execute→verify rounds used",
          },
        },
        required: ["taskDescription", "taskType", "workingDirectory", "outcome", "selfAssessment"],
      },
    },
    {
      name: "turbocontext_status",
      description: "Get statistics about the session memory corpus.",
      inputSchema: {
        type: "object",
        properties: {},
      },
    },
  ],
}));

// ── Tool: call ──

server.setRequestHandler(CallToolRequestSchema, async (request) => {
  const { name, arguments: args } = request.params;

  switch (name) {
    case "turbocontext_recall": {
      const result = memory.recall({
        taskDescription: (args as any).taskDescription,
        workingDirectory: (args as any).workingDirectory,
        taskType: (args as any).taskType,
        maxResults: (args as any).maxResults ?? 5,
      });

      return {
        content: [
          {
            type: "text",
            text: JSON.stringify(result, null, 2),
          },
        ],
      };
    }

    case "turbocontext_record": {
      const a = args as any;
      const record = memory.record({
        taskDescription: a.taskDescription,
        taskType: a.taskType,
        workingDirectory: a.workingDirectory,
        filesRead: a.filesRead ?? [],
        filesModified: a.filesModified ?? [],
        strategy: a.strategy ?? "",
        outcome: a.outcome,
        selfAssessment: a.selfAssessment,
        notes: a.notes ?? "",
        roundCount: a.roundCount ?? 1,
      });

      const stats = memory.stats();
      return {
        content: [
          {
            type: "text",
            text: JSON.stringify({
              recorded: { id: record.id, timestamp: record.timestamp },
              corpusStats: stats,
            }, null, 2),
          },
        ],
      };
    }

    case "turbocontext_status": {
      const stats = memory.stats();
      return {
        content: [
          {
            type: "text",
            text: JSON.stringify(stats, null, 2),
          },
        ],
      };
    }

    default:
      throw new Error(`Unknown tool: ${name}`);
  }
});

// ── Start ──

async function main() {
  const transport = new StdioServerTransport();
  await server.connect(transport);
  const stats = memory.stats();
  console.error(
    `turbocontext MCP server ready: ${stats.totalSessions} sessions, ` +
    `${stats.totalUniqueFiles} unique files across ${Object.keys(stats.perTaskType).length} task types`,
  );
}

main().catch((err) => {
  console.error("turbocontext MCP server failed:", err);
  process.exit(1);
});
