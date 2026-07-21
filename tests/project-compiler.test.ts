/**
 * Tests for project-compiler.ts — pure functions for compilation verification.
 */

import { describe, expect, it, vi } from "vitest";
import {
  cleanupTempDir,
  createTempDir,
  detectProjectType,
  extractAndWriteCodeBlocks,
  type CompilationResult,
} from "../src/core/project-compiler.js";
import { existsSync, mkdirSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

// ------------------------------------------------------------------
// detectProjectType
// ------------------------------------------------------------------

describe("detectProjectType", () => {
  it("detects TypeScript project when tsconfig.json exists", () => {
    // The turbocontext project itself has a tsconfig.json
    const config = detectProjectType(process.cwd());
    expect(config.projectType).toBe("typescript");
    expect(config.hasConfigFile).toBe(true);
    expect(config.tsconfigPath).toBeTruthy();
  });

  it("detects unknown when no config files exist", () => {
    const config = detectProjectType(tmpdir());
    expect(config.hasConfigFile).toBe(false);
    // tmpdir likely has no tsconfig, but if it does (edge case), still fine
    if (!config.hasConfigFile) {
      expect(config.projectType).toBe("unknown");
    }
  });

  it("detects TypeScript in a temp dir with tsconfig.json", () => {
    const dir = createTempDir("detect-test");
    writeFileSync(join(dir, "tsconfig.json"), "{}");
    writeFileSync(join(dir, "package.json"), "{}");

    const config = detectProjectType(dir);
    expect(config.projectType).toBe("typescript");
    expect(config.hasConfigFile).toBe(true);
    expect(config.tsconfigPath).toBe(join(dir, "tsconfig.json"));
    expect(config.packageJsonPath).toBe(join(dir, "package.json"));

    cleanupTempDir(dir);
  });

  it("detects JavaScript when only package.json exists", () => {
    const dir = createTempDir("detect-js");
    writeFileSync(join(dir, "package.json"), "{}");

    const config = detectProjectType(dir);
    expect(config.projectType).toBe("javascript");
    expect(config.hasConfigFile).toBe(true);

    cleanupTempDir(dir);
  });
});

// ------------------------------------------------------------------
// createTempDir / cleanupTempDir
// ------------------------------------------------------------------

describe("createTempDir and cleanupTempDir", () => {
  it("creates a directory under ~/.turbocontext/tmp/", () => {
    const dir = createTempDir("test");
    expect(dir).toContain("turbocontext");
    expect(dir).toContain("test");
    expect(existsSync(dir)).toBe(true);
    cleanupTempDir(dir);
    expect(existsSync(dir)).toBe(false);
  });

  it("cleanupTempDir handles non-existent dirs gracefully", () => {
    // Should not throw
    expect(() => cleanupTempDir("/tmp/nonexistent-dir-abc-xyz-123")).not.toThrow();
  });
});

// ------------------------------------------------------------------
// extractAndWriteCodeBlocks
// ------------------------------------------------------------------

describe("extractAndWriteCodeBlocks", () => {
  it("extracts TypeScript code blocks from markdown output", () => {
    const dir = createTempDir("extract-test");
    const output = [
      "Here is the login function:",
      "",
      "```typescript",
      "export function login(username: string, password: string): boolean {",
      "  return true;",
      "}",
      "```",
      "",
      "And here is the test:",
      "```ts",
      "import { describe, it } from 'vitest';",
      "describe('login', () => { it('works', () => {}); });",
      "```",
    ].join("\n");

    const entries = extractAndWriteCodeBlocks(output, dir);

    expect(entries.length).toBe(2);
    expect(entries[0].language).toBe("typescript");
    expect(entries[1].language).toBe("ts");
    expect(existsSync(entries[0].filePath)).toBe(true);
    expect(existsSync(entries[1].filePath)).toBe(true);

    // Verify content was written correctly
    const fs = require("node:fs");
    const content0 = fs.readFileSync(entries[0].filePath, "utf-8");
    expect(content0).toContain("export function login");

    cleanupTempDir(dir);
  });

  it("uses filename hints from code comments", () => {
    const dir = createTempDir("extract-hint");
    const output = [
      "```typescript",
      "// src/auth/login.ts",
      "export function login() { return true; }",
      "```",
    ].join("\n");

    const entries = extractAndWriteCodeBlocks(output, dir);

    expect(entries.length).toBe(1);
    expect(entries[0].filePath).toContain("auth/login.ts");

    cleanupTempDir(dir);
  });

  it("returns empty array when no code blocks found", () => {
    const dir = createTempDir("extract-empty");
    const entries = extractAndWriteCodeBlocks("Just some text, no code blocks.", dir);
    expect(entries).toEqual([]);
    cleanupTempDir(dir);
  });

  it("skips empty code blocks", () => {
    const dir = createTempDir("extract-skip");
    const output = [
      "```typescript",
      "",
      "```",
      "```ts",
      "const x = 1;",
      "```",
    ].join("\n");

    const entries = extractAndWriteCodeBlocks(output, dir);
    expect(entries.length).toBe(1); // empty block skipped
    cleanupTempDir(dir);
  });
});

// ------------------------------------------------------------------
// compileProject (with mock process runner)
// ------------------------------------------------------------------

describe("compileProject with mock", () => {
  it("returns compiled=true when process exits 0", async () => {
    const dir = createTempDir("compile-mock-ok");
    // Write a valid TypeScript file
    writeFileSync(
      join(dir, "test.ts"),
      "export const x: number = 1;",
    );

    const mockRunner = vi.fn().mockResolvedValue({
      stdout: "",
      stderr: "",
      exitCode: 0,
      timedOut: false,
    });

    // Direct test of the runner pattern
    const { compileTypeScript } = await import("../src/core/project-compiler.js");
    // We need to create the tsconfig first for this to work
    const result = await compileTypeScript(dir, 5000, mockRunner);

    expect(result.compiled).toBe(true);
    expect(result.errors).toEqual([]);
    expect(result.durationMs).toBeGreaterThanOrEqual(0);

    cleanupTempDir(dir);
  });

  it("returns compiled=false when tsc reports errors", async () => {
    const dir = createTempDir("compile-mock-fail");
    writeFileSync(
      join(dir, "bad.ts"),
      "const x: number = 'string';", // type error
    );

    const mockRunner = vi.fn().mockResolvedValue({
      stdout: "bad.ts(1,7): error TS2322: Type 'string' is not assignable to type 'number'.",
      stderr: "",
      exitCode: 2,
      timedOut: false,
    });

    const { compileTypeScript } = await import("../src/core/project-compiler.js");
    const result = await compileTypeScript(dir, 5000, mockRunner);

    expect(result.compiled).toBe(false);
    expect(result.errors.length).toBeGreaterThan(0);
    expect(result.errors[0]).toContain("TS2322");

    cleanupTempDir(dir);
  });

  it("handles ENOENT (command not found) gracefully", async () => {
    const dir = createTempDir("compile-mock-enoent");
    writeFileSync(join(dir, "test.ts"), "const x = 1;");

    const mockRunner = vi.fn().mockResolvedValue({
      stdout: "",
      stderr: "npx: command not found",
      exitCode: null,
      timedOut: false,
    });

    const { compileTypeScript } = await import("../src/core/project-compiler.js");
    // Should not throw
    const result = await compileTypeScript(dir, 5000, mockRunner);
    expect(result).toBeDefined();

    cleanupTempDir(dir);
  });
});

// ------------------------------------------------------------------
// smokeTestTypeScript
// ------------------------------------------------------------------

describe("smokeTestTypeScript", () => {
  it("passes for valid, callable TypeScript code", async () => {
    const { smokeTestTypeScript } = await import("../src/core/project-compiler.js");
    const dir = createTempDir("smoke-ok");

    // Write valid TS that exports a function
    writeFileSync(
      join(dir, "hello.ts"),
      "export function greet(name: string): string { return 'Hello, ' + name; }",
    );

    // We need a mock runner for tsx
    const mockRunner = vi.fn().mockResolvedValue({
      stdout: "PASS:greet\nSMOKE_RESULT:1:0",
      stderr: "",
      exitCode: 0,
      timedOut: false,
    });

    const result = await smokeTestTypeScript(dir, 5000, mockRunner);

    expect(result.passed).toBe(true);
    expect(result.functionsCalled).toBe(1);
    expect(result.functionsFailed).toBe(0);
    expect(result.errors).toEqual([]);

    cleanupTempDir(dir);
  });

  it("fails when exported function throws", async () => {
    const { smokeTestTypeScript } = await import("../src/core/project-compiler.js");
    const dir = createTempDir("smoke-fail");

    writeFileSync(
      join(dir, "broken.ts"),
      "export function crash(): never { throw new Error('boom'); }",
    );

    const mockRunner = vi.fn().mockResolvedValue({
      stdout: "FAIL:crash:boom\nSMOKE_RESULT:0:1",
      stderr: "",
      exitCode: 1,
      timedOut: false,
    });

    const result = await smokeTestTypeScript(dir, 5000, mockRunner);

    expect(result.passed).toBe(false);
    expect(result.functionsFailed).toBe(1);
    expect(result.errors.length).toBeGreaterThan(0);
    expect(result.errors[0]).toContain("crash");

    cleanupTempDir(dir);
  });

  it("skips gracefully when no TS files found", async () => {
    const { smokeTestTypeScript } = await import("../src/core/project-compiler.js");
    const dir = createTempDir("smoke-empty");

    // No .ts files in dir — only tsconfig
    writeFileSync(join(dir, "tsconfig.json"), "{}");

    const mockRunner = vi.fn();
    const result = await smokeTestTypeScript(dir, 5000, mockRunner);

    expect(result.passed).toBe(true);
    expect(result.functionsCalled).toBe(0);
    expect(mockRunner).not.toHaveBeenCalled();

    cleanupTempDir(dir);
  });

  it("handles process runner crash gracefully", async () => {
    const { smokeTestTypeScript } = await import("../src/core/project-compiler.js");
    const dir = createTempDir("smoke-crash");

    writeFileSync(
      join(dir, "test.ts"),
      "export const x = 1;",
    );

    const mockRunner = vi.fn().mockRejectedValue(new Error("tsx killed"));
    const result = await smokeTestTypeScript(dir, 5000, mockRunner);

    expect(result.passed).toBe(false);
    expect(result.errors.length).toBeGreaterThan(0);

    cleanupTempDir(dir);
  });
});
