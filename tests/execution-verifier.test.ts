/**
 * Tests for execution-verifier.ts — ExecutionCodeVerifier.
 *
 * Tests cover:
 * - Backward compatibility (no context → same as CodeVerifier)
 * - Fallback when no project config
 * - Compilation success / failure signals
 * - Verifier crash handling
 * - Metrics propagation
 */

import { describe, expect, it, vi } from "vitest";
import { ExecutionCodeVerifier } from "../src/core/execution-verifier.js";
import { CodeVerifier } from "../src/core/verifier.js";
import {
  cleanupTempDir,
  createTempDir,
} from "../src/core/project-compiler.js";
import { writeFileSync } from "node:fs";
import { join } from "node:path";

function makeTask(overrides: Partial<import("../src/types.js").Task> = {}) {
  return {
    id: "t1",
    description: "write a login function in TypeScript",
    type: "code_generation" as const,
    ...overrides,
  };
}

// ------------------------------------------------------------------
// Backward compatibility
// ------------------------------------------------------------------

describe("ExecutionCodeVerifier — Backward Compatibility", () => {
  it("without context, returns same result as CodeVerifier", async () => {
    const execVerifier = new ExecutionCodeVerifier();
    const codeVerifier = new CodeVerifier();

    const goodOutput = [
      "```typescript",
      "export function login(username: string, password: string): boolean {",
      "  return username === 'admin' && password === 'secret';",
      "}",
      "```",
    ].join("\n");

    const task = makeTask();
    const execResult = await execVerifier.verify(goodOutput, task);
    const codeResult = await codeVerifier.verify(goodOutput, task);

    // Should produce the same structural result when no working dir
    expect(execResult.hardSignal).toBe(codeResult.hardSignal);
    expect(execResult.passed).toBe(codeResult.passed);
    expect(execResult.crashed).toBe(false);
  });

  it("without context, bad code is still flagged", async () => {
    const execVerifier = new ExecutionCodeVerifier();
    const task = makeTask();

    // Code with actual structural issue: empty function body
    const badOutput = [
      "```typescript",
      "export function emptyFunction() {}",
      "```",
    ].join("\n");

    const result = await execVerifier.verify(badOutput, task);
    // The empty definition body check should penalize this
    expect(result.hardSignal).toBeLessThan(1.0);
    expect(result.crashed).toBe(false);
  });
});

// ------------------------------------------------------------------
// Fallback: no project config
// ------------------------------------------------------------------

describe("ExecutionCodeVerifier — Fallback to Structural", () => {
  it("falls back to structural when workingDir has no tsconfig", async () => {
    const execVerifier = new ExecutionCodeVerifier();
    const task = makeTask();

    const output = [
      "```typescript",
      "export function hello(): string { return 'world'; }",
      "```",
    ].join("\n");

    // Use os.tmpdir() which typically has no tsconfig
    const { tmpdir } = require("node:os");
    const result = await execVerifier.verify(output, task, {
      workingDir: tmpdir(),
    });

    // Should still run structural (CodeVerifier internally)
    expect(result.crashed).toBe(false);
    expect(result.hardSignal).toBeDefined();
    // Details should come from structural check
    expect(result.details.length).toBeGreaterThan(0);
  });
});

// ------------------------------------------------------------------
// Compilation success path (with mock process runner)
// ------------------------------------------------------------------

describe("ExecutionCodeVerifier — Compilation Success", () => {
  it("returns hardSignal +1.0 when compilation and smoke test both pass", async () => {
    // Two calls: first = tsc compilation, second = tsx smoke test
    const mockRunner = vi.fn()
      .mockResolvedValueOnce({
        stdout: "",
        stderr: "",
        exitCode: 0,       // tsc passes
        timedOut: false,
      })
      .mockResolvedValueOnce({
        stdout: "PASS:login\nSMOKE_RESULT:1:0",
        stderr: "",
        exitCode: 0,       // tsx smoke test passes
        timedOut: false,
      });

    const execVerifier = new ExecutionCodeVerifier(mockRunner, 5000);
    const tempDir = createTempDir("exec-verify-test-ok");

    // Write a tsconfig.json so the verifier detects TypeScript
    writeFileSync(join(tempDir, "tsconfig.json"), JSON.stringify({
      compilerOptions: { strict: true, noEmit: true },
      include: ["**/*.ts"],
    }));

    const task = makeTask();
    const output = [
      "```typescript",
      "export function login(username: string, password: string): boolean {",
      "  return true;",
      "}",
      "```",
    ].join("\n");

    const result = await execVerifier.verify(output, task, {
      workingDir: tempDir,
    });

    expect(result.passed).toBe(true);
    expect(result.crashed).toBe(false);
    expect(result.hardSignal).toBe(1.0);
    expect(result.details).toContain("Compilation passed");
    expect(result.details).toContain("Smoke test passed");

    // Metrics should include both compile and test results
    expect(result.metrics).toBeDefined();
    expect(result.metrics!.compiled).toBe(1);
    expect(result.metrics!.testsPassed).toBe(1);
    expect(result.metrics!.functionsCalled).toBe(1);
    expect(result.metrics!.errors).toBe(0);

    cleanupTempDir(tempDir);
  });

  it("returns reduced hardSignal when compilation passes but smoke test fails", async () => {
    const mockRunner = vi.fn()
      .mockResolvedValueOnce({
        stdout: "",
        stderr: "",
        exitCode: 0,
        timedOut: false,
      })
      .mockResolvedValueOnce({
        stdout: "FAIL:login:TypeError\nSMOKE_RESULT:0:1",
        stderr: "",
        exitCode: 1,
        timedOut: false,
      });

    const execVerifier = new ExecutionCodeVerifier(mockRunner, 5000);
    const tempDir = createTempDir("exec-verify-test-smokefail");
    writeFileSync(join(tempDir, "tsconfig.json"), JSON.stringify({
      compilerOptions: { strict: true, noEmit: true },
      include: ["**/*.ts"],
    }));

    const task = makeTask();
    const output = [
      "```typescript",
      "export function login(): boolean { throw new Error('nope'); }",
      "```",
    ].join("\n");

    const result = await execVerifier.verify(output, task, {
      workingDir: tempDir,
    });

    // Smoke test failure reduces signal
    expect(result.hardSignal).toBe(-0.3);
    expect(result.details).toContain("Smoke test FAILED");
    expect(result.metrics?.testsPassed).toBe(0);
    expect(result.metrics?.functionsFailed).toBe(1);

    cleanupTempDir(tempDir);
  });
});

// ------------------------------------------------------------------
// Compilation failure path
// ------------------------------------------------------------------

describe("ExecutionCodeVerifier — Compilation Failure", () => {
  it("returns hardSignal -0.5 when compilation fails", async () => {
    const mockRunner = vi.fn().mockResolvedValue({
      stdout: "generated_0.ts(1,7): error TS2322: Type 'string' is not assignable to type 'number'.",
      stderr: "",
      exitCode: 2,
      timedOut: false,
    });

    const execVerifier = new ExecutionCodeVerifier(mockRunner, 5000);
    const tempDir = createTempDir("exec-verify-test-fail");
    writeFileSync(join(tempDir, "tsconfig.json"), JSON.stringify({
      compilerOptions: { strict: true, noEmit: true },
      include: ["**/*.ts"],
    }));

    const task = makeTask();
    const output = [
      "```typescript",
      "const x: number = 'oops';",
      "```",
    ].join("\n");

    const result = await execVerifier.verify(output, task, {
      workingDir: tempDir,
    });

    expect(result.passed).toBe(false);
    expect(result.crashed).toBe(false);
    expect(result.hardSignal).toBe(-0.5);
    expect(result.details).toContain("Compilation failed");

    expect(result.metrics).toBeDefined();
    expect(result.metrics!.compiled).toBe(0);
    expect(result.metrics!.errors).toBeGreaterThan(0);

    cleanupTempDir(tempDir);
  });
});

// ------------------------------------------------------------------
// Verifier crash
// ------------------------------------------------------------------

describe("ExecutionCodeVerifier — Crash Handling", () => {
  it("handles process runner rejections gracefully (as compilation failure)", async () => {
    // When the process runner rejects, compileTypeScript catches it and returns
    // a CompilationResult with the error — no verifier crash, just compilation failure
    const mockRunner = vi.fn().mockRejectedValue(new Error("Process killed"));

    const execVerifier = new ExecutionCodeVerifier(mockRunner, 5000);
    const tempDir = createTempDir("exec-verify-test-graceful");
    writeFileSync(join(tempDir, "tsconfig.json"), JSON.stringify({
      compilerOptions: { strict: true, noEmit: true },
      include: ["**/*.ts"],
    }));

    const task = makeTask();
    const output = [
      "```typescript",
      "export const x = 1;",
      "```",
    ].join("\n");

    const result = await execVerifier.verify(output, task, {
      workingDir: tempDir,
    });

    // Process runner rejection is caught by compileTypeScript → non-crash failure
    expect(result.crashed).toBe(false);
    expect(result.hardSignal).toBeLessThanOrEqual(-0.5);
    expect(result.details).toContain("Compilation failed");

    cleanupTempDir(tempDir);
  });

  it("treats 'command not found' as verifier crash", async () => {
    const mockRunner = vi.fn().mockResolvedValue({
      stdout: "",
      stderr: "npx: command not found",
      exitCode: null,
      timedOut: false,
    });

    const execVerifier = new ExecutionCodeVerifier(mockRunner, 5000);
    const tempDir = createTempDir("exec-verify-test-enoent");
    writeFileSync(join(tempDir, "tsconfig.json"), JSON.stringify({
      compilerOptions: { strict: true, noEmit: true },
      include: ["**/*.ts"],
    }));

    const task = makeTask();
    const output = [
      "```typescript",
      "export const x = 1;",
      "```",
    ].join("\n");

    const result = await execVerifier.verify(output, task, {
      workingDir: tempDir,
    });

    expect(result.crashed).toBe(true);
    expect(result.hardSignal).toBe(-1.0);

    cleanupTempDir(tempDir);
  });
});

// ------------------------------------------------------------------
// Language detection
// ------------------------------------------------------------------

describe("ExecutionCodeVerifier — Language Detection", () => {
  it("falls back to structural for non-TS projects", async () => {
    // Don't inject mock runner — won't be called because project is unknown
    const execVerifier = new ExecutionCodeVerifier();
    const tempDir = createTempDir("exec-verify-test-py");
    // Write pyproject.toml to simulate Python project
    writeFileSync(join(tempDir, "pyproject.toml"), "[project]\nname = 'test'");

    const task = makeTask({ type: "code_generation" });
    const output = [
      "```python",
      "def login(username, password):",
      "    return True",
      "```",
    ].join("\n");

    const result = await execVerifier.verify(output, task, {
      workingDir: tempDir,
    });

    // Should fall back to structural (Python compilation not implemented)
    expect(result.crashed).toBe(false);
    expect(result.metrics?.compiled).toBeUndefined(); // No execution metrics

    cleanupTempDir(tempDir);
  });

  it("skips execution when no code blocks in output", async () => {
    const execVerifier = new ExecutionCodeVerifier();
    const tempDir = createTempDir("exec-verify-test-nocode");
    writeFileSync(join(tempDir, "tsconfig.json"), "{}");

    const task = makeTask({ type: "code_generation" });
    const result = await execVerifier.verify("No code here, just text.", task, {
      workingDir: tempDir,
    });

    // Should use structural result (which will flag no code blocks)
    expect(result.crashed).toBe(false);
    expect(result.hardSignal).toBe(-0.5); // CodeVerifier: no code blocks

    cleanupTempDir(tempDir);
  });

  it("falls back when code blocks are non-TypeScript", async () => {
    const execVerifier = new ExecutionCodeVerifier();
    const tempDir = createTempDir("exec-verify-test-js");
    writeFileSync(join(tempDir, "tsconfig.json"), "{}");

    const task = makeTask();
    const output = [
      "```python",
      "def hello():",
      "    return 'world'",
      "```",
    ].join("\n");

    const result = await execVerifier.verify(output, task, {
      workingDir: tempDir,
    });

    // Non-TS code blocks with TS project = structural only
    expect(result.crashed).toBe(false);
    // No execution metrics (non-TS blocks)
    expect(result.metrics?.compiled).toBeUndefined();

    cleanupTempDir(tempDir);
  });
});

// ------------------------------------------------------------------
// Code block extraction from output
// ------------------------------------------------------------------

describe("ExecutionCodeVerifier — Code Block Extraction", () => {
  it("handles multiple code blocks in mixed languages", async () => {
    const mockRunner = vi.fn().mockResolvedValue({
      stdout: "",
      stderr: "",
      exitCode: 0,
      timedOut: false,
    });

    const execVerifier = new ExecutionCodeVerifier(mockRunner, 5000);
    const tempDir = createTempDir("exec-verify-test-multi");
    writeFileSync(join(tempDir, "tsconfig.json"), JSON.stringify({
      compilerOptions: { strict: true, noEmit: true },
      include: ["**/*.ts"],
    }));

    const task = makeTask();
    const output = [
      "```typescript",
      "export function login(): boolean { return true; }",
      "```",
      "Some explanation...",
      "```python",
      "def helper(): pass",
      "```",
    ].join("\n");

    const result = await execVerifier.verify(output, task, {
      workingDir: tempDir,
    });

    // Has TS blocks, so mock runner is called
    expect(mockRunner).toHaveBeenCalled();
    expect(result.passed).toBe(true); // mock returns success

    cleanupTempDir(tempDir);
  });
});

// ------------------------------------------------------------------
// Metrics propagation
// ------------------------------------------------------------------

describe("ExecutionCodeVerifier — Metrics Propagation", () => {
  it("populates structured metrics on compilation success", async () => {
    const mockRunner = vi.fn().mockResolvedValue({
      stdout: "",
      stderr: "",
      exitCode: 0,
      timedOut: false,
    });

    const execVerifier = new ExecutionCodeVerifier(mockRunner, 5000);
    const tempDir = createTempDir("exec-verify-test-metrics");
    writeFileSync(join(tempDir, "tsconfig.json"), JSON.stringify({
      compilerOptions: { strict: true, noEmit: true },
      include: ["**/*.ts"],
    }));

    const task = makeTask();
    const output = [
      "```typescript",
      "export const x = 42;",
      "```",
    ].join("\n");

    const result = await execVerifier.verify(output, task, {
      workingDir: tempDir,
    });

    expect(result.metrics).toBeDefined();
    // Keys should include: compiled, exitCode, errors, warnings, durationMs, projectType, structuralPassed
    const m = result.metrics!;
    expect(m.compiled).toBe(1);
    expect(m.exitCode).toBe(0);
    expect(m.errors).toBe(0);
    expect(typeof m.durationMs).toBe("number");
    expect(m.projectType).toBe(1);

    cleanupTempDir(tempDir);
  });

  it("populates error counts on compilation failure", async () => {
    const mockRunner = vi.fn().mockResolvedValue({
      stdout: "generated_0.ts(1,7): error TS2322: Type mismatch.",
      stderr: "",
      exitCode: 2,
      timedOut: false,
    });

    const execVerifier = new ExecutionCodeVerifier(mockRunner, 5000);
    const tempDir = createTempDir("exec-verify-test-errormetrics");
    writeFileSync(join(tempDir, "tsconfig.json"), JSON.stringify({
      compilerOptions: { strict: true, noEmit: true },
      include: ["**/*.ts"],
    }));

    const task = makeTask();
    const output = [
      "```typescript",
      "const x: number = 'bad';",
      "```",
    ].join("\n");

    const result = await execVerifier.verify(output, task, {
      workingDir: tempDir,
    });

    expect(result.metrics?.compiled).toBe(0);
    expect(result.metrics?.errors).toBeGreaterThan(0);

    cleanupTempDir(tempDir);
  });
});
