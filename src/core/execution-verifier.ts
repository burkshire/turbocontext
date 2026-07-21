/**
 * execution-verifier.ts — Execution-based code verification.
 *
 * Wraps the existing CodeVerifier with an additional compilation layer.
 * When a working directory is available AND the project has a tsconfig.json,
 * generated code is written to a temp directory and compiled with `tsc --noEmit`.
 * The compilation result provides a much stronger signal than structural checks alone.
 *
 * Architecture (Decorator Pattern):
 *   ExecutionCodeVerifier.verify(output, task, context)
 *     ├── Step 1: CodeVerifier.verify(output, task)          [structural, always]
 *     ├── Step 2: if no workingDir → return structural result [backward compat]
 *     ├── Step 3: detectProjectType → if no tsconfig → return structural result
 *     ├── Step 4: extractAndWriteCodeBlocks → compileProject
 *     └── Step 5: return composite result (execution overrides structural)
 */

import type { Task } from "../types.js";
import type { Verifier, VerificationResult } from "./verifier.js";
import { CodeVerifier } from "./verifier.js";
import {
  type ProcessRunner,
  type ProjectConfig,
  cleanupTempDir,
  compileProject,
  createTempDir,
  detectProjectType,
  extractAndWriteCodeBlocks,
  smokeTestTypeScript,
  type TestResult,
} from "./project-compiler.js";

// ------------------------------------------------------------------
// ExecutionCodeVerifier
// ------------------------------------------------------------------

export class ExecutionCodeVerifier implements Verifier {
  private structuralVerifier: CodeVerifier;
  private processRunner?: ProcessRunner;
  private timeoutMs: number;

  constructor(processRunner?: ProcessRunner, timeoutMs: number = 30_000) {
    this.structuralVerifier = new CodeVerifier();
    this.processRunner = processRunner;
    this.timeoutMs = timeoutMs;
  }

  async verify(
    output: string,
    task: Task,
    context?: { workingDir?: string; sourceFiles?: string[] },
  ): Promise<VerificationResult> {
    // Step 1: Always run structural verification first
    const structural = await this.structuralVerifier.verify(output, task, context);

    // Step 2: Skip execution verification if no working directory
    if (!context?.workingDir) {
      return structural;
    }

    // Step 3: Detect project type
    let projectConfig: ProjectConfig;
    try {
      projectConfig = detectProjectType(context.workingDir);
    } catch {
      return structural;
    }

    // Step 4: Only TypeScript projects get compilation verification (for now)
    if (projectConfig.projectType !== "typescript" || !projectConfig.hasConfigFile) {
      return structural;
    }

    // Step 5: Extract code blocks
    let tempDir: string | undefined;
    try {
      tempDir = createTempDir("exec-verify");

      const entries = extractAndWriteCodeBlocks(output, tempDir);

      if (entries.length === 0) {
        // No code blocks extracted — LLM didn't produce code in the expected format
        // This is already caught by the structural verifier, so just return that
        cleanupTempDir(tempDir);
        return structural;
      }

      // Check that at least one entry is TypeScript
      const tsEntries = entries.filter((e) =>
        ["typescript", "ts", "tsx"].includes(e.language),
      );

      if (tsEntries.length === 0) {
        // Non-TS code blocks only — structural check is our best signal
        cleanupTempDir(tempDir);
        return structural;
      }

      // Step 6: Run compilation
      const compileResult = await compileProject(
        tempDir,
        projectConfig.projectType,
        this.timeoutMs,
        this.processRunner,
      );

      // Step 7: If compilation passed, run runtime smoke test
      let testResult: TestResult | undefined;
      if (compileResult.compiled && projectConfig.projectType === "typescript") {
        try {
          testResult = await smokeTestTypeScript(
            tempDir,
            15_000,
            this.processRunner,
          );
        } catch {
          // Smoke test failure is non-fatal — still use compilation result
        }
      }

      // Step 8: Build composite result (includes test metrics if available)
      const executionResult = buildExecutionResult(
        structural,
        compileResult,
        testResult,
        projectConfig,
      );

      cleanupTempDir(tempDir);
      return executionResult;
    } catch (err) {
      // Verifier itself crashed — clean up and return crash signal
      if (tempDir) cleanupTempDir(tempDir);

      return {
        passed: false,
        crashed: true,
        hardSignal: -1.0,
        details: `Execution verifier crashed: ${(err as Error).message}`,
        metrics: {
          verifierCrashed: 1,
          projectType: projectConfig?.projectType === "typescript" ? 1 : 0,
        },
      };
    }
  }
}

// ------------------------------------------------------------------
// Result construction
// ------------------------------------------------------------------

/**
 * Build the final VerificationResult by combining structural checks,
 * compilation results, and optional runtime smoke test results.
 * Signal precedence: smoke test > compilation > structural.
 */
function buildExecutionResult(
  structural: VerificationResult,
  compileResult: { compiled: boolean; errors: string[]; warnings: string[]; exitCode: number | null; durationMs: number; command?: string },
  testResult: TestResult | undefined,
  projectConfig: ProjectConfig,
): VerificationResult {
  const compiled = compileResult.compiled;
  const tested = testResult?.passed === true;
  const functionsCalled = testResult?.functionsCalled ?? 0;

  // Build detail lines and metrics
  const detailParts: string[] = [];
  const metrics: Record<string, number> = {
    projectType: 1, // typescript
    structuralPassed: structural.passed ? 1 : 0,
  };

  if (compiled) {
    detailParts.push(`Compilation passed (${compileResult.durationMs}ms)`);
    if (compileResult.warnings.length > 0) {
      detailParts.push(`${compileResult.warnings.length} compiler warning(s)`);
    }
    metrics.compiled = 1;
    metrics.exitCode = compileResult.exitCode ?? 0;
    metrics.errors = 0;
    metrics.warnings = compileResult.warnings.length;
    metrics.durationMs = compileResult.durationMs;

    // Smoke test results
    if (testResult) {
      if (tested) {
        detailParts.push(
          `Smoke test passed: ${functionsCalled} function(s) called, 0 failed (${testResult.durationMs}ms)`,
        );
        metrics.testsPassed = 1;
        metrics.functionsCalled = functionsCalled;
      } else if (testResult.functionsFailed > 0) {
        detailParts.push(
          `Smoke test FAILED: ${testResult.functionsFailed}/${functionsCalled} function(s) threw`,
        );
        testResult.errors.slice(0, 3).forEach((e) => detailParts.push(e));
        metrics.testsPassed = 0;
        metrics.functionsCalled = functionsCalled;
        metrics.functionsFailed = testResult.functionsFailed;
      }
    }
  } else {
    // Compilation failed
    const isVerifierCrash =
      compileResult.exitCode === null &&
      compileResult.errors.some((e) => e.includes("command not found"));

    if (isVerifierCrash) {
      return {
        passed: false,
        crashed: true,
        hardSignal: -1.0,
        details: `Compiler not available: ${compileResult.errors.join("; ")}`,
        metrics: {
          compiled: 0,
          exitCode: -1,
          errors: compileResult.errors.length,
          warnings: 0,
          durationMs: compileResult.durationMs,
          projectType: 1,
          verifierCrashed: 1,
        },
      };
    }

    detailParts.push(`Compilation failed: ${compileResult.errors.length} error(s)`);
    compileResult.errors.slice(0, 5).forEach((e) => detailParts.push(e));
    metrics.compiled = 0;
    metrics.exitCode = compileResult.exitCode ?? 1;
    metrics.errors = compileResult.errors.length;
    metrics.warnings = compileResult.warnings.length;
    metrics.durationMs = compileResult.durationMs;
  }

  detailParts.push(`Structural: ${structural.details}`);

  // Hard signal: test > compile > structural
  let hardSignal: number;
  if (tested) {
    hardSignal = 1.0; // Compilation + smoke test both passed
  } else if (compiled) {
    // Compilation passed but smoke test failed or wasn't run
    hardSignal = testResult && !tested ? -0.3 : 0.8;
  } else {
    hardSignal = -0.5; // Compilation failed
  }

  return {
    passed: hardSignal >= 0.3,
    crashed: false,
    hardSignal: Math.round(hardSignal * 10000) / 10000,
    details: detailParts.filter(Boolean).join("; "),
    metrics,
  };
}
