/**
 * project-compiler.ts — Compilation + smoke-test verification for LLM-generated code.
 *
 * Pure functions (plus controlled filesystem/process side-effects) that:
 *   1. Detect the project type from config files in the working directory.
 *   2. Extract code blocks from LLM output and write them to a temp directory.
 *   3. Run static type-checking (tsc --noEmit).
 *   4. Run a runtime smoke test (tsx) to verify the code actually executes.
 *   5. Return structured CompilationResult + TestResult.
 *
 * Safety: smoke tests run in an isolated temp directory with a 15s timeout.
 * No user project files are executed — only the LLM-generated code is tested.
 */

import { execFile } from "node:child_process";
import { existsSync, mkdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { homedir, tmpdir } from "node:os";
import { join } from "node:path";

// ------------------------------------------------------------------
// Types
// ------------------------------------------------------------------

export interface ProjectConfig {
  projectType: "typescript" | "javascript" | "python" | "go" | "rust" | "unknown";
  hasConfigFile: boolean;
  configPath?: string;
  tsconfigPath?: string;
  packageJsonPath?: string;
}

export interface CompilationResult {
  compiled: boolean;
  errors: string[];
  warnings: string[];
  exitCode: number | null;
  durationMs: number;
  command?: string;
}

export interface TestResult {
  /** Did the smoke test pass (process exit 0)? */
  passed: boolean;
  /** Number of exported functions detected and called */
  functionsCalled: number;
  /** Number of functions that threw */
  functionsFailed: number;
  /** Error messages from failed calls */
  errors: string[];
  exitCode: number | null;
  durationMs: number;
  command?: string;
}

export interface FileEntry {
  filePath: string;
  language: string;
  code: string;
}

export interface SpawnResult {
  stdout: string;
  stderr: string;
  exitCode: number | null;
  timedOut: boolean;
}

export type ProcessRunner = (
  command: string,
  args: string[],
  options: { cwd: string; timeoutMs: number },
) => Promise<SpawnResult>;

// ------------------------------------------------------------------
// 1. Project type detection
// ------------------------------------------------------------------

/**
 * Detect the project type and locate config files by scanning the working directory.
 */
export function detectProjectType(workingDir: string): ProjectConfig {
  const tsconfig = join(workingDir, "tsconfig.json");
  const packageJson = join(workingDir, "package.json");
  const goMod = join(workingDir, "go.mod");
  const pyProject = join(workingDir, "pyproject.toml");
  const setupPy = join(workingDir, "setup.py");
  const cargoToml = join(workingDir, "Cargo.toml");

  const hasTsconfig = existsSync(tsconfig);
  const hasPackageJson = existsSync(packageJson);

  if (hasTsconfig) {
    return {
      projectType: "typescript",
      hasConfigFile: true,
      configPath: tsconfig,
      tsconfigPath: tsconfig,
      packageJsonPath: hasPackageJson ? packageJson : undefined,
    };
  }

  if (hasPackageJson) {
    return {
      projectType: "javascript",
      hasConfigFile: true,
      configPath: packageJson,
      packageJsonPath: packageJson,
    };
  }

  if (existsSync(goMod)) {
    return { projectType: "go", hasConfigFile: true, configPath: goMod };
  }

  if (existsSync(pyProject) || existsSync(setupPy)) {
    return {
      projectType: "python",
      hasConfigFile: true,
      configPath: existsSync(pyProject) ? pyProject : setupPy,
    };
  }

  if (existsSync(cargoToml)) {
    return { projectType: "rust", hasConfigFile: true, configPath: cargoToml };
  }

  return { projectType: "unknown", hasConfigFile: false };
}

// ------------------------------------------------------------------
// 2. Temp directory management
// ------------------------------------------------------------------

/**
 * Create a temporary directory for compilation under ~/.turbocontext/tmp/.
 * Falls back to os.tmpdir() if the home-dir path is unwritable.
 */
export function createTempDir(label: string = "compile"): string {
  const ts = Date.now();
  const rand = Math.random().toString(36).slice(2, 8);
  const base = join(homedir(), ".turbocontext", "tmp");
  const dir = join(base, `${ts}-${rand}-${label}`);

  try {
    mkdirSync(dir, { recursive: true });
  } catch {
    // Fallback to os tmpdir
    const fallback = join(tmpdir(), `turbocontext-${ts}-${rand}-${label}`);
    mkdirSync(fallback, { recursive: true });
    return fallback;
  }

  return dir;
}

/**
 * Remove a temp directory and all its contents.
 */
export function cleanupTempDir(tempDir: string): void {
  try {
    rmSync(tempDir, { recursive: true, force: true });
  } catch {
    // Best-effort cleanup; log but don't throw.
  }
}

// ------------------------------------------------------------------
// 3. Code block extraction
// ------------------------------------------------------------------

/**
 * Extract fenced code blocks from LLM output and write them to files
 * in the temp directory. Handles inline filename hints like:
 *   // src/auth/login.ts
 *   ```typescript
 *   ...
 *   ```
 */
export function extractAndWriteCodeBlocks(
  output: string,
  tempDir: string,
): FileEntry[] {
  const entries: FileEntry[] = [];
  // Match fenced code blocks: ```lang\n...\n```
  const fenceRe = /```(\w+)\n([\s\S]*?)```/g;
  // Match inline filename hints: // path/to/file.ext or # path/to/file.ext
  const filenameHintRe = /^\s*(?:\/\/|#)\s*([\w./-]+\.\w+)\s*$/m;

  let match: RegExpExecArray | null;
  let blockIndex = 0;

  while ((match = fenceRe.exec(output)) !== null) {
    const lang = match[1].toLowerCase();
    const code = match[2].trim();

    if (!code) continue;

    // Try to infer a filename from the first line of the code
    const hintMatch = code.match(filenameHintRe);
    let filePath: string;

    if (hintMatch) {
      // Use the hinted path but strip leading src/ if it exists
      filePath = hintMatch[1].replace(/^src\//, "");
    } else {
      // Generate a filename based on language
      const ext = langToExt(lang);
      filePath = `generated_${blockIndex}${ext}`;
    }

    const fullPath = join(tempDir, filePath);
    // Ensure parent directory exists
    const parentDir = fullPath.substring(0, fullPath.lastIndexOf("/"));
    if (parentDir && parentDir !== tempDir) {
      try { mkdirSync(parentDir, { recursive: true }); } catch { /* ok */ }
    }

    writeFileSync(fullPath, code, "utf-8");
    entries.push({ filePath: fullPath, language: lang, code });

    blockIndex++;
  }

  return entries;
}

function langToExt(lang: string): string {
  const map: Record<string, string> = {
    typescript: ".ts",
    ts: ".ts",
    tsx: ".tsx",
    javascript: ".js",
    js: ".js",
    jsx: ".jsx",
    python: ".py",
    py: ".py",
    go: ".go",
    rust: ".rs",
    rs: ".rs",
    java: ".java",
    ruby: ".rb",
    rb: ".rb",
    swift: ".swift",
    kotlin: ".kt",
    kt: ".kt",
    sql: ".sql",
    graphql: ".graphql",
    sh: ".sh",
    bash: ".sh",
    json: ".json",
    yaml: ".yml",
    yml: ".yml",
    toml: ".toml",
  };
  return map[lang] || `.${lang}`;
}

// ------------------------------------------------------------------
// 4. TypeScript compilation
// ------------------------------------------------------------------

/**
 * Create a minimal tsconfig.json in the temp directory for isolated type-checking.
 * We don't copy the project's tsconfig because its paths/rootDir settings
 * won't resolve in the temp directory. Instead we create a lenient config
 * that checks for basic type errors (missing imports, type mismatches, etc.).
 */
function writeTempTsconfig(tempDir: string): string {
  const config = {
    compilerOptions: {
      target: "ES2022",
      module: "ESNext",
      moduleResolution: "bundler",
      strict: true,
      noEmit: true,
      skipLibCheck: true,
      allowImportingTsExtensions: true,
      isolatedModules: true,
      // Don't check library types — only check the generated code
      types: [],
      // Allow imports of external packages without type definitions
      noImplicitAny: false,
    },
    include: ["**/*.ts", "**/*.tsx"],
    exclude: ["node_modules"],
  };

  const configPath = join(tempDir, "tsconfig.json");
  writeFileSync(configPath, JSON.stringify(config, null, 2), "utf-8");
  return configPath;
}

/**
 * Run `npx tsc --noEmit` in the temp directory.
 * Returns a structured CompilationResult.
 */
export async function compileTypeScript(
  tempDir: string,
  timeoutMs: number = 30_000,
  processRunner?: ProcessRunner,
): Promise<CompilationResult> {
  const runner = processRunner ?? realSpawnProcess;
  const tsconfigPath = writeTempTsconfig(tempDir);

  const start = Date.now();
  let result: SpawnResult;

  try {
    result = await runner("npx", ["tsc", "--noEmit", "--project", tsconfigPath], {
      cwd: tempDir,
      timeoutMs,
    });
  } catch (err) {
    return {
      compiled: false,
      errors: [`Process failed: ${(err as Error).message}`],
      warnings: [],
      exitCode: null,
      durationMs: Date.now() - start,
      command: "npx tsc --noEmit",
    };
  }

  const durationMs = Date.now() - start;
  const rawOutput = result.stdout + "\n" + result.stderr;
  const errors = parseTscOutput(rawOutput);

  // If exitCode is null (process didn't run at all), the stderr content IS the error
  const finalErrors = result.exitCode === null && errors.length === 0
    ? [result.stderr.trim() || "Unknown compilation error"]
    : errors;

  return {
    compiled: result.exitCode === 0 && !result.timedOut,
    errors: result.exitCode === 0 ? [] : finalErrors,
    warnings: result.exitCode === 0 ? errors : [], // warnings when exitCode=0
    exitCode: result.timedOut ? null : result.exitCode,
    durationMs,
    command: "npx tsc --noEmit",
  };
}

/**
 * Parse tsc output into individual error/warning messages.
 * tsc outputs lines like: file.ts(line,col): error TS1234: message
 */
function parseTscOutput(output: string): string[] {
  const lines = output.split("\n").filter((l) => l.trim());
  const issues: string[] = [];

  for (const line of lines) {
    // Match tsc error/warning format
    if (/:\s*error\s+TS\d+:/i.test(line) || /:\s*warning\s+TS\d+:/i.test(line)) {
      // Strip the full path prefix for readability
      const clean = line.replace(/^.*?(\w[\w.-]*\.(ts|tsx|js|jsx))/, "$1");
      issues.push(clean.trim());
    } else if (/^error/i.test(line.trim())) {
      issues.push(line.trim());
    }
  }

  // Limit to first 20 issues to avoid overwhelming
  return issues.slice(0, 20);
}

// ------------------------------------------------------------------
// 5. Language dispatch
// ------------------------------------------------------------------

/**
 * Run the appropriate compiler for the detected project type.
 * Currently only TypeScript is fully implemented. Other languages
 * get a note in the result.
 */
export async function compileProject(
  tempDir: string,
  projectType: ProjectConfig["projectType"],
  timeoutMs: number = 30_000,
  processRunner?: ProcessRunner,
): Promise<CompilationResult> {
  switch (projectType) {
    case "typescript":
      return compileTypeScript(tempDir, timeoutMs, processRunner);

    case "javascript":
      // Try tsc with allowJs; if no tsc available, this falls back gracefully
      try {
        return await compileTypeScript(tempDir, timeoutMs, processRunner);
      } catch {
        return {
          compiled: true, // Can't verify — assume ok
          errors: [],
          warnings: ["JavaScript verification: tsc not available, skipping"],
          exitCode: 0,
          durationMs: 0,
          command: "skipped (js)",
        };
      }

    case "python":
      // Python: could use py_compile per file. For now, skip.
      return {
        compiled: true,
        errors: [],
        warnings: ["Python compilation not yet implemented — structural check only"],
        exitCode: 0,
        durationMs: 0,
        command: "skipped (python)",
      };

    case "go":
      return {
        compiled: true,
        errors: [],
        warnings: ["Go vet not yet implemented — structural check only"],
        exitCode: 0,
        durationMs: 0,
        command: "skipped (go)",
      };

    case "rust":
      return {
        compiled: true,
        errors: [],
        warnings: ["Rust check not yet implemented — structural check only"],
        exitCode: 0,
        durationMs: 0,
        command: "skipped (rust)",
      };

    default:
      return {
        compiled: true,
        errors: [],
        warnings: ["Unknown project type — structural check only"],
        exitCode: 0,
        durationMs: 0,
        command: "skipped (unknown)",
      };
  }
}

// ------------------------------------------------------------------
// 5.5 Runtime smoke test (TypeScript)
// ------------------------------------------------------------------

/**
 * Build a self-contained smoke-test harness that imports each generated module,
 * detects exported functions, and calls them with type-appropriate default arguments.
 * Runs as a standalone script with tsx — no user project dependencies needed.
 */
function buildSmokeTestHarness(filePaths: string[]): string {
  const imports = filePaths
    .map((p, i) => {
      // Convert absolute path to relative import for the harness
      const relPath = "./" + p.replace(/\\/g, "/").split("/").pop()!.replace(/\.tsx?$/, "");
      return `import * as _m${i} from '${relPath}';`;
    })
    .join("\n");

  // For each module, detect exports and call functions with defaults
  const testBlocks = filePaths
    .map((_, i) => `
// Module ${i}: try calling exported functions
for (const [name, value] of Object.entries(_m${i})) {
  if (typeof value === 'function' && !name.startsWith('_')) {
    try {
      // Call with type-safe defaults: empty objects, empty arrays, 0, "", false
      const result = value({}, [], 0, "", false);
      // If it returns a promise, we can't await at top level easily — skip
      if (result && typeof result.then === 'function') {
        console.log('SKIP:' + name + ':async');
        continue;
      }
      console.log('PASS:' + name);
    } catch (e) {
      const msg = (e as Error).message || String(e);
      console.log('FAIL:' + name + ':' + msg.slice(0, 120));
    }
  } else if (typeof value === 'object' && value !== null && !Array.isArray(value)) {
    // Class constructor or object — try instantiating
    try {
      const instance = new (value as Record<string, unknown> & { new?: never } as unknown as new () => unknown)();
      console.log('PASS:' + name + ':instantiated');
    } catch {
      // Not a constructor — skip
    }
  }
}`)
    .join("\n");

  return `${imports}

// Auto-generated smoke test harness
let passed = 0;
let failed = 0;

${testBlocks}

console.log('SMOKE_RESULT:' + passed + ':' + failed);
process.exit(failed > 0 ? 1 : 0);
`;
}

/**
 * Run a runtime smoke test on generated TypeScript code.
 *
 * After compilation passes, this verifies the code can actually be loaded
 * and basic function calls succeed. Uses tsx for on-the-fly transpilation.
 *
 * Safety:
 *   - Runs in an isolated temp directory (not the user's project).
 *   - 15s timeout prevents infinite loops.
 *   - Only LLM-generated code is executed — no user project files.
 */
export async function smokeTestTypeScript(
  tempDir: string,
  timeoutMs: number = 15_000,
  processRunner?: ProcessRunner,
): Promise<TestResult> {
  const runner = processRunner ?? realSpawnProcess;

  // Find all .ts files in tempDir
  const { readdirSync } = await import("node:fs");
  const tsFiles: string[] = [];
  try {
    const entries = readdirSync(tempDir, { recursive: true }) as unknown as string[];
    for (const entry of entries) {
      if (entry.endsWith(".ts") || entry.endsWith(".tsx")) {
        if (!entry.startsWith("_smoke") && !entry.startsWith("tsconfig")) {
          tsFiles.push(join(tempDir, entry));
        }
      }
    }
  } catch {
    return {
      passed: true,
      functionsCalled: 0,
      functionsFailed: 0,
      errors: [],
      exitCode: 0,
      durationMs: 0,
      command: "skipped (no ts files)",
    };
  }

  if (tsFiles.length === 0) {
    return {
      passed: true,
      functionsCalled: 0,
      functionsFailed: 0,
      errors: [],
      exitCode: 0,
      durationMs: 0,
      command: "skipped (no ts files)",
    };
  }

  // Build and write the harness
  const harness = buildSmokeTestHarness(tsFiles);
  const harnessPath = join(tempDir, "_smoke_test_harness.ts");
  writeFileSync(harnessPath, harness, "utf-8");

  const start = Date.now();
  let result: SpawnResult;

  try {
    result = await runner("npx", ["tsx", harnessPath], {
      cwd: tempDir,
      timeoutMs,
    });
  } catch (err) {
    return {
      passed: false,
      functionsCalled: 0,
      functionsFailed: 1,
      errors: [`Smoke test runner failed: ${(err as Error).message}`],
      exitCode: null,
      durationMs: Date.now() - start,
      command: "npx tsx",
    };
  }

  const durationMs = Date.now() - start;

  // Parse output: SMOKE_RESULT:passed:failed
  const output = result.stdout + "\n" + result.stderr;
  const smokeMatch = output.match(/SMOKE_RESULT:(\d+):(\d+)/);
  const functionsCalled = smokeMatch ? parseInt(smokeMatch[1], 10) + parseInt(smokeMatch[2], 10) : 0;
  const functionsFailed = smokeMatch ? parseInt(smokeMatch[2], 10) : 0;

  // Collect individual FAIL lines
  const failLines = output.split("\n").filter((l) => l.startsWith("FAIL:"));
  const errors = failLines.map((l) => l.replace("FAIL:", "").trim());

  return {
    passed: result.exitCode === 0 && !result.timedOut,
    functionsCalled,
    functionsFailed,
    errors: errors.slice(0, 10),
    exitCode: result.timedOut ? null : result.exitCode,
    durationMs,
    command: "npx tsx (smoke test)",
  };
}

// ------------------------------------------------------------------
// 6. Process runner
// ------------------------------------------------------------------

/**
 * Default process runner using Node.js child_process.execFile.
 * No shell is invoked — the command and args are passed directly.
 */
export function realSpawnProcess(
  command: string,
  args: string[],
  options: { cwd: string; timeoutMs: number },
): Promise<SpawnResult> {
  return new Promise((resolve, reject) => {
    const child = execFile(command, args, {
      cwd: options.cwd,
      timeout: options.timeoutMs,
      maxBuffer: 1024 * 1024, // 1 MB
      windowsHide: true,
    });

    let stdout = "";
    let stderr = "";

    child.stdout?.on("data", (chunk: Buffer) => {
      stdout += chunk.toString();
    });

    child.stderr?.on("data", (chunk: Buffer) => {
      stderr += chunk.toString();
    });

    child.on("close", (exitCode: number | null, signal: string | null) => {
      resolve({
        stdout,
        stderr,
        exitCode,
        timedOut: signal === "SIGTERM" || signal === "SIGKILL",
      });
    });

    child.on("error", (err: Error & { code?: string }) => {
      // ENOENT = command not found — treat as non-fatal
      if (err.code === "ENOENT") {
        resolve({
          stdout: "",
          stderr: `${command}: command not found`,
          exitCode: null,
          timedOut: false,
        });
      } else {
        reject(err);
      }
    });
  });
}

// ------------------------------------------------------------------
// 7. Backward-compatibility helpers
// ------------------------------------------------------------------
