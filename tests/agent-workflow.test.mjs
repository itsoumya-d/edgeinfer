// SPDX-License-Identifier: MIT
import { test, describe } from 'node:test';
import assert from 'node:assert/strict';
import { join, dirname } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const {
  AgentWorkflowEngine,
  runAgentWorkflow,
  AgentWorkflowError,
} = await import(pathToFileURL(join(__dirname, '../dist/index.mjs')).href);

describe('agentic workflow execution engine', () => {
  test('agent workflow normal execution executes multi-step pipeline and returns state', async () => {
    const report = await runAgentWorkflow({
      name: 'code-review-pipeline',
      initialContext: { input: 'def foo(): pass', reviewed: false, fixes: [] },
      steps: [
        {
          id: 'lint',
          name: 'Syntax and Linter',
          run: async (ctx) => ({ fixes: [...ctx.fixes, 'formatted'] }),
        },
        {
          id: 'review',
          name: 'Security Evaluation',
          run: async (ctx) => ({ reviewed: true }),
        },
      ],
      leverageEvaluator: (ctx, steps) => ({
        score: 0.92,
        highLeverage: true,
        rationale: 'Fully automated review complete',
      }),
    });

    assert.equal(report.success, true);
    assert.equal(report.finalContext.reviewed, true);
    assert.deepEqual(report.finalContext.fixes, ['formatted']);
    assert.equal(report.steps.length, 2);
    assert.equal(report.steps[0].stepId, 'lint');
    assert.equal(report.steps[1].stepId, 'review');
    assert.equal(report.leverage?.highLeverage, true);
    assert.equal(report.leverage?.score, 0.92);
  });

  test('agent workflow empty steps throws EMPTY_STEPS error', async () => {
    await assert.rejects(
      async () => {
        await runAgentWorkflow({
          name: 'empty-pipeline',
          initialContext: {},
          steps: [],
        });
      },
      (err) => {
        assert.ok(err instanceof AgentWorkflowError);
        assert.equal(err.code, 'EMPTY_STEPS');
        return true;
      }
    );
  });

  test('agent workflow malformed step configuration throws MALFORMED_STEP error', async () => {
    await assert.rejects(
      async () => {
        await runAgentWorkflow({
          name: 'malformed-step-pipeline',
          initialContext: {},
          steps: [{ id: '', run: null }],
        });
      },
      (err) => {
        assert.ok(err instanceof AgentWorkflowError);
        assert.equal(err.code, 'MALFORMED_STEP');
        return true;
      }
    );
  });

  test('agent workflow boundary conditions zero timeout and skipped condition handled', async () => {
    const report = await runAgentWorkflow({
      name: 'boundary-pipeline',
      initialContext: { count: 0, flag: false },
      steps: [
        {
          id: 'skip-me',
          name: 'Conditional Step',
          condition: (ctx) => ctx.flag === true,
          run: async () => ({ count: 999 }),
        },
        {
          id: 'zero-retries-zero-timeout',
          maxRetries: 0,
          timeoutMs: 0, // defaults to 60000ms boundary
          run: async (ctx) => ({ count: ctx.count + 1 }),
        },
      ],
    });

    assert.equal(report.success, true);
    assert.equal(report.finalContext.count, 1);
    assert.equal(report.steps.length, 1);
    assert.equal(report.steps[0].stepId, 'zero-retries-zero-timeout');
  });

  test('agent workflow missing tool dependency or failure triggers retry and throws MAX_RETRIES_EXCEEDED error', async () => {
    let callCount = 0;
    await assert.rejects(
      async () => {
        await runAgentWorkflow({
          name: 'retry-exhaustion-pipeline',
          initialContext: {},
          steps: [
            {
              id: 'flake-service',
              maxRetries: 2,
              run: async () => {
                callCount++;
                throw new Error('missing connection to external tool dependency');
              },
            },
          ],
        });
      },
      (err) => {
        assert.ok(err instanceof AgentWorkflowError);
        assert.equal(err.code, 'MAX_RETRIES_EXCEEDED');
        assert.equal(err.stepId, 'flake-service');
        assert.equal(callCount, 3);
        return true;
      }
    );
  });

  test('agent workflow invariant property holds across all successful step transitions', async () => {
    const initial = { balance: 100, log: [] };
    const report = await runAgentWorkflow({
      name: 'invariant-check',
      initialContext: initial,
      steps: [
        {
          id: 'debit',
          run: async (ctx) => ({ balance: ctx.balance - 30, log: [...ctx.log, 'debit 30'] }),
        },
        {
          id: 'credit',
          run: async (ctx) => ({ balance: ctx.balance + 10, log: [...ctx.log, 'credit 10'] }),
        },
      ],
    });

    assert.equal(report.finalContext.balance, 80);
    assert.equal(initial.balance, 100); // immutable initial context preserved
    assert.equal(report.steps.every((s) => s.success && s.durationMs >= 0), true);
  });

  test('agent workflow step timeout aborts and throws STEP_TIMEOUT error', async () => {
    await assert.rejects(
      async () => {
        await runAgentWorkflow({
          name: 'timeout-pipeline',
          initialContext: {},
          steps: [
            {
              id: 'hanging-step',
              timeoutMs: 30,
              maxRetries: 0,
              run: async () => new Promise((resolve) => setTimeout(resolve, 200)),
            },
          ],
        });
      },
      (err) => {
        assert.ok(err instanceof AgentWorkflowError);
        assert.equal(err.code, 'STEP_TIMEOUT');
        assert.equal(err.stepId, 'hanging-step');
        return true;
      }
    );
  });

  test('agent workflow leverage evaluation failure throws LEVERAGE_EVALUATION_FAILED error', async () => {
    await assert.rejects(
      async () => {
        await runAgentWorkflow({
          name: 'leverage-fail-pipeline',
          initialContext: {},
          steps: [
            {
              id: 'step1',
              run: async () => ({ done: true }),
            },
          ],
          leverageEvaluator: () => {
            throw new Error('evaluator exploded');
          },
        });
      },
      (err) => {
        assert.ok(err instanceof AgentWorkflowError);
        assert.equal(err.code, 'LEVERAGE_EVALUATION_FAILED');
        return true;
      }
    );
  });
});
