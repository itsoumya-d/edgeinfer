// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2026 Soumya Debnath
//
// Structured agentic workflow engine for edge inference pipelines.
// Coordinates multi-stage agent workflows across planning, execution,
// tool dispatching, and quality evaluation with timeout and retry gates.

export type AgentWorkflowErrorCode =
  | 'EMPTY_STEPS'
  | 'MALFORMED_STEP'
  | 'STEP_TIMEOUT'
  | 'TOOL_EXECUTION_FAILED'
  | 'MAX_RETRIES_EXCEEDED'
  | 'LEVERAGE_EVALUATION_FAILED';

export class AgentWorkflowError extends Error {
  readonly code: AgentWorkflowErrorCode;
  readonly stepId?: string;
  readonly cause?: unknown;

  constructor(
    code: AgentWorkflowErrorCode,
    message: string,
    stepId?: string,
    options?: { cause?: unknown }
  ) {
    super(`[${code}] ${message}${stepId ? ` (step: ${stepId})` : ''}`);
    this.name = 'AgentWorkflowError';
    this.code = code;
    this.stepId = stepId;
    if (options && 'cause' in options) {
      this.cause = options.cause;
    }
  }
}

export interface WorkflowStepResult<TContext> {
  stepId: string;
  name: string;
  durationMs: number;
  retries: number;
  success: boolean;
  delta?: Partial<TContext>;
}

export interface WorkflowStepConfig<TContext> {
  id: string;
  name?: string;
  run: (context: Readonly<TContext>) => Promise<Partial<TContext> | void>;
  maxRetries?: number;
  timeoutMs?: number;
  condition?: (context: Readonly<TContext>) => boolean;
}

export interface LeverageScoreReport {
  score: number;
  highLeverage: boolean;
  rationale: string;
}

export interface AgentWorkflowOptions<TContext> {
  name: string;
  steps: WorkflowStepConfig<TContext>[];
  initialContext: TContext;
  leverageEvaluator?: (
    context: Readonly<TContext>,
    stepHistory: WorkflowStepResult<TContext>[]
  ) => LeverageScoreReport;
}

export interface AgentWorkflowExecutionReport<TContext> {
  workflowName: string;
  success: boolean;
  finalContext: TContext;
  totalDurationMs: number;
  steps: WorkflowStepResult<TContext>[];
  leverage?: LeverageScoreReport;
}

export class AgentWorkflowEngine<TContext extends Record<string, unknown>> {
  async execute(
    options: AgentWorkflowOptions<TContext>
  ): Promise<AgentWorkflowExecutionReport<TContext>> {
    if (!options || !Array.isArray(options.steps) || options.steps.length === 0) {
      throw new AgentWorkflowError('EMPTY_STEPS', 'Workflow must contain at least one step');
    }

    const startTime = Date.now();
    let currentContext = { ...options.initialContext };
    const stepResults: WorkflowStepResult<TContext>[] = [];

    for (const step of options.steps) {
      if (!step || typeof step.id !== 'string' || !step.id.trim() || typeof step.run !== 'function') {
        throw new AgentWorkflowError(
          'MALFORMED_STEP',
          'Step configuration must have a valid id and run function',
          step?.id
        );
      }

      if (step.condition && !step.condition(currentContext)) {
        continue;
      }

      const maxRetries =
        typeof step.maxRetries === 'number' && step.maxRetries >= 0 ? step.maxRetries : 0;
      const timeoutMs =
        typeof step.timeoutMs === 'number' && step.timeoutMs > 0 ? step.timeoutMs : 60000;

      let attempt = 0;
      let stepSuccess = false;
      let lastError: unknown = null;
      const stepStart = Date.now();

      while (attempt <= maxRetries) {
        try {
          const delta = await this.runWithTimeout(
            () => step.run(Object.freeze({ ...currentContext })),
            timeoutMs,
            step.id
          );

          if (delta && typeof delta === 'object') {
            currentContext = { ...currentContext, ...delta };
          }

          stepResults.push({
            stepId: step.id,
            name: step.name || step.id,
            durationMs: Math.max(0, Date.now() - stepStart),
            retries: attempt,
            success: true,
            delta: delta || undefined,
          });

          stepSuccess = true;
          break;
        } catch (err) {
          lastError = err;
          attempt++;
        }
      }

      if (!stepSuccess) {
        if (lastError instanceof AgentWorkflowError) {
          throw lastError;
        }
        throw new AgentWorkflowError(
          'MAX_RETRIES_EXCEEDED',
          `Step failed after ${maxRetries + 1} attempt(s): ${
            lastError instanceof Error ? lastError.message : String(lastError)
          }`,
          step.id,
          { cause: lastError }
        );
      }
    }

    let leverageReport: LeverageScoreReport | undefined;
    if (options.leverageEvaluator) {
      try {
        leverageReport = options.leverageEvaluator(
          Object.freeze({ ...currentContext }),
          stepResults
        );
      } catch (err) {
        throw new AgentWorkflowError(
          'LEVERAGE_EVALUATION_FAILED',
          `Leverage evaluation failed: ${err instanceof Error ? err.message : String(err)}`,
          undefined,
          { cause: err }
        );
      }
    }

    return {
      workflowName: options.name || 'unnamed-workflow',
      success: true,
      finalContext: currentContext,
      totalDurationMs: Math.max(0, Date.now() - startTime),
      steps: stepResults,
      leverage: leverageReport,
    };
  }

  private async runWithTimeout<T>(
    fn: () => Promise<T>,
    timeoutMs: number,
    stepId: string
  ): Promise<T> {
    let timer: ReturnType<typeof setTimeout> | null = null;
    const timeoutPromise = new Promise<never>((_, reject) => {
      timer = setTimeout(() => {
        reject(
          new AgentWorkflowError(
            'STEP_TIMEOUT',
            `Execution exceeded timeout of ${timeoutMs}ms`,
            stepId
          )
        );
      }, timeoutMs);
    });

    try {
      return await Promise.race([fn(), timeoutPromise]);
    } finally {
      if (timer) clearTimeout(timer);
    }
  }
}

export async function runAgentWorkflow<TContext extends Record<string, unknown>>(
  options: AgentWorkflowOptions<TContext>
): Promise<AgentWorkflowExecutionReport<TContext>> {
  const engine = new AgentWorkflowEngine<TContext>();
  return engine.execute(options);
}
