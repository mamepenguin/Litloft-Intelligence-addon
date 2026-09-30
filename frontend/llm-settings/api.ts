export const LLM_ENDPOINT = "/api/addons/intelligence/admin/llm";
export const EXPOSURE_ENDPOINT = "/api/addons/intelligence/admin/llm/exposure";

export interface ProfileView {
  provider?: string;
  base_url?: string;
  model?: string;
  vision_model?: string;
  offhost?: boolean;
  agentic?: boolean;
  api_key_env?: string;
  api_key_present?: boolean;
  [key: string]: unknown;
}

export interface RoutingView {
  default?: string;
  local_fallback?: string | null;
  features?: Record<string, string>;
  [key: string]: unknown;
}

export interface LLMView {
  profiles: Record<string, ProfileView>;
  routing: RoutingView;
  legacy: boolean;
  error: string | null;
  output_language: string;
  output_language_restart_pending: boolean;
  features: string[];
  available_providers: string[];
  available_output_languages: string[];
  overrides_present: boolean;
}

export interface LLMUpdateBody {
  profiles: Record<string, ProfileView>;
  routing: RoutingView;
  output_language?: string;
}

export interface SaveResult {
  status: string;
  restart_required: boolean;
  core_notified?: string;
}

export type Destination = "sends" | "falls_back" | "skips" | "unknown";

export interface FeatureExposure {
  profile: string | null;
  offhost: boolean;
  drives?: Record<string, Destination>;
}

export interface ExposureView {
  features: Record<string, FeatureExposure>;
  local_fallback: string | null;
}

export class RequestError extends Error {
  readonly status: number;

  constructor(status: number, detail: string) {
    super(detail);
    this.status = status;
  }
}

async function detailOf(resp: Response): Promise<string> {
  try {
    const body = await resp.json();
    if (typeof body?.detail === "string") return body.detail;
  } catch {
    // A body that is not JSON carries no detail worth showing.
  }
  return `HTTP ${resp.status}`;
}

async function request<T>(url: string, init?: RequestInit): Promise<T> {
  const resp = await fetch(url, init);
  if (!resp.ok) throw new RequestError(resp.status, await detailOf(resp));
  return (await resp.json()) as T;
}

export function fetchLLM(): Promise<LLMView> {
  return request<LLMView>(LLM_ENDPOINT, { method: "GET" });
}

export function fetchExposure(): Promise<ExposureView> {
  return request<ExposureView>(EXPOSURE_ENDPOINT, { method: "GET" });
}

export function saveLLM(body: LLMUpdateBody): Promise<SaveResult> {
  return request<SaveResult>(LLM_ENDPOINT, {
    method: "PUT",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body),
  });
}

export function resetLLM(): Promise<SaveResult> {
  return request<SaveResult>(LLM_ENDPOINT, { method: "DELETE" });
}

export function errorDetail(err: unknown, fallback: string): string {
  return err instanceof Error && err.message ? err.message : fallback;
}
