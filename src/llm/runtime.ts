export function getRequiredStringEnv(name: string): string {
  const value = process.env[name];
  if (!value || value.trim().length === 0) {
    throw new Error(
      `${name} must be defined in host.config.cjs (shared defaults) or host.config.local.cjs (machine-specific overrides). ` +
        `If this is a fresh clone, copy host.config.local.cjs.example to host.config.local.cjs and fill in the local values.`,
    );
  }

  return value;
}

export function getRequiredStringListEnv(name: string): string[] {
  const value = getRequiredStringEnv(name);
  const values = value
    .split(',')
    .map((item) => item.trim())
    .filter((item) => item.length > 0);

  if (values.length === 0) {
    throw new Error(`${name} must contain at least one value in host.config.cjs.`);
  }

  return values;
}

export function getRequiredBooleanEnv(name: string): boolean {
  const value = getRequiredStringEnv(name).toLowerCase();
  if (['1', 'true', 'yes', 'on'].includes(value)) {
    return true;
  }
  if (['0', 'false', 'no', 'off'].includes(value)) {
    return false;
  }

  throw new Error(`${name} must be a boolean value in host.config.cjs.`);
}

export function getRequiredPositiveIntegerEnv(name: string): number {
  const raw = getRequiredStringEnv(name);
  const parsed = Number(raw);
  if (!Number.isFinite(parsed) || parsed <= 0 || !Number.isInteger(parsed)) {
    throw new Error(
      `${name} must be a positive integer in host.config.cjs. Received "${raw}".`,
    );
  }
  return parsed;
}

export function getRequiredPositiveNumberEnv(name: string): number {
  const raw = getRequiredStringEnv(name);
  const parsed = Number(raw);
  if (!Number.isFinite(parsed) || parsed <= 0) {
    throw new Error(
      `${name} must be a positive number in host.config.cjs. Received "${raw}".`,
    );
  }
  return parsed;
}

export function getRequiredFractionEnv(name: string): number {
  const raw = getRequiredStringEnv(name);
  const parsed = Number(raw);
  if (!Number.isFinite(parsed) || parsed <= 0 || parsed >= 1) {
    throw new Error(
      `${name} must be a number strictly between 0 and 1 in host.config.cjs. Received "${raw}".`,
    );
  }
  return parsed;
}

export function parseKeepAlive(value: string): string | number {
  const numeric = Number(value);
  return Number.isFinite(numeric) ? numeric : value;
}

export function formatKeepAlive(value: string | number): string {
  return typeof value === 'number' ? String(value) : value;
}
