export function getStringEnv(name: string, fallback: string): string {
  const value = process.env[name];
  return value && value.trim().length > 0 ? value : fallback;
}

export function getNumberEnv(name: string, fallback: number): number {
  const value = process.env[name];
  if (!value) {
    return fallback;
  }

  const parsed = Number(value);
  return Number.isFinite(parsed) && parsed > 0 ? parsed : fallback;
}

export function getIntegerEnv(name: string, fallback: number): number {
  return Math.trunc(getNumberEnv(name, fallback));
}

export function getBooleanEnv(name: string, fallback: boolean): boolean {
  const value = process.env[name];
  if (value === undefined) {
    return fallback;
  }

  return ['1', 'true', 'yes', 'on'].includes(value.toLowerCase());
}

export function parseKeepAlive(value: string): string | number {
  const numeric = Number(value);
  return Number.isFinite(numeric) ? numeric : value;
}

export function formatKeepAlive(value: string | number): string {
  return typeof value === 'number' ? String(value) : value;
}
