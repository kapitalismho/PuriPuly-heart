import type { Context } from 'hono';

import type { BrokerEnv } from './contract';

import { BROKER_PUBLIC_INPUT_BOUNDS } from './persistence';

const CONTROL_CHARACTER_PATTERN = /\p{Cc}/u;
const NEWLINE_PATTERN = /[\r\n\u0085\u2028\u2029]/u;
const countCharacters = (value: string): number => Array.from(value).length;
const isWhitespaceOnly = (value: string): boolean => value.trim().length === 0;

export function stringValue(value: unknown): string | null {
  return typeof value === 'string' ? value : null;
}

export function validatePublicInput(
  field: keyof typeof BROKER_PUBLIC_INPUT_BOUNDS,
  value: string,
): string | null {
  const bounds = BROKER_PUBLIC_INPUT_BOUNDS[field];

  if (bounds.rejectWhitespaceOnly && isWhitespaceOnly(value)) {
    return `${field} must not be blank or whitespace-only`;
  }

  const characterCount = countCharacters(value);

  if (characterCount < bounds.minLength || characterCount > bounds.maxLength) {
    return `${field} must be between ${bounds.minLength} and ${bounds.maxLength} characters`;
  }

  const hasDisallowedControlOrNewline =
    (bounds.rejectControlCharacters && CONTROL_CHARACTER_PATTERN.test(value)) ||
    (bounds.rejectNewlines && NEWLINE_PATTERN.test(value));

  if (hasDisallowedControlOrNewline) {
    return `${field} must not contain control characters or newlines`;
  }

  return null;
}

export function nonEmptyString(value: unknown): string | null {
  return typeof value === 'string' && value.trim().length > 0 ? value : null;
}

export async function readJsonBody<T = Record<string, unknown>>(
  c: Context<BrokerEnv>,
): Promise<
  | { ok: true; value: T }
  | { ok: false; reason: 'invalid_json' | 'not_object' }
> {
  try {
    const value = await c.req.json();
    if (typeof value !== 'object' || value === null || Array.isArray(value)) {
      return { ok: false, reason: 'not_object' };
    }

    return { ok: true, value: value as T };
  } catch {
    return { ok: false, reason: 'invalid_json' };
  }
}

const STRICT_ISO_8601_TIMESTAMP =
  /^(?<year>\d{4})-(?<month>0[1-9]|1[0-2])-(?<day>0[1-9]|[12]\d|3[01])T(?<hour>[01]\d|2[0-3]):(?<minute>[0-5]\d):(?<second>[0-5]\d)(?:\.(?<millisecond>\d{3}))?(?:(?<utc>Z)|(?<offsetSign>[+-])(?<offsetHour>[01]\d|2[0-3]):(?<offsetMinute>[0-5]\d))$/u;

export function parseIsoDate(value: string): Date | null {
  const match = STRICT_ISO_8601_TIMESTAMP.exec(value);
  if (!match?.groups) {
    return null;
  }

  const year = Number(match.groups.year);
  const month = Number(match.groups.month);
  const day = Number(match.groups.day);
  const hour = Number(match.groups.hour);
  const minute = Number(match.groups.minute);
  const second = Number(match.groups.second);
  const millisecond = Number(match.groups.millisecond ?? '0');
  const offsetMinutes = match.groups.utc
    ? 0
    : (match.groups.offsetSign === '-' ? -1 : 1) *
      (Number(match.groups.offsetHour) * 60 + Number(match.groups.offsetMinute));

  const timestamp =
    Date.UTC(year, month - 1, day, hour, minute, second, millisecond) -
    offsetMinutes * 60_000;
  const reconstructedLocalTime = new Date(timestamp + offsetMinutes * 60_000);

  if (
    reconstructedLocalTime.getUTCFullYear() !== year ||
    reconstructedLocalTime.getUTCMonth() + 1 !== month ||
    reconstructedLocalTime.getUTCDate() !== day ||
    reconstructedLocalTime.getUTCHours() !== hour ||
    reconstructedLocalTime.getUTCMinutes() !== minute ||
    reconstructedLocalTime.getUTCSeconds() !== second ||
    reconstructedLocalTime.getUTCMilliseconds() !== millisecond
  ) {
    return null;
  }

  return new Date(timestamp);
}
