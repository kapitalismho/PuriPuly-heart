import { readdirSync, readFileSync } from 'node:fs';
import { DatabaseSync } from 'node:sqlite';

const MIGRATIONS_DIR = new URL('../../migrations/', import.meta.url);

export const BROKER_MIGRATION_FILENAMES = readdirSync(MIGRATIONS_DIR)
  .filter((name: string) => name.endsWith('.sql'))
  .sort();

export function readBrokerMigrationSql(fileName: string): string {
  return readFileSync(new URL(fileName, MIGRATIONS_DIR), 'utf8');
}

export function applyBrokerMigrations(
  db: DatabaseSync,
  options: {
    through?: string;
    after?: string;
  } = {},
): void {
  const startIndex = options.after
    ? BROKER_MIGRATION_FILENAMES.indexOf(options.after) + 1
    : 0;
  const endIndex = options.through
    ? BROKER_MIGRATION_FILENAMES.indexOf(options.through) + 1
    : BROKER_MIGRATION_FILENAMES.length;

  if (options.after && startIndex === 0) {
    throw new Error(`unknown migration: ${options.after}`);
  }

  if (options.through && endIndex === 0) {
    throw new Error(`unknown migration: ${options.through}`);
  }

  for (const fileName of BROKER_MIGRATION_FILENAMES.slice(startIndex, endIndex)) {
    db.exec(readBrokerMigrationSql(fileName));
  }
}
