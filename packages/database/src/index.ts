import { config } from "dotenv";
import { resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";

// Load DATABASE_URL from this package's .env (or fall back to monorepo root)
config({ path: resolve(dirname(fileURLToPath(import.meta.url)), "..", ".env") });

export { PrismaClient } from '@prisma/client'
export type * from '@prisma/client'
