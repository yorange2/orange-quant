import { PrismaClient } from "@orange-quant/database";
import { fetchFeed } from "./rss.js";

const prisma = new PrismaClient();

const DEFAULT_INTERVAL_MINUTES = 5;
const FEED_FETCH_DELAY_MS = 2_000; // 2s pause between feeds (be a good citizen)

async function ingestAll(): Promise<void> {
  const sources = await prisma.source.findMany({
    where: { active: true },
  });

  if (sources.length === 0) {
    console.warn("⚠ No active sources found. Run `pnpm seed` to add default sources.");
    return;
  }

  console.log(`📡 Fetching ${sources.length} feed(s)...`);

  let totalNew = 0;

  for (const source of sources) {
    const result = await fetchFeed(source.feedUrl, source.name);

    if (result.error) {
      console.error(`  ✗ ${source.name}: ${result.error}`);
      continue;
    }

    let newCount = 0;
    let skippedCount = 0;

    for (const item of result.items) {
      try {
        await prisma.newsItem.create({
          data: {
            title: item.title,
            url: item.url,
            summary: item.summary,
            publishedAt: item.publishedAt,
            sourceId: source.id,
          },
        });
        newCount++;
      } catch (err: unknown) {
        // Prisma unique constraint violation — duplicate URL
        if (
          typeof err === "object" &&
          err !== null &&
          "code" in err &&
          (err as { code: string }).code === "P2002"
        ) {
          skippedCount++;
        } else {
          const message = err instanceof Error ? err.message : String(err);
          console.error(`  ✗ DB error for "${item.title}": ${message}`);
        }
      }
    }

    const symbol = result.error ? "✗" : "✓";
    console.log(
      `  ${symbol} ${source.name}: ${newCount} new, ${skippedCount} skipped`,
    );

    totalNew += newCount;

    // Small delay between feeds to avoid hammering servers
    if (sources.length > 1) {
      await sleep(FEED_FETCH_DELAY_MS);
    }
  }

  if (totalNew > 0) {
    console.log(`✅ Saved ${totalNew} new item(s)`);
  } else {
    console.log("💤 No new items");
  }
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function main(): Promise<void> {
  const args = process.argv.slice(2);
  const once = args.includes("--once");

  console.log("📰 Orange Quant — News Ingestor");
  console.log(`   Mode: ${once ? "single-run" : `loop (every ${DEFAULT_INTERVAL_MINUTES}m)`}`);
  console.log("");

  // eslint-disable-next-line no-constant-condition
  while (true) {
    const startTime = Date.now();

    try {
      await ingestAll();
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      console.error(`❌ Fatal error in ingest cycle: ${message}`);
    }

    if (once) break;

    const elapsed = Date.now() - startTime;
    const waitMs = Math.max(0, DEFAULT_INTERVAL_MINUTES * 60_000 - elapsed);

    const nextRun = new Date(Date.now() + waitMs);
    console.log(
      `⏳ Next fetch at ${nextRun.toLocaleTimeString()} (in ${Math.round(waitMs / 1000)}s)`,
    );
    console.log("");

    await sleep(waitMs);
  }

  await prisma.$disconnect();
}

main().catch(async (err) => {
  console.error("Fatal startup error:", err);
  await prisma.$disconnect();
  process.exit(1);
});
