import { PrismaClient } from "@orange-quant/database";

const prisma = new PrismaClient();

const DEFAULT_SOURCES = [
  {
    name: "WSJ Markets",
    feedUrl: "https://feeds.a.dj.com/rss/RSSMarketsMain.xml",
    type: "rss",
  },
  {
    name: "Bloomberg Markets",
    feedUrl: "https://feeds.bloomberg.com/markets/news.rss",
    type: "rss",
  },
];

async function seed(): Promise<void> {
  console.log("🌱 Seeding news sources...");

  let added = 0;
  let skipped = 0;

  for (const src of DEFAULT_SOURCES) {
    const existing = await prisma.source.findFirst({
      where: { feedUrl: src.feedUrl },
    });

    if (existing) {
      console.log(`  → ${src.name} already exists — skipping`);
      skipped++;
      continue;
    }

    await prisma.source.create({ data: src });
    console.log(`  ✓ ${src.name} added`);
    added++;
  }

  console.log(`\n✅ Done: ${added} added, ${skipped} skipped`);
  await prisma.$disconnect();
}

seed().catch(async (err) => {
  console.error("Seed failed:", err);
  await prisma.$disconnect();
  process.exit(1);
});
