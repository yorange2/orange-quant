import Parser from "rss-parser";

const parser = new Parser({
  timeout: 30_000, // 30s timeout per feed
  maxRedirects: 3,
  headers: {
    "User-Agent": "OrangeQuant-NewsIngestor/0.1",
  },
});

export interface ParsedItem {
  title: string;
  url: string;
  summary: string | null;
  publishedAt: Date;
}

export interface FetchResult {
  sourceName: string;
  items: ParsedItem[];
  error?: string;
}

/**
 * Fetch and parse an RSS feed, returning normalized items.
 * Never throws — errors are returned in the result.
 */
export async function fetchFeed(
  feedUrl: string,
  sourceName: string,
): Promise<FetchResult> {
  try {
    const feed = await parser.parseURL(feedUrl);

    if (!feed.items || feed.items.length === 0) {
      return { sourceName, items: [] };
    }

    const items: ParsedItem[] = [];

    for (const item of feed.items) {
      // Skip items without a link — we deduplicate on URL
      const url = item.link?.trim();
      if (!url) continue;

      const title = item.title?.trim() ?? "Untitled";
      const summary = item.contentSnippet?.trim() ?? item.summary?.trim() ?? null;

      // Some RSS feeds use isoDate, others use pubDate
      const rawDate = item.isoDate ?? item.pubDate;
      const publishedAt = rawDate ? new Date(rawDate) : new Date();

      items.push({ title, url, summary, publishedAt });
    }

    return { sourceName, items };
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    return { sourceName, items: [], error: message };
  }
}
