import { chromium } from 'playwright'
import { z } from 'zod'
import { openai } from '@ai-sdk/openai'
import LLMScraper from 'llm-scraper'

// Launch a browser instance
const browser = await chromium.launch()

// Initialize LLM provider with a model that has larger context
const llm = openai.chat('gpt-4.1')

// Create a new LLMScraper
const scraper = new LLMScraper(llm)

// Open new page
const page = await browser.newPage()
await page.goto('https://s2025.conference-schedule.org')

// Define schema to extract contents into
const schema = z.object({
  top: z
    .array(
      z.object({
        session_presentation_title: z.string(),
        date: z.string(),
        time: z.string(),
        type: z.string(),
      })
    )
    .length(5)
    .describe('Top 50 sessions on Siggrahph 2025 conference'),
})

// Run the scraper with content limiting options
const { data } = await scraper.run(page, schema, {
  format: 'html'
})

// Show the result from LLM
console.log(data.top)

await page.close()
await browser.close()