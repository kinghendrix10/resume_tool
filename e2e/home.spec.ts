import { expect, test } from "@playwright/test";

test("home loads workspace title", async ({ page }) => {
  await page.goto("/");
  await expect(page.getByText("Resume intelligence")).toBeVisible();
  await expect(page.getByText("Candidate analysis")).toBeVisible();
});
