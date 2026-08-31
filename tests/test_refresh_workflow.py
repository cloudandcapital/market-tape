import unittest
from pathlib import Path


ROOT = Path(__file__).parents[1]


class RefreshWorkflowTests(unittest.TestCase):
    def test_workflow_publishes_to_remote_default_branch_via_retry_helper(self):
        workflow = (ROOT / ".github/workflows/refresh_data.yml").read_text()
        helper = (ROOT / "scripts/refresh_and_publish.sh").read_text()

        self.assertIn("github.event.repository.default_branch", workflow)
        self.assertIn('refresh_and_publish.sh "$TARGET_BRANCH"', workflow)
        self.assertIn('git checkout --force -B refresh-data "origin/$target_branch"', helper)
        self.assertIn('git rebase "origin/$target_branch"', helper)
        self.assertIn("git rebase --abort", helper)
        self.assertIn("rebuild_from_latest", helper)
        self.assertNotIn("git push --force", helper)

    def test_schedule_avoids_top_and_half_hour_load_windows(self):
        workflow = (ROOT / ".github/workflows/refresh_data.yml").read_text()

        self.assertIn('cron: "17,47 13-21 * * 1-5"', workflow)
        self.assertNotIn('cron: "*/30', workflow)


if __name__ == "__main__":
    unittest.main()
