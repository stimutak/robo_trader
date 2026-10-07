"""Regression checks for the pinned automated review workflow."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _workflow() -> str:
    return (ROOT / ".github/workflows/claude-code-review.yml").read_text(encoding="utf-8")


def test_review_action_uses_pinned_v1_release_and_prompt() -> None:
    workflow = _workflow()
    assert "anthropics/claude-code-action@58985842b834ed26087302ba27d07bc24ca8697a" in workflow
    assert "anthropics/claude-code-action@beta" not in workflow
    assert "direct_prompt:" not in workflow
    assert "prompt: |" in workflow
    assert "REPO: ${{ github.repository }}" in workflow
    assert "PR NUMBER: ${{ github.event.pull_request.number }}" in workflow
    assert "track_progress: true" in workflow


def test_review_preserves_auth_and_fork_boundary() -> None:
    workflow = _workflow()
    assert "claude_code_oauth_token: ${{ secrets.CLAUDE_CODE_OAUTH_TOKEN }}" in workflow
    assert "github.event.pull_request.head.repo.full_name == github.repository" in workflow
    assert "github_token:" not in workflow
    assert "anthropic_api_key:" not in workflow
    assert "contents: write" not in workflow
    assert "pull_request_target:" not in workflow


def test_review_cannot_silently_pass_without_execution() -> None:
    workflow = _workflow()
    assert "Require completed review execution" in workflow
    assert "REVIEW_CONCLUSION: ${{ steps.claude-review.outputs.conclusion }}" in workflow
    assert 'if [[ "$REVIEW_CONCLUSION" != "success" ]]; then' in workflow
    assert "workflow-validation skip or authentication failure" in workflow
    assert "exit 1" in workflow
