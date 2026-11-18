# TikTok Wizard Workspace Rules

All active engineering directives, silicon execution guidelines, and system invariants (`INV-QUOTA-01`, `INV-QUOTA-02`, `INV-CODE-SOP-01`, `INV-VERIFY-01`, `INV-RELEASE-01`) are codified as the Single Source of Truth in [AGENTS.md](file:///C:/Users/User/Desktop/TikTokWizard/AGENTS.md).

<!-- tiktok-wizard-rules:start -->
# TikTok Wizard Release & Versioning Policy

For EVERY new change, fix, or feature in TikTok Wizard, you MUST ALWAYS:
1. **Increment the Version Number** sequentially (e.g. 2.0 -> 2.1 -> 2.2 -> 2.3...).
2. **Update Script Top Bar / Header**:
   - In `DO NOT TOUCH/src/cli.py`, update `print_header("TikTok Wizard vX.X")`.
3. **Update Documentation & Badges**:
   - In `README.md`, update `# 🪄 TIKTOK WIZARD vX.X 🪄` and the release badge link.
4. **Push & Tag on GitHub**:
   - Commit changes with message describing update.
   - Tag the commit: `git tag -a vX.X -m "TikTok Wizard vX.X - <summary>"`
   - Push to GitHub: `git push origin main --tags`
   - Create GitHub release: `gh release create vX.X --title "TikTok Wizard vX.X" --notes "<notes>"`
5. **Sync Mirrors**:
   - Keep `W:\TikTokWizard`, `C:\Users\User\Desktop\TikTokWizard`, and `C:\Users\User\Desktop\TikTokWizard_Repo` in complete sync.
<!-- tiktok-wizard-rules:end -->
